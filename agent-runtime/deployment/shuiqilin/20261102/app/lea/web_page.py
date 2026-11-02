"""Read bounded public HTML/text pages with pinned public DNS and TLS names."""
from datetime import datetime, timezone
import ipaddress
import re
import socket
import time
from urllib.parse import unquote, urljoin, urlsplit, urlunsplit

import httpx

from .web import SearchPage, clean, public_url


class PublicTransport(httpx.BaseTransport):
    """Pin the checked IP for each request, retaining original Host and TLS SNI."""
    def __init__(self, *, inner=None, resolver=None):
        self.inner = inner or httpx.HTTPTransport()
        self.resolver = resolver or socket.getaddrinfo

    def handle_request(self, request):
        if not public_url(str(request.url)):
            raise ValueError('Only public HTTP(S) pages are readable')
        host = request.url.host
        answers = self.resolver(host, request.url.port or (443 if request.url.scheme == 'https' else 80),
                                type=socket.SOCK_STREAM)
        addresses = list(dict.fromkeys(item[4][0] for item in sorted(answers, key=lambda a: a[0] != socket.AF_INET)))
        if not addresses or any(not ipaddress.ip_address(ip).is_global for ip in addresses):
            raise ValueError('Page hostname resolved outside the public Internet')
        pinned = httpx.Request(request.method, request.url.copy_with(host=addresses[0]),
                               headers=request.headers, stream=request.stream,
                               extensions={**request.extensions, 'sni_hostname': host})
        return self.inner.handle_request(pinned)

    def close(self):
        self.inner.close()


def extract_page(html, url):
    page = SearchPage()
    page.feed(html)
    page.close()
    title_node = next(page.root.find(lambda n: n.tag == 'title'), None)
    title = clean(title_node.text()) if title_node else ''
    if re.search(r'安全验证|访问验证|验证码|just a moment|access denied|verify.*human|captcha', title, re.I):
        return dict(status='blocked', retrieved=False, message='网页要求验证，未取得正文。')
    hidden_classes = set()
    for style in page.root.find(lambda n: n.tag == 'style'):
        raw = ''.join(c for c in style.children if isinstance(c, str))
        for selectors, body in re.findall(r'([^{}]+)\{([^{}]*)\}', raw):
            if re.search(r'display\s*:\s*none|visibility\s*:\s*hidden', body, re.I):
                for selector in selectors.split(','):
                    if re.fullmatch(r'\s*\.[a-zA-Z_][\w-]*\s*', selector):
                        hidden_classes.add(selector.strip()[1:])

    def skip(node):
        return (node.tag in {'script','style','nav','noscript','footer','header','aside','form','button','svg'}
                or 'hidden' in node.attrs or node.attrs.get('aria-hidden') == 'true'
                or node.attrs.get('role') in {'navigation', 'dialog'}
                or bool(set(node.attrs.get('class','').split()) & (hidden_classes | {'headerlink', 'sphinxsidebar'}))
                or bool(re.search(r'display\s*:\s*none|visibility\s*:\s*hidden', node.attrs.get('style',''), re.I)))

    def visible(node):
        if skip(node):
            return ''
        return ''.join(visible(c) if hasattr(c,'tag') else c for c in node.children)

    # Prefer semantic article/main roots to navigation and unrelated sidebars.
    roots = []
    for content_id in ('content_views', 'article_content'):
        roots = list(page.root.find(lambda n: n.attrs.get('id') == content_id))
        if roots:
            break
    if not roots:
        roots = list(page.root.find(lambda n: n.tag == 'article'
                                   or 'article-intro' in n.attrs.get('class','').split()))
    if not roots:
        roots = list(page.root.find(lambda n: n.tag == 'main' or n.attrs.get('role') == 'main'))
    if not roots:
        roots = list(page.root.find(lambda n: n.attrs.get('id') in {'content','js-card-content'}
                                   or bool(set(n.attrs.get('class','').split()) & {'article-intro','main-content-text'})))
    root = max(roots, key=lambda n: len(visible(n))) if roots else next(page.root.find(lambda n: n.tag == 'body'), page.root)
    fragment = unquote(urlsplit(url).fragment)
    if fragment:
        def locate(node, path):
            if node.attrs.get('id') == fragment:
                return path+[node]
            for child in node.children:
                if hasattr(child, 'tag'):
                    found = locate(child, path+[node])
                    if found:
                        return found
            return None
        path = locate(root, [])
        if path:
            root = next((n for n in reversed(path) if n.tag in {'section','article','dl'}), path[-1])
    code_blocks = []

    def render(node):
        if skip(node):
            return ''
        if node.tag == 'pre':
            code_blocks.append(visible(node).strip('\r\n'))
            return '\n\n@@CODE_BLOCK_' + str(len(code_blocks)-1) + '@@\n\n'
        text = ''.join(render(c) if hasattr(c,'tag') else c for c in node.children)
        if node.tag == 'br':
            return '\n'
        if node.tag in {'p','h1','h2','h3','h4','h5','h6','li','div','section','dl','dt','dd','tr','blockquote'}:
            return '\n\n'+text+'\n\n'
        if node.tag in {'td','th'}:
            return text+' | '
        return text

    text = render(root).replace('\r\n','\n')
    text = re.sub(r'[ \t\f\v]+', ' ', text)
    text = re.sub(r' *\n *', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    for number, code in enumerate(code_blocks):
        text = text.replace('@@CODE_BLOCK_'+str(number)+'@@', '```\n'+code+'\n```')
    if len(text) < 80:
        return dict(status='requires_browser', retrieved=False,
                    message='未取得足够正文，页面可能依赖JavaScript、登录或验证。')
    links, seen = [], set()
    for link in root.find(lambda n: n.tag == 'a'):
        target = public_url(urljoin(url, link.attrs.get('href','')))
        label = clean(visible(link))
        if target and target not in seen and label:
            links.append(dict(title=label[:100], url=target))
            seen.add(target)
        if len(links) == 12:
            break
    return dict(status='ok', retrieved=True, title=title, full_text=text, links=links,
                fragment_applied=bool(fragment and path))


def read_page(url, query='', *, transport=None, resolver=None):
    if not isinstance(url, str) or len(url) > 2000 or not public_url(url):
        raise ValueError('Use an issued result ID or a public HTTP(S) URL')
    if not isinstance(query, str) or len(query) > 120 or any(ord(c) < 32 for c in query):
        raise ValueError('Optional query must be a short focus term')
    tick = time.monotonic()
    result = dict(url=url, status='unavailable', retrieved=False, cached=False,
                  fetched_at=datetime.now(timezone.utc).isoformat(),
                  untrusted_web_content=True, content_kind='page_body_excerpt')
    current = url
    try:
        with httpx.Client(transport=PublicTransport(inner=transport, resolver=resolver), trust_env=False,
                          timeout=8, follow_redirects=False,
                          headers={'User-Agent':'LocalEvolvingAgent/0.1 (read-only page reader)',
                                   'Accept':'text/html,text/plain', 'Accept-Encoding':'identity'}) as client:
            for hop in range(5):
                if not public_url(current):
                    raise ValueError('Redirect target is not a public page')
                with client.stream('GET', current) as response:
                    result['http_status'] = response.status_code
                    if response.status_code in (301,302,303,307,308):
                        if hop == 4:
                            raise ValueError('Too many page redirects')
                        current = urljoin(current, response.headers.get('location',''))
                        continue
                    result['final_url'] = current
                    if response.status_code in (401,403,429):
                        result.update(status='blocked', message='网页限制访问、要求登录或验证，未取得正文。')
                        break
                    response.raise_for_status()
                    kind = response.headers.get('content-type','').lower()
                    if not any(t in kind for t in ('text/html','application/xhtml+xml','text/plain')):
                        result.update(status='unsupported', message='本工具只读取HTML或纯文本，不下载二进制文件。')
                        break
                    chunks, size = [], 0
                    for chunk in response.iter_bytes(16384):
                        size += len(chunk)
                        if size > 2_000_000 or time.monotonic()-tick > 15:
                            raise ValueError('Page response exceeded limits')
                        chunks.append(chunk)
                    encoding = re.search(r'charset=["\x27]?([\w-]+)', kind)
                    raw = b''.join(chunks)
                    if not encoding:
                        encoding = re.search(br'charset\s*=\s*["\x27]?([\w-]+)', raw[:4096], re.I)
                    codec = encoding.group(1) if encoding else 'utf-8'
                    if isinstance(codec, bytes):
                        codec = codec.decode('ascii')
                    body = raw.decode(codec, 'replace')
                    parsed = (dict(status='ok', retrieved=True, title='', full_text=body, links=[])
                              if 'text/plain' in kind else extract_page(body, current))
                    if not parsed['retrieved']:
                        result.update(parsed)
                        break
                    text = parsed.pop('full_text')
                    start = 0
                    if query:
                        pos = text.casefold().find(query.casefold())
                        if pos >= 0:
                            start = max(0, pos-600)
                            boundary = text.rfind('\n\n', 0, start)
                            if boundary >= max(0, start-1000):
                                start = boundary
                    excerpt = text[start:start+2400]
                    result.update(parsed, text=excerpt, body_chars=len(text), excerpt_chars=len(excerpt),
                                  excerpt_start=start, truncated=start>0 or start+len(excerpt)<len(text),
                                  focus=query, bytes_received=size)
                    break
    except (httpx.HTTPError, OSError, ValueError, LookupError, RecursionError) as error:
        result.update(status='unavailable', error_kind=type(error).__name__, error_detail=str(error)[:180], message='本次未能读取网页正文。')
    result['seconds'] = round(time.monotonic()-tick,3)
    return result
