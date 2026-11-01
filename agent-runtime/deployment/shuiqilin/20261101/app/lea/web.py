"""Browser-style tools: search, next five results, and read a selected page."""
from copy import deepcopy
from datetime import datetime, timezone
from html.parser import HTMLParser
import ipaddress
import re
import threading
import time
import uuid
from urllib.parse import parse_qs, urlencode, urljoin, urlsplit, urlunsplit

import httpx


def clean(value):
    return re.sub(r"\s+", " ", value).strip()


def public_url(value):
    try:
        p = urlsplit(value)
        host = (p.hostname or "").lower().rstrip(".")
        if (p.scheme not in ("http", "https") or not host or "." not in host
                or p.username or p.password or p.port not in (None, 80, 443)
                or any(ord(c) < 32 for c in value)
                or host.endswith((".local", ".localhost", ".internal", ".test"))):
            return None
        try:
            if not ipaddress.ip_address(host).is_global:
                return None
        except ValueError:
            pass
        # Engine-generated answers, redirects and ads are not external sources.
        if host == "so.com" or host.endswith(".so.com"):
            return None
        return value
    except ValueError:
        return None


class Node:
    def __init__(self, tag="", attrs=()):
        self.tag, self.attrs, self.children = tag, dict(attrs), []

    def find(self, predicate):
        for child in self.children:
            if isinstance(child, Node):
                if predicate(child):
                    yield child
                yield from child.find(predicate)

    def text(self):
        if (self.tag in {"script", "style", "nav", "noscript"}
                or "hidden" in self.attrs or self.attrs.get("aria-hidden") == "true"
                or re.search(r"display\s*:\s*none|visibility\s*:\s*hidden", self.attrs.get("style", ""), re.I)):
            return ""
        return "".join(c.text() if isinstance(c, Node) else c for c in self.children)


class SearchPage(HTMLParser):
    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = Node()
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        node = Node(tag, attrs)
        self.stack[-1].children.append(node)
        if tag not in self.VOID:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in self.VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag == tag:
                del self.stack[i:]
                break

    def handle_data(self, data):
        self.stack[-1].children.append(data)


def parse_search_page(html, query):
    page = SearchPage()
    page.feed(html)
    page.close()
    title = next(page.root.find(lambda n: n.tag == "title"), None)
    if not title or clean(query).casefold() not in clean(title.text()).casefold():
        raise ValueError("Search page did not identify the requested query")
    results, seen = [], set()
    for item in page.root.find(lambda n: n.tag == "li" and "res-list" in n.attrs.get("class", "").split()
                               and n.attrs.get("data-result-type") in {"normal", "richsum"}):
        heading = next(item.find(lambda n: n.tag == "h3"), None)
        if not heading:
            continue
        link = next(heading.find(lambda n: n.tag == "a"), None)
        if not link:
            continue
        url = public_url(link.attrs.get("data-mdurl") or link.attrs.get("href", ""))
        text = clean(heading.text())
        if not url or url in seen or not text:
            continue
        preview = next(item.find(lambda n: "res-desc" in n.attrs.get("class", "").split()
                                 or "data-result-summary" in n.attrs), None)
        results.append(dict(id="S" + str(len(results) + 1), title=text[:160], url=url,
                            snippet=clean(preview.text())[:240] if preview else "",
                            page_read=False, accessibility="not_checked"))
        seen.add(url)
    next_url = None
    for link in page.root.find(lambda n: n.tag == 'a' and '下一页' in clean(n.text())):
        target = urljoin('https://www.so.com/s', link.attrs.get('href', ''))
        p = urlsplit(target)
        if (p.scheme == 'https' and p.hostname == 'www.so.com' and p.port in (None, 443)
                and not p.username and not p.password and p.path == '/s'
                and parse_qs(p.query).get('q') == [query]):
            next_url = target
            break
    return results, next_url


def parse_results(html, query, limit=5):
    return parse_search_page(html, query)[0][:limit]


def canonical_url(value):
    p = urlsplit(value)
    # Anchor variants and scheme variants should not consume another result slot.
    return urlunsplit(('', p.netloc.lower(), p.path or '/', p.query, ''))


class WebClient:
    def __init__(self, *, transport=None):
        self.transport, self.cache, self.lock = transport, {}, threading.Lock()
        self.sessions, self.references, self.page_cache = {}, {}, {}

    def _fetch_search(self, url, query):
        tick = time.monotonic()
        with httpx.Client(transport=self.transport, trust_env=False, follow_redirects=False,
                          timeout=8, headers={'User-Agent': 'LocalEvolvingAgent/0.1 (read-only web search)',
                          'Accept': 'text/html', 'Accept-Encoding': 'identity'}) as client:
            for redirects in range(3):
                p = urlsplit(url)
                if (p.scheme != 'https' or p.hostname != 'www.so.com' or p.port not in (None, 443)
                        or p.username or p.password or p.path != '/s'
                        or parse_qs(p.query).get('q') != [query]):
                    raise ValueError('Search URL left its endpoint or query')
                with client.stream('GET', url) as response:
                    if response.status_code in (301, 302, 303, 307, 308):
                        if redirects == 2:
                            raise ValueError('Too many search redirects')
                        url = urljoin(url, response.headers.get('location', ''))
                        continue
                    response.raise_for_status()
                    if 'text/html' not in response.headers.get('content-type', '').lower():
                        raise ValueError('Expected search HTML')
                    chunks, size = [], 0
                    for chunk in response.iter_bytes(16384):
                        size += len(chunk)
                        if size > 2_000_000 or time.monotonic() - tick > 12:
                            raise ValueError('Search response exceeded limits')
                        chunks.append(chunk)
                    matches, following = parse_search_page(b''.join(chunks).decode('utf-8', 'replace'), query)
                    return matches, following, url
        raise ValueError('No search response')

    def _view(self, sid, offset):
        tick = time.monotonic()
        session = self.sessions[sid]
        key = offset
        if key in session['views']:
            return {**deepcopy(session['views'][key]), 'cached': True, 'seconds': 0.0}
        result = dict(query=session['query'], search_id=sid, status='unavailable', retrieved=False,
                      provider='360搜索', results=[], next_cursor=None, has_more=False,
                      content_kind='search_results_only', pages_opened=False, untrusted_web_content=True,
                      fetched_at=datetime.now(timezone.utc).isoformat(), cached=False, offset=offset,
                      message='仅返回搜索候选及摘要，未打开或验证候选网页。')
        try:
            for _ in range(3):
                if len(session['results']) >= offset + 5 or not session['next_url']:
                    break
                if len(session['engine_pages']) >= 20:
                    session['next_url'] = None
                    break
                target = session['next_url']
                if target in session['visited']:
                    session['next_url'] = None
                    break
                rows, following, final_url = self._fetch_search(target, session['query'])
                session['visited'].add(target)
                session['engine_pages'].append(final_url)
                session['next_url'] = following
                for row in rows:
                    canon = canonical_url(row['url'])
                    if canon not in session['seen']:
                        session['seen'].add(canon)
                        session['results'].append(row)
            matches = deepcopy(session['results'][offset:offset+5])
            for rank, row in enumerate(matches, offset+1):
                row['id'], row['rank'] = f'{sid}:{rank}', rank
                self.references[row['id']] = row['url']
            more = len(session['results']) > offset+len(matches) or bool(session['next_url'])
            # Cache each displayed batch, making retrying its cursor idempotent.
            result.update(results=matches, retrieved=bool(matches), status='ok' if matches else 'end',
                          has_more=more, next_cursor=f'{sid}:{offset+len(matches)}' if more else None,
                          engine_pages=list(session['engine_pages']))
            result['seconds'] = round(time.monotonic()-tick, 3)
            if matches or not more:
                session['views'][key] = deepcopy(result)
        except (httpx.HTTPError, OSError, ValueError, RecursionError) as error:
            blocked = isinstance(error, httpx.HTTPStatusError) and error.response.status_code in (403, 429)
            result.update(status='blocked' if blocked else 'unavailable', error_kind=type(error).__name__,
                          message='本次未取得新的搜索候选。可重试同一游标，未跳过任何结果。',
                          retry_cursor=f'{sid}:{offset}')
        result['seconds'] = round(time.monotonic()-tick, 3)
        return result

    def search(self, query):
        if (not isinstance(query, str) or not query.strip() or len(query) > 200
                or any(ord(c) < 32 for c in query)):
            raise ValueError("query must be nonempty public search terms, at most 200 characters")
        query = query.strip()
        if re.search(r"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{24,}", query):
            raise ValueError("Do not send credentials in public search queries")
        with self.lock:
            for sid in list(self.sessions):
                if self.sessions[sid]['expires'] < time.monotonic():
                    del self.sessions[sid]
                    for ref in list(self.references):
                        if ref.startswith(sid+':'):
                            del self.references[ref]
            cached = self.cache.get(query)
            if cached and cached[0] > time.monotonic() and cached[1]['search_id'] in self.sessions:
                return {**deepcopy(cached[1]), 'cached': True}
            if len(self.sessions) >= 64:
                oldest = next(iter(self.sessions))
                del self.sessions[oldest]
                for ref in list(self.references):
                    if ref.startswith(oldest+':'):
                        del self.references[ref]
            sid = uuid.uuid4().hex[:12]
            url = "https://www.so.com/s?" + urlencode({"q": query})
            self.sessions[sid] = dict(query=query, expires=time.monotonic()+900, results=[], seen=set(),
                                      next_url=url, visited=set(), engine_pages=[], views={})
            result = self._view(sid, 0)
            result['search_url'] = url
            if len(self.cache) >= 128:
                self.cache.pop(next(iter(self.cache)))
            self.cache[query] = (time.monotonic() + (180 if result["retrieved"] else 15), result)
            return result

    def next_page(self, cursor):
        if not isinstance(cursor, str) or not re.fullmatch(r'[0-9a-f]{12}:\d{1,3}', cursor):
            raise ValueError('Use the next_cursor returned by search or next_page')
        sid, offset = cursor.split(':')
        with self.lock:
            session = self.sessions.get(sid)
            if not session or session['expires'] < time.monotonic():
                return dict(status='expired', retrieved=False, results=[], next_cursor=None,
                            message='搜索会话已过期，请重新搜索。')
            # Only issued cursors may advance the session.
            issued = {view.get('next_cursor') for view in session['views'].values()}
            if cursor not in issued and int(offset) != 0:
                raise ValueError('Cursor was not issued by this search session')
            return self._view(sid, int(offset))

    def open(self, url, query='', *, transport=None, resolver=None):
        from .web_page import read_page
        if not isinstance(url, str) or not isinstance(query, str):
            raise ValueError('Expected a result ID/public URL and an optional focus string')
        target = self.references.get(url, url) if isinstance(url, str) else url
        key = (target, query)
        with self.lock:
            old = self.page_cache.get(key)
            if old and old[0] > time.monotonic():
                return {**deepcopy(old[1]), 'cached': True, 'seconds': 0.0}
            result = read_page(target, query=query, transport=transport, resolver=resolver)
            if len(self.page_cache) >= 32:
                self.page_cache.pop(next(iter(self.page_cache)))
            self.page_cache[key] = (time.monotonic()+(180 if result['retrieved'] else 15), result)
            return result


_client = WebClient()


def search(query):
    """Return up to five public website candidates for one search query."""
    return _client.search(query)


def next_page(cursor):
    """Return the next five distinct candidates; retrying the same cursor is safe."""
    return _client.next_page(cursor)


def open(url, query=''):
    """Read a selected result ID or public URL, optionally focusing on a term."""
    return _client.open(url, query=query)


if __name__ == "__main__":
    import argparse
    import json
    import sys
    parser = argparse.ArgumentParser(description="搜索公开网页，返回标题、网址与摘要。")
    parser.add_argument("query")
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(search(parser.parse_args().query), ensure_ascii=False))
