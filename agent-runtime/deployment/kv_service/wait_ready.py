import json,time,urllib.request
for _ in range(120):
    try:
        with urllib.request.urlopen('http://127.0.0.1:18804/health',timeout=3) as f:
            r=json.load(f)
        if r.get('status')=='ok':break
    except OSError:pass
    time.sleep(1)
else:raise SystemExit('Native KV backend is not ready')
