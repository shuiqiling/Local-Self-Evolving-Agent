"""Activate the reviewed feature overlay after isolated real-model checks pass."""
from pathlib import Path
import hashlib,json,os,shutil,sqlite3,subprocess,sys,time,urllib.request

ROOT=Path('/home/ubuntu/AI-Native/embodied-agent')
WORK=Path(__file__).resolve().parent
UNITS=Path.home()/'.config/systemd/user'
NAMES=['config.py','model.py','runtime.py','native_kv.py']
REPORT=WORK/'activation.json'
report=dict(completed=False,phase='preflight',started_at=time.time())

def save():REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def system(*args):subprocess.run(['systemctl','--user',*args],check=True,timeout=150)
def active():
    with sqlite3.connect('file:'+str(ROOT/'data/memory.sqlite3')+'?mode=ro',uri=True) as db:
        return db.execute("SELECT id,status FROM jobs WHERE status IN ('queued','running')").fetchall()
def health():
    for _ in range(120):
        try:
            with urllib.request.urlopen('http://127.0.0.1:18800/api/health',timeout=5) as f:h=json.load(f)
            if h['model']['ready'] and h['model'].get('active_primary_backend')=='native_kvzip':return h
        except OSError:pass
        time.sleep(1)
    raise RuntimeError('Formal Agent native health deadline')

save();stopped=False
try:
    for name in ['stage-backend-test-report.json','stage-agent-test-report.json']:
        result=json.loads((WORK/name).read_text())
        assert result['completed'] and result['checks'] and all(result['checks'].values()),(name,result)
    manifest=json.loads((WORK/'release.json').read_text())
    for name,value in manifest['files'].items():assert sha(WORK/name)==value,('release changed',name)
    expected=json.loads((WORK/'baseline-sha256.json').read_text())
    for name in NAMES:
        original=ROOT/'app/lea'/name
        assert (sha(original) if original.exists() else None)==expected[name],('production changed',name)
    assert not active(),('Formal jobs active',active())
    unit=UNITS/'lea-native-kv.service';drop=UNITS/'lea-agent.service.d/zzzz-native-kv.conf'
    assert not unit.exists() and not drop.exists(),'Existing native activation must be inspected first'
    backup=ROOT/'deployments'/('native-kvzip-'+time.strftime('%Y%m%d-%H%M%S'))
    backup.mkdir(parents=True,exist_ok=False)
    for name in NAMES:
        p=ROOT/'app/lea'/name
        if p.exists():shutil.copy2(p,backup/name)
    (backup/'agent-unit.txt').write_text(subprocess.run(['systemctl','--user','cat','lea-agent.service'],capture_output=True,text=True,check=True).stdout)
    report.update(backup=str(backup),original_hashes=expected,installed_hashes={n:sha(WORK/'app/lea'/n) for n in NAMES},release_identity=manifest['identity'])
    report['phase']='quiesce';save()
    system('stop','lea-agent.service');stopped=True
    assert not active(),('A formal job arrived before stop; restarting original service',active())
    with sqlite3.connect(ROOT/'data/memory.sqlite3') as src,sqlite3.connect(backup/'memory-forensics.sqlite3') as dst:src.backup(dst)
    system('stop','lea-agent-kv-stage.service','lea-native-kv-stage.service')
    for name in NAMES:shutil.copy2(WORK/'app/lea'/name,ROOT/'app/lea'/name)
    report['phase']='install_service';save()
    unit.write_text(f'''[Unit]
Description=Primary Agent native Fast KVzip sessions
[Service]
NoNewPrivileges=true
UMask=0077
KillMode=control-group
Restart=on-failure
RestartSec=3
TimeoutStopSec=25
Environment=PYTHONPATH={ROOT}/app
Environment=LEA_ROOT={ROOT}
Environment=LEA_KV_STATE_DIR={ROOT}/data/native-kv
WorkingDirectory={WORK}
ExecStart={ROOT}/venv/bin/python -m uvicorn backend:app --host 127.0.0.1 --port 18804
[Install]
WantedBy=default.target
''')
    drop.parent.mkdir(exist_ok=True)
    drop.write_text(f'''[Unit]
Wants=lea-native-kv.service
After=lea-native-kv.service
[Service]
Environment=LEA_NATIVE_KV_URL=http://127.0.0.1:18804
ExecStartPre={ROOT}/venv/bin/python {WORK}/wait_ready.py
''')
    system('daemon-reload');system('enable','--now','lea-native-kv.service')
    subprocess.run([str(ROOT/'venv/bin/python'),str(WORK/'wait_ready.py')],check=True,timeout=150)
    report['phase']='start_formal_agent';save();system('start','lea-agent.service')
    report['health']=health()
    native=report['health']['model']['native_kv']
    assert native['window_budget']==3072 and native['retained_per_head']==614,native
    report.update(completed=True,phase='active',finished_at=time.time());save()
    print(json.dumps(report,ensure_ascii=False),flush=True)
except BaseException as e:
    report.update(error=repr(e),phase='activation_failed',finished_at=time.time());save()
    if stopped:
        if 'backup' in report:
            r=subprocess.run([sys.executable,str(WORK/'rollback.py'),'--automatic'],capture_output=True,text=True,timeout=180)
            report['automatic_rollback']={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr};save()
        else:system('start','lea-agent.service')
    raise
