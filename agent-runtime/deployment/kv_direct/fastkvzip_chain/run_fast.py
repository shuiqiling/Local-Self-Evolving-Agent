"""Verified offline assets and guarded native experiment. Never changes live services."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,os,sqlite3,subprocess,time,urllib.request

ROOT=Path('/home/ubuntu/AI-Native/embodied-agent')
WORK=Path(__file__).resolve().parent
SRC=ROOT/'experiments/q4-vnni-20261029/source'
LIB=ROOT/'runtime/q4-vnni-20261029-r1'
PY=ROOT/'training/concise-v1/venv/bin/python'
REPORT=WORK/'run-report.json'
assert not REPORT.exists(), 'Inspect existing run first'
report={'completed':False,'production_configuration_changed':False,
        'started_at':datetime.now(timezone.utc).isoformat()}
def save():
    t=REPORT.with_suffix('.tmp');t.write_text(json.dumps(report,ensure_ascii=False,indent=2));t.replace(REPORT)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def get(url):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'KV-CPU-Experiment'}),timeout=30) as f:return f.read()
        except Exception:
            if attempt==2:raise
def jobs():
    with sqlite3.connect('file:'+str(ROOT/'data/memory.sqlite3')+'?mode=ro',uri=True) as db:
        return db.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]
def pids():return {u:subprocess.check_output(['systemctl','--user','show',u,'-p','MainPID','--value'],text=True).strip() for u in ['lea-agent.service','lea-model.service','lea-embedding.service']}
children=[];logs=[]
try:
    assert jobs()==0,'Production work active'
    assert subprocess.check_output(['systemctl','--user','show','qwen-novel-llama.service','-p','MainPID','--value'],text=True).strip()=='0'
    report['production_service_pids_before']=pids()
    manifest=json.loads((LIB/'manifest.json').read_text())
    for n in ['libllama.so','libggml-cpu.so']:assert sha(LIB/n)==manifest['files_sha256'][n]
    report['fingerprints']={str(p):sha(p) for p in [WORK/'probe.cpp',WORK/'gate_server.py',SRC/'include/llama.h',LIB/'libllama.so',LIB/'libggml-cpu.so']}
    provenance=json.loads((WORK/'provenance.json').read_text(encoding='utf-8-sig'))
    gate=WORK/'gate.pt'
    assert sha(gate)==provenance['gate_sha256'],'Gate SHA mismatch'
    assert gate.stat().st_size==provenance['gate_bytes']
    report.update(provenance)
    route=json.loads((WORK/'gate-mirror-download-report.json').read_text())
    assert route.get('verified') and route['sha256']==provenance['gate_sha256']
    report['download_route']=route
    for name,digest in provenance['official_source_sha256'].items():
        assert sha(WORK/name)==digest,'Official source mismatch'
    report['phase']='compile';save()
    inc=next(SRC.glob('vendor/**/json.hpp')).parent
    command=['g++','-O3','-std=c++17',str(WORK/'probe.cpp'),'-I'+str(SRC/'include'),'-I'+str(SRC/'ggml/include'),'-I'+str(inc),'-L'+str(LIB),'-Wl,-rpath,'+str(LIB),'-lllama','-lggml','-lggml-base','-pthread','-o',str(WORK/'probe')]
    with (WORK/'compile.log').open('w') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=180)
    env={**os.environ,'LD_LIBRARY_PATH':str(LIB),'GGML_BACKEND_PATH':str(LIB),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'};env.pop('LD_PRELOAD',None)
    err=(WORK/'gate.log').open('w');logs.append(err)
    server=subprocess.Popen([str(PY),'-u',str(WORK/'gate_server.py')],stdout=err,stderr=subprocess.STDOUT,env=env);children.append(server)
    start=time.monotonic()
    while not (WORK/'gate-ready.json').exists():
        assert server.poll() is None,'Gate loader failed; inspect gate.log'
        assert time.monotonic()-start<90,'Gate startup timeout'
        time.sleep(.5)
    report['gate_runtime']=json.loads((WORK/'gate-ready.json').read_text());report['phase']='native_probe';save()
    out=(WORK/'native.stdout').open('w');err=(WORK/'native.stderr').open('w');logs.extend([out,err])
    native=subprocess.Popen([str(WORK/'probe'),str(ROOT),str(WORK/'result.json'),str(WORK/'gate.sock')],stdout=out,stderr=err,env=env);children.append(native)
    start=time.monotonic()
    while native.poll() is None:
        assert time.monotonic()-start<1200,'Native experiment timeout'
        assert jobs()==0,'Production work arrived; stop isolated CPU experiment'
        time.sleep(2)
    assert native.returncode==0,'Native experiment failed; inspect result.json/native.stderr'
    server.wait(timeout=15);assert server.returncode==0,'Gate server failed'
    if (WORK/'control-input.json').exists():
        report['phase']='full_history_control';save()
        with (WORK/'control.stdout').open('w') as out, (WORK/'control.stderr').open('w') as err:
            control=subprocess.Popen([str(WORK/'probe'),str(ROOT),str(WORK/'control-result.json'),'unused','control',str(WORK/'control-input.json')],stdout=out,stderr=err,env=env)
            children.append(control);start=time.monotonic()
            while control.poll() is None:
                assert time.monotonic()-start<900,'Full-history control timeout'
                assert jobs()==0,'Production work arrived; stop full-history control'
                time.sleep(2)
            assert control.returncode==0,'Full-history control failed'
        report['full_history_control_completed']=True
    else: report['full_history_control_completed']=None
    report['completed']=True;report['phase']='complete'
except BaseException as e:
    report.update(phase='failed',error=repr(e));raise
finally:
    for child in reversed(children):
        if child.poll() is None:
            child.terminate()
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill();child.wait()
    for f in logs:f.close()
    report['production_service_pids_after']=pids();report['services_unchanged']=report['production_service_pids_before']==report['production_service_pids_after']
    report['finished_at']=datetime.now(timezone.utc).isoformat();save();print(json.dumps(report,ensure_ascii=False),flush=True)
