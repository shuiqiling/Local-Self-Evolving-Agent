from pathlib import Path
import hashlib,json,os,subprocess,time

ROOT=Path('/home/ubuntu/AI-Native/embodied-agent')
WORK=Path(__file__).resolve().parent
SRC=ROOT/'experiments/q4-vnni-20261029/source'
LIB=ROOT/'runtime/q4-vnni-20261029-r1'
assets=ROOT/'experiments/kv-direct-20261029/fastkvzip_cpu_verify'

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

for name in ['gate.pt','official_gate.py']:
    p=WORK/name
    if not p.exists():os.link(assets/name,p)
assert sha(WORK/'gate.pt')=='5b3f75dbaada8f7252a5e365262f02b5cbca9149db1f3f4cc411aaeb3dfc1368'
json_header=next(SRC.glob('vendor/**/json.hpp')).parent
command=['g++','-O3','-std=c++17',str(WORK/'native_worker.cpp'),
    '-I'+str(SRC/'include'),'-I'+str(SRC/'ggml/include'),'-I'+str(SRC/'common'),'-I'+str(json_header),
    '-L'+str(LIB),'-Wl,-rpath,'+str(LIB),'-lllama-common','-lllama','-lggml','-lggml-base','-pthread',
    '-o',str(WORK/'native-worker')]
start=time.monotonic()
with (WORK/'compile.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
manifest=dict(version='native-kvzip-20261029-r1',built_at=time.time(),compile_s=time.monotonic()-start,
    files={n:sha(WORK/n) for n in ['native-worker','native_worker.cpp','backend.py','gate_server.py','gate.pt','official_gate.py']},
    runtime={n:sha(LIB/n) for n in ['libllama.so','libggml-cpu.so']},
    model=sha(ROOT/'models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf'),
    adapter=sha(ROOT/'adapters/concise-v2/concise-v2-f16.gguf'))
manifest['identity']=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
(WORK/'release.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest),flush=True)
