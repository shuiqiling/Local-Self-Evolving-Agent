from pathlib import Path
import hashlib,json,os,shutil,subprocess,time
ROOT=Path('/home/ubuntu/AI-Native/embodied-agent');WORK=ROOT/'experiments/q4-vnni-20261029'
reference=WORK/'reference';candidate=WORK/'candidate';source=WORK/'source';build=WORK/'build-vnni'
report={'phase':'preflight','completed':False,'production_modified':False}
def save():
 p=WORK/'build-report.tmp';p.write_text(json.dumps(report,indent=2));p.replace(WORK/'build-report.json')
def run(command,name):
 report['phase']=name;save();print(json.dumps({'phase':name}),flush=True)
 with (WORK/(name+'.log')).open('w') as out:subprocess.run(command,stdout=out,stderr=subprocess.STDOUT,check=True,timeout=900)
try:
 assert json.loads((WORK/'profile-report.json').read_text())['completed']
 active=[s for s in subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines() if '/train.py' in s and '/training/' in s];assert not active,'Training present'
 run(['/usr/bin/python3',str(WORK/'patch.py'),'--source',str(source),'--out',str(WORK)],'patch')
 run(['cmake','-S',str(source),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release','-DGGML_NATIVE=ON','-DGGML_BLAS=OFF',
      '-DGGML_CPU_REPACK=ON','-DGGML_LLAMAFILE=ON','-DGGML_OPENMP=ON','-DLLAMA_BUILD_TESTS=OFF',
      '-DLLAMA_BUILD_EXAMPLES=OFF','-DLLAMA_CURL=OFF','-DLLAMA_OPENSSL=OFF'],'configure')
 run(['cmake','--build',str(build),'--target','ggml-cpu','-j','8'],'compile')
 shutil.copytree(reference,candidate,symlinks=False)
 libs=list((build/'bin').glob('libggml-cpu.so*'));assert libs
 for p in libs:shutil.copy2(p,candidate/p.name,follow_symlinks=True)
 report['candidate_cpu_libraries']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in candidate.glob('libggml-cpu.so*')}
 functions={}
 for name in ['ggml_gemv_q4_K_8x8_q8_K','ggml_gemm_q4_K_8x8_q8_K']:
  dis=subprocess.check_output(['objdump','-d','-M','intel','--disassemble='+name,str(candidate/'libggml-cpu.so')],text=True)
  functions[name]={'vpdpbusd':dis.count('vpdpbusd'),'vpmaddubsw':dis.count('vpmaddubsw'),'vpmulld':dis.count('vpmulld')}
  assert functions[name]['vpdpbusd']>0,'VNNI instructions absent'
 report.update(phase='complete',completed=True,disassembly=functions)
except BaseException as e:report.update(phase='failed',error=type(e).__name__+': '+str(e));raise
finally:save();print(json.dumps(report),flush=True)
