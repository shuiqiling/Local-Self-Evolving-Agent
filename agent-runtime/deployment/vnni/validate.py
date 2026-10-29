from pathlib import Path
import hashlib,json,os,subprocess,sys,time
ROOT=Path('/home/ubuntu/AI-Native/embodied-agent');WORK=ROOT/'experiments/q4-vnni-20261029'
report={'phase':'waiting_for_build','completed':False,'production_modified':False}
def save():
 p=WORK/'validation-report.tmp';p.write_text(json.dumps(report,indent=2));p.replace(WORK/'validation-report.json')
try:
 for _ in range(240):
  p=WORK/'build-report.json'
  if p.exists():
   b=json.loads(p.read_text())
   if b.get('error'):raise RuntimeError(b['error'])
   if b.get('completed'):break
  time.sleep(3)
 else:raise TimeoutError('Build not complete')
 source=WORK/'source';reference=WORK/'reference';candidate=WORK/'candidate'
 report['phase']='compile_kernel_tests';save();print(json.dumps({'phase':report['phase']}),flush=True)
 command=['g++','-O2','-std=c++17',str(WORK/'test_kernels.cpp'),'-ldl','-o',str(WORK/'test-kernels')]
 for rel in ['ggml/include','ggml/src','ggml/src/ggml-cpu']:command+=['-I',str(source/rel)]
 with (WORK/'test-kernels-build.log').open('w') as out:subprocess.run(command,stdout=out,stderr=subprocess.STDOUT,check=True,timeout=120)
 report['phase']='kernel_correctness_and_microbenchmark';save();print(json.dumps({'phase':report['phase']}),flush=True)
 active=[s for s in subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines() if '/train.py' in s and '/training/' in s];assert not active,'Training present'
 env={**os.environ,'LD_LIBRARY_PATH':str(reference),'GGML_BACKEND_PATH':str(reference)};env.pop('LD_PRELOAD',None)
 with (WORK/'test-kernels.stderr.log').open('w') as err:
  result=subprocess.run([str(WORK/'test-kernels'),str(reference/'libggml-cpu.so'),str(candidate/'libggml-cpu.so')],
     stdout=subprocess.PIPE,stderr=err,env=env,text=True,timeout=240)
 (WORK/'test-kernels.jsonl').write_text(result.stdout)
 assert result.returncode==0,'Kernel comparison failed: '+(WORK/'test-kernels.stderr.log').read_text()[-2000:]
 values=[json.loads(l) for l in result.stdout.splitlines()];correct=[v for v in values if v['kind']=='correctness'];assert len(correct)==1 and correct[0]['passed']
 report['correctness']=correct[0];report['microbenchmarks']=[v for v in values if v['kind']=='micro']
 sys.path.insert(0,str(source/'gguf-py'))
 from gguf import GGUFReader
 metadata=GGUFReader(str(ROOT/'models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf'))
 tensors={}
 for t in metadata.tensors:
  group=tensors.setdefault(t.tensor_type.name,{'tensors':0,'bytes':0,'shapes':[]});group['tensors']+=1;group['bytes']+=int(t.n_bytes)
  shape=[int(x) for x in t.shape]
  if len(shape)==2 and shape not in group['shapes']:group['shapes'].append(shape)
 report['actual_model_tensors']=tensors
 report.update(phase='complete',completed=True)
except BaseException as e:report.update(phase='failed',error=type(e).__name__+': '+str(e));raise
finally:save();print(json.dumps(report),flush=True)
