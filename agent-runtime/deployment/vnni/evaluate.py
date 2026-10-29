"""Uninstrumented, alternating whole-model TPS and frozen API parity checks."""
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,shlex,signal,socket,sqlite3,statistics,subprocess,sys,time,urllib.request

ROOT=Path('/home/ubuntu/AI-Native/embodied-agent');WORK=ROOT/'experiments/q4-vnni-20261029'
report={'started_at':datetime.now(timezone.utc).isoformat(),'phase':'preflight','completed':False,
        'production_modified':False,'benchmarks':[],'api_samples':[],
        'scope':'Same Q4_K_M weights, only CPU library substituted in isolated processes. TPS uses uninstrumented llama-bench; API parity uses old revealed synthetic prompts, not a new accuracy evaluation.'}
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def save():
 p=WORK/'evaluation-report.tmp';p.write_text(json.dumps(report,ensure_ascii=False,indent=2));p.replace(WORK/'evaluation-report.json')
def log(**v): print(json.dumps(v,ensure_ascii=False),flush=True)
def ctl(*a):return subprocess.check_output(['systemctl','--user',*a],text=True).strip()
def guard():
 active=[s for s in subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines() if '/train.py' in s and '/training/' in s]
 assert not active,'Concurrent training'
 assert ctl('show','qwen-novel-llama.service','-p','MainPID','--value')=='0'
 with sqlite3.connect(ROOT/'data/memory.sqlite3') as db:
  assert db.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]==0,'Active app jobs'
def api(path,body=None,port=18808):
 req=urllib.request.Request(f'http://127.0.0.1:{port}{path}',data=None if body is None else json.dumps(body,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
 with opener.open(req,timeout=180) as f:return json.load(f)
def environment(folder):
 env={**os.environ,'LD_LIBRARY_PATH':str(folder),'GGML_BACKEND_PATH':str(folder),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
 env.pop('LD_PRELOAD',None);return env
def replace_flag(command,flag,value):
 command=list(command)
 if flag in command:i=command.index(flag);command[i:i+2]=[flag,str(value)]
 else:command+=[flag,str(value)]
 return command
@contextmanager
def server(variant,command):
 folder=WORK/variant;command=[str(folder/'llama-server'),*command[1:]];command=replace_flag(command,'--port',18808)
 with (WORK/(variant+'-api-server.log')).open('w') as out:
  child=subprocess.Popen(command,env=environment(folder),stdout=out,stderr=subprocess.STDOUT)
  try:
   for _ in range(160):
    if child.poll() is not None:raise RuntimeError('API server exited: '+variant)
    try:
     if api('/health').get('status')=='ok':break
    except Exception:pass
    time.sleep(.25)
   else:raise TimeoutError('API server not ready')
   mappings=Path('/proc/'+str(child.pid)+'/maps').read_text()
   libraries=sorted(set(s.split()[-1] for s in mappings.splitlines() if '/libggml' in s or '/libllama' in s))
   assert libraries and all('/experiments/q4-vnni-20261029/'+variant+'/' in p for p in libraries),'Library isolation failed: '+str(libraries)
   report.setdefault('api_loaded_libraries',{})[variant]=libraries;save()
   adapters=api('/lora-adapters');assert len(adapters)==1 and adapters[0]['id']==0 and adapters[0]['scale']==0
   yield
  finally:
   if child.poll() is None:
    child.terminate()
    try:child.wait(timeout=15)
    except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)

def cases():
 sys.path.insert(0,str(ROOT/'app'))
 from lea.schemas import Decision,CatalogSummary
 values=[]
 p=ROOT/'training/concise-v2/evaluation-test/report.json';raw=p.read_bytes()
 assert hashlib.sha256(raw).hexdigest()=='1249d0debf8104232cb288f2193df8e09e47893238e948b14a9b5a0558e913a3'
 for row in [r for r in json.loads(raw)['answers'] if r['variant']=='lora'][:4]:
  messages=row['source_case']['messages'][:2];payload=json.loads(messages[-1]['content']);schema=Decision.model_json_schema();schema['properties']['response']['maxLength']=200
  actions=schema['properties']['action']['enum']
  if payload.get('memory_queries_remaining',1)<=0:actions.remove('recall')
  if not (payload.get('last_result') or {}).get('memory_candidates') or payload.get('memory_reads_remaining',0)<=0:actions.remove('read_memory')
  values.append({'id':'primary-'+row['id'],'role':'primary','schema':Decision,'body':{'messages':messages,'temperature':0,'seed':20261028,'max_tokens':384,'stream':False,'cache_prompt':False,'lora':[{'id':0,'scale':1.0}],
      'response_format':{'type':'json_schema','json_schema':{'name':'Decision','schema':schema,'strict':True}}}})
 p=ROOT/'training/index-v1/evaluation-test/report.json';raw=p.read_bytes();report['index_frozen_sha256']=hashlib.sha256(raw).hexdigest()
 for row in [r for r in json.loads(raw)['answers'] if r['variant']=='base'][:4]:
  values.append({'id':'indexer-'+row['id'],'role':'indexer_base','schema':CatalogSummary,'body':{'messages':row['source_case']['messages'][:2],'temperature':0,'seed':20261029,'max_tokens':200,'stream':False,'cache_prompt':False,'lora':[{'id':0,'scale':0.0}],
      'response_format':{'type':'json_schema','json_schema':{'name':'CatalogSummary','schema':CatalogSummary.model_json_schema(),'strict':True}}}})
 return values

paused=False
try:
 for _ in range(60):
  p=WORK/'validation-final-report.json'
  if p.exists():
   validation=json.loads(p.read_text());assert validation['completed'] and validation['correctness']['passed'];break
  time.sleep(1)
 else:raise TimeoutError('Numerical validation not finalized')
 guard()
 with socket.socket() as sock:assert sock.connect_ex(('127.0.0.1',18808))!=0,'Benchmark port occupied'
 pid=ctl('show','lea-model.service','-p','MainPID','--value');assert int(pid)>0
 command=[x.decode() for x in Path('/proc/'+pid+'/cmdline').read_bytes().split(b'\0') if x]
 report['production_command']=command
 config=ROOT/'runtime-config.json';before=config.read_bytes();report['production_config_sha256']=hashlib.sha256(before).hexdigest()
 source_cpu=Path('/home/ubuntu/ai/tools/llama.cpp/build/bin/libggml-cpu.so');source_hash=hashlib.sha256(source_cpu.read_bytes()).hexdigest()
 report['production_cpu_sha256']=source_hash;report['health_before']=api('/api/health',port=18800)
 frozen=cases();save();signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
 paused=True;ctl('stop','lea-agent.service')
 report['phase']='alternating_tps';save()
 for round_number in range(3):
  for variant in (['reference','candidate'] if round_number%2==0 else ['candidate','reference']):
   folder=WORK/variant
   for phase,prompt,gen,threads in [('prefill',1024,0,16),('decode',0,64,6)]:
    guard();name=f'{round_number+1}-{variant}-{phase}';log(phase='benchmark',round=round_number+1,variant=variant,kind=phase)
    cmd=[str(folder/'llama-bench'),'-m',str(ROOT/'models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf'),'-ngl','0','-p',str(prompt),'-n',str(gen),'-t',str(threads),'-b','512','-ub','512','-fa','on','-r','2','-o','json']
    with (WORK/(name+'.stderr.log')).open('w') as out:r=subprocess.run(cmd,env=environment(folder),stdout=subprocess.PIPE,stderr=out,text=True,timeout=180,check=True)
    guard();(WORK/(name+'.json')).write_text(r.stdout);data=json.loads(r.stdout);assert len(data)==1
    report['benchmarks'].append({'round':round_number+1,'variant':variant,'phase':phase,'raw':data[0]});save()
    log(phase='benchmark_complete',round=round_number+1,variant=variant,kind=phase,tps=data[0]['avg_ts'])
 report['phase']='api_parity';save()
 for variant in ['reference','candidate']:
  guard();log(phase='api_variant',variant=variant)
  with server(variant,command):
   api('/v1/chat/completions',frozen[0]['body']) # excluded warmup
   for case in frozen:
    guard();tick=time.monotonic();r=api('/v1/chat/completions',case['body']);seconds=time.monotonic()-tick;guard()
    choice=r['choices'][0];assert choice.get('finish_reason')!='length'
    result=case['schema'].model_validate_json(choice['message']['content']).model_dump();assert r['timings'].get('cache_n',0)==0
    report['api_samples'].append({'id':case['id'],'role':case['role'],'variant':variant,'seconds':seconds,'result':result,'usage':r['usage'],'timings':r['timings']});save()
    log(phase='api_case',id=case['id'],variant=variant,seconds=round(seconds,3))
 report['summary']={}
 for phase in ['prefill','decode']:
  group={}
  for variant in ['reference','candidate']:
   rows=[r for r in report['benchmarks'] if r['phase']==phase and r['variant']==variant]
   samples=[v for r in rows for v in r['raw'].get('samples_ts',[r['raw']['avg_ts']])]
   group[variant]={'process_blocks':len(rows),'samples':samples,'median_tps':statistics.median(samples),'mean_tps':statistics.mean(samples)}
  group['median_gain_pct']=(group['candidate']['median_tps']/group['reference']['median_tps']-1)*100
  group['paired_round_gain_pct']=[(next(r['raw']['avg_ts'] for r in report['benchmarks'] if r['phase']==phase and r['variant']=='candidate' and r['round']==i)/next(r['raw']['avg_ts'] for r in report['benchmarks'] if r['phase']==phase and r['variant']=='reference' and r['round']==i)-1)*100 for i in [1,2,3]]
  report['summary'][phase]=group
 baseline={r['id']:r for r in report['api_samples'] if r['variant']=='reference'}
 candidates=[r for r in report['api_samples'] if r['variant']=='candidate']
 assert all(r['usage']['prompt_tokens']==baseline[r['id']]['usage']['prompt_tokens'] for r in candidates)
 report['api_exact_matches']=sum(r['result']==baseline[r['id']]['result'] for r in candidates)
 report['api_case_count']=len(candidates)
 report['api_summary']={}
 for variant in ['reference','candidate']:
  report['api_summary'][variant]={role:statistics.median(r['seconds'] for r in report['api_samples'] if r['variant']==variant and r['role']==role) for role in ['primary','indexer_base']}
 assert config.read_bytes()==before and hashlib.sha256(source_cpu.read_bytes()).hexdigest()==source_hash,'Production files changed externally'
 report.update(phase='complete',completed=True)
except BaseException as e:report.update(phase='failed',error=type(e).__name__+': '+str(e));raise
finally:
 if paused:
  restore=subprocess.run(['systemctl','--user','start','lea-agent.service'],capture_output=True,text=True)
  if restore.returncode:report['restore_error']=restore.stderr
 report['agent_restored']=subprocess.run(['systemctl','--user','is-active','lea-agent.service'],capture_output=True).returncode==0
 report['finished_at']=datetime.now(timezone.utc).isoformat();save();log(phase=report['phase'],agent_restored=report['agent_restored'])
