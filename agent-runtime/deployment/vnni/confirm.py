"""ABBA/BAAB confirmation with fixed token input and two resident servers."""
from contextlib import ExitStack,contextmanager
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,signal,socket,sqlite3,statistics,subprocess,time,urllib.request
ROOT=Path('/home/ubuntu/AI-Native/embodied-agent');WORK=ROOT/'experiments/q4-vnni-20261029'
report={'phase':'preflight','completed':False,'production_modified':False,'samples':[],'started_at':datetime.now(timezone.utc).isoformat(),
        'scope':'Same explicit 1024-token prompt, 64 generated tokens, concise-v2 at scale 1, cache disabled. Two isolated servers remain loaded; order ABBA/BAAB, excludes one warmup per variant. Synthetic throughput, not an accuracy test.'}
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def save():
 p=WORK/'confirmation-report.tmp';p.write_text(json.dumps(report,ensure_ascii=False,indent=2));p.replace(WORK/'confirmation-report.json')
def log(**v):print(json.dumps(v),flush=True)
def ctl(*a):return subprocess.check_output(['systemctl','--user',*a],text=True).strip()
def guard():
 assert not [s for s in subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines() if '/train.py' in s and '/training/' in s],'Concurrent training'
 with sqlite3.connect(ROOT/'data/memory.sqlite3') as db:assert db.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]==0,'Active app jobs'
 assert ctl('show','qwen-novel-llama.service','-p','MainPID','--value')=='0'
def api(port,path,body=None):
 req=urllib.request.Request(f'http://127.0.0.1:{port}{path}',data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with opener.open(req,timeout=180) as f:return json.load(f)
def telemetry():
 data={}
 for label,name in [('p_core_khz','/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq'),('e_core_khz','/sys/devices/system/cpu/cpu12/cpufreq/scaling_cur_freq'),('governor','/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor')]:
  p=Path(name)
  try:data[label]=p.read_text().strip()
  except OSError:pass
 for p in Path('/sys/class/hwmon').glob('hwmon*'):
  try:
   if (p/'name').read_text().strip()=='coretemp':data['package_temperature_mC']=(p/'temp1_input').read_text().strip()
  except OSError:pass
 return data
@contextmanager
def server(variant,port,production):
 folder=WORK/variant;command=[str(folder/'llama-server'),*production[1:]];command[command.index('--port')+1]=str(port)
 env={**os.environ,'LD_LIBRARY_PATH':str(folder),'GGML_BACKEND_PATH':str(folder),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'};env.pop('LD_PRELOAD',None)
 with (WORK/(variant+'-confirmation-server.log')).open('w') as out:
  child=subprocess.Popen(command,env=env,stdout=out,stderr=subprocess.STDOUT)
  try:
   for _ in range(160):
    if child.poll() is not None:raise RuntimeError('Server exited')
    try:
     if api(port,'/health').get('status')=='ok':break
    except Exception:pass
    time.sleep(.25)
   else:raise TimeoutError('Server not ready')
   paths=sorted(set(s.split()[-1] for s in Path('/proc/'+str(child.pid)+'/maps').read_text().splitlines() if '/libggml' in s or '/libllama' in s))
   assert paths and all('/experiments/q4-vnni-20261029/'+variant+'/' in p for p in paths)
   report.setdefault('libraries',{})[variant]=paths;save();yield
  finally:
   if child.poll() is None:
    child.terminate()
    try:child.wait(timeout=15)
    except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
paused=False
try:
 assert json.loads((WORK/'evaluation-report.json').read_text())['completed']
 assert json.loads((WORK/'extra-checks-report.json').read_text())['passed']
 guard()
 for port in [18808,18809]:
  with socket.socket() as s:assert s.connect_ex(('127.0.0.1',port))!=0,'Port occupied'
 pid=ctl('show','lea-model.service','-p','MainPID','--value')
 command=[v.decode() for v in Path('/proc/'+pid+'/cmdline').read_bytes().split(b'\0') if v]
 config=(ROOT/'runtime-config.json').read_bytes();report['config_sha256']=hashlib.sha256(config).hexdigest()
 body={'prompt':[1001+i%100 for i in range(1024)],'n_predict':64,'temperature':0,'seed':20261029,
       'cache_prompt':False,'ignore_eos':True,'stream':False,'return_tokens':True,'lora':[{'id':0,'scale':1.0}]}
 report['request']=body;save();signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
 paused=True;ctl('stop','lea-agent.service')
 ports={'reference':18808,'candidate':18809}
 with ExitStack() as stack:
  for variant,port in ports.items():stack.enter_context(server(variant,port,command))
  report['phase']='warmup';save()
  for variant,port in ports.items():api(port,'/completion',body)
  report['phase']='abba_baab';save()
  for position,variant in enumerate(['reference','candidate','candidate','reference','candidate','reference','reference','candidate']):
   guard();before=telemetry();tick=time.monotonic();r=api(ports[variant],'/completion',body);elapsed=time.monotonic()-tick;after=telemetry();guard()
   t=r['timings'];assert t.get('cache_n',0)==0 and t['predicted_n']==64 and t['prompt_n']>=1000
   sample={'position':position+1,'variant':variant,'seconds':elapsed,'timings':t,'content':r['content'],'tokens':r.get('tokens',[]),'telemetry_before':before,'telemetry_after':after}
   report['samples'].append(sample);save();log(position=position+1,variant=variant,prompt_tps=t['prompt_per_second'],output_tps=t['predicted_per_second'])
 assert len({(r['timings']['prompt_n'],r['timings']['predicted_n']) for r in report['samples']})==1
 report['summary']={}
 for variant in ports:
  values=[r for r in report['samples'] if r['variant']==variant]
  report['summary'][variant]={'n':len(values),'prompt_tps_median':statistics.median(r['timings']['prompt_per_second'] for r in values),
      'output_tps_median':statistics.median(r['timings']['predicted_per_second'] for r in values),'seconds_median':statistics.median(r['seconds'] for r in values)}
 for phase in ['prompt','output']:report[phase+'_gain_pct']=(report['summary']['candidate'][phase+'_tps_median']/report['summary']['reference'][phase+'_tps_median']-1)*100
 report['all_contents_identical']=len({r['content'] for r in report['samples']})==1
 report['all_tokens_identical']=all(r['tokens'] for r in report['samples']) and len({tuple(r['tokens']) for r in report['samples']})==1
 assert (ROOT/'runtime-config.json').read_bytes()==config
 report.update(phase='complete',completed=True)
except BaseException as e:report.update(phase='failed',error=type(e).__name__+': '+str(e));raise
finally:
 if paused:subprocess.run(['systemctl','--user','start','lea-agent.service'],check=True)
 report['agent_restored']=subprocess.run(['systemctl','--user','is-active','lea-agent.service'],capture_output=True).returncode==0
 report['finished_at']=datetime.now(timezone.utc).isoformat();save();log(phase=report['phase'],agent_restored=report['agent_restored'])
