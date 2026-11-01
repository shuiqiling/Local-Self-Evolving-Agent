from pathlib import Path
import sys,tempfile,json,unittest
from types import SimpleNamespace
from unittest.mock import patch
P=Path(__file__).resolve().parent
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'runtime'),str(Path(__file__).resolve().parents[1]/'app')]
import kv_after_lora as K
import dual_adapter_promotion as D
from auto_memory_data import sha,save
from light_lora_policy import load_checked_manifest
from auto_memory_policy import validate_runtime_adapter

class Native:
 def __init__(self,fail_erase=0):self.cached=0;self.ram=[];self.calls=[];self.erases=0;self.fail_erase=fail_erase;self.busy=False
 def __call__(self,base,path,body=None,timeout=None):
  self.calls.append(path)
  if path=='/v1/chat/completions':
   if self.cached:self.ram.append(self.cached)
   self.cached=7
   return {'choices':[{'message':{'content':'好'}}]}
  if path=='/slots':return [{'id':0,'is_processing':self.busy}]
  if path=='/slots/0?action=erase':
   self.erases+=1
   if self.erases==self.fail_erase:raise RuntimeError('controlled native erase failure')
   n=self.cached;self.cached=0;return {'id_slot':0,'n_erased':n}
  raise AssertionError(path)

class ResetTests(unittest.TestCase):
 def test_no_probe_enters_ram_cache_and_final_kv_empty(self):
  with tempfile.TemporaryDirectory() as folder:
   native=Native();p=Path(folder)/'report.json'
   result=K.check_namespaces_then_clear(native,'m','v',fresh_backend=True,output=p)
   self.assertTrue(result['completed']);self.assertEqual(native.cached,0);self.assertEqual(native.ram,[])
   self.assertEqual(native.calls,['/v1/chat/completions','/slots','/slots/0?action=erase']*2)
   self.assertEqual([x['erase_acknowledgements'][0]['n_erased'] for x in result['clears']],[7,7])
 def test_existing_backend_cannot_be_certified_empty(self):
  native=Native()
  with self.assertRaises(RuntimeError):K.check_namespaces_then_clear(native,'m','v',fresh_backend=False,output='unused')
  self.assertEqual(native.calls,[])
 def test_busy_slot_is_not_erased(self):
  native=Native();native.busy=True
  with self.assertRaises(RuntimeError):K.erase_idle_slots(native)
  self.assertEqual(native.erases,0)
 def test_failure_does_not_publish_success(self):
  with tempfile.TemporaryDirectory() as folder:
   target=Path(folder)/'report.json'
   with self.assertRaises(RuntimeError):K.check_namespaces_then_clear(Native(2),'m','v',fresh_backend=True,output=target)
   self.assertFalse(target.exists())
 def test_invalid_erase_acknowledgement_is_rejected(self):
  def call(base,path,body=None,timeout=None):return [{'id':0,'is_processing':False}] if path=='/slots' else {'id_slot':0,'n_erased':True}
  with self.assertRaises(RuntimeError):K.erase_idle_slots(call)
 def test_future_adapter_replacement_keeps_slot_api_and_cpu_arguments(self):
  argv=['server','--ctx-size','8192','--cpu-mask','555','--slot-save-path','/bounded/slots','--lora-scaled','old:0']
  changed=D.command(argv,'new-main','same-helper')
  self.assertEqual(changed[changed.index('--slot-save-path')+1],'/bounded/slots')
  self.assertEqual(changed[changed.index('--cpu-mask')+1],'555')
  self.assertEqual(changed[-4:],['--lora-scaled','new-main:0','--lora-scaled','same-helper:0'])
 def fixture(self,folder,fail_erase=0):
  root=Path(folder);auto=root/'auto';auto.mkdir();work=root/'work';work.mkdir();u=root/'runtime';u.mkdir()
  old=root/'old.gguf';old.write_bytes(b'old');helperfile=root/'helper.gguf';helperfile.write_bytes(b'helper');new=root/'new.gguf';new.write_bytes(b'new')
  decision=root/'decision.json';save(decision,{'source':'explicit_user_request','allow_load_checked_promotion':True})
  parent={'gguf':str(old),'gguf_sha256':sha(b'old'),'version':'old'}
  helper={'enable_thinking':False,'adapter_id':1,'chat_adapter_id':0,'gguf':str(helperfile),'gguf_sha256':sha(b'helper'),'chat_gguf':str(old)}
  save(auto/'helper-active.json',helper)
  config={'training_mode':'light','light_mode_decision':str(decision),'memory_teacher':{'runtime_manifest':str(auto/'helper-active.json')}}
  save(auto/'active.json',parent);save(auto/'config.json',config);save(auto/'window-cursor.json',{'handoffs':106,'last_seq':3231})
  candidate={'gguf':str(new),'gguf_sha256':sha(b'new'),'peft':str(root/'adapter'),'peft_sha256':'peft-sha','version':'new','model':'Qwen3-30B-A3B-Q4_K_M'}
  frozen={'runtime_fingerprint':'fp','source_mode_decision_sha256':sha(decision.read_bytes()),'mode_decision':str(decision),'mode_decision_sha256':sha(decision.read_bytes()),'memory_teacher':{'helper_adapter_sha256':helper['gguf_sha256']}}
  native=Native(fail_erase);events=[];state={'restarted':False}
  def system(*args,**kwargs):
   events.append(tuple(args))
   if args[0]=='restart':state['restarted']=True;native.cached=0;native.ram=[]
   return SimpleNamespace(stdout=('2' if state['restarted'] else '1') if args[0]=='show' else '')
  def request(base,path,body=None,timeout=None):
   if path=='/lora-adapters':return [{'id':0,'path':str(new),'scale':0},{'id':1,'path':str(helperfile),'scale':0}]
   if path=='/api/helper-profile':return {'adapter_sha256':helper['gguf_sha256'],'thinking_enabled':False}
   result=native(base,path,body,timeout)
   events.append(('native',path))
   return result
  api=SimpleNamespace(A=auto,U=u,wait_idle=lambda:None,runtime_fingerprint=lambda _: 'fp',system=system,
   wait_backend=lambda:None,request=request,load_checked_manifest=load_checked_manifest,
   validate_runtime_adapter=validate_runtime_adapter,wait_agent=lambda version: {'primary_adapter':version})
  return root,auto,work,parent,candidate,frozen,api,native,events
 def run_promotion(self,parts):
  root,auto,work,parent,candidate,config,api,native,events=parts
  original=Path.read_bytes
  def read(path):return b'server\0--model\0base.gguf\0--lora-scaled\0old:0\0' if str(path).replace('\\','/').startswith('/proc/') else original(path)
  # Windows represents an absolute /proc path with the current drive prefix.
  def read_cross_platform(path):return b'server\0--model\0base.gguf\0--lora-scaled\0old:0\0' if '/proc/' in str(path).replace('\\','/') else original(path)
  with patch.object(Path,'home',return_value=root),patch.object(Path,'read_bytes',read_cross_platform):
   return D.promote(work,candidate,parent,config,{'step_count':8,'steps':[{'lr':5e-6}]},{'finite_weights':True},api=api)
 def test_promote_clears_before_frontend_resume_and_preserves_cursor(self):
  with tempfile.TemporaryDirectory() as folder:
   parts=self.fixture(folder);cursor=(parts[1]/'window-cursor.json').read_bytes()
   result=self.run_promotion(parts);events=parts[-1]
   last_erase=max(i for i,x in enumerate(events) if x==('native','/slots/0?action=erase'))
   resume=events.index(('start','lea-shuiqilin.service'))
   self.assertLess(last_erase,resume);self.assertTrue(result['kv_cleared_before_window_resume'])
   self.assertEqual((parts[1]/'window-cursor.json').read_bytes(),cursor)
 def test_failed_clear_rolls_back_active_helper_and_status(self):
  with tempfile.TemporaryDirectory() as folder:
   parts=self.fixture(folder,2);auto=parts[1]
   before={n:(auto/n).read_bytes() for n in ['active.json','config.json','helper-active.json','window-cursor.json']}
   with self.assertRaises(RuntimeError):self.run_promotion(parts)
   self.assertTrue(all((auto/n).read_bytes()==value for n,value in before.items()))
   self.assertFalse((auto/'kv-reset-status.json').exists())
   self.assertTrue((parts[2]/'rollback-completed.json').exists())

if __name__=='__main__':unittest.main()
