"""Independent boundary tests for automatic source training (no model scoring)."""
from pathlib import Path
import sys,json,tempfile,unittest,copy
from unittest.mock import patch
P=Path(__file__).resolve().parent
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'runtime'),str(Path(__file__).resolve().parents[1]/'app')]
import seven_projection as S
import auto_evidence_training as T
import dual_adapter_promotion as D
from auto_memory_data import canonical,sha

def fixture():
    events=[]
    for i,temperature in enumerate([None,0.,31.]):
        snapshot={'collected_at_utc':f'2026-01-01T00:0{i}:00Z',
          'cpu':{'temperature_c':temperature,'usage_percent':0},
          'network':{'rx_mib_per_second':0.,'tx_mib_per_second':2.}}
        events.append({'id':'s'+str(i),'seq':i+1,'role':'internal_state','kind':'internal_state_observation',
          'task_id':'t','window_id':'w','body':canonical({'id':'s'+str(i),'snapshot':canonical(snapshot)}),'created':snapshot['collected_at_utc']})
    return {'events':events,'events_sha256':sha(canonical(events)),'start_exclusive_seq':0,'end_inclusive_seq':3,'handoffs':10}

class Tokenizer:
    eos_token='<eos>'
    def encode(self,text,**kw):return list(text)
    def apply_chat_template(self,messages,**kw):return [0]*10

class Integration(unittest.TestCase):
    def test_statistics_preserve_missing_and_zero(self):
        value,_=S.aggregate(fixture());m=value['metrics']['cpu']['temperature_c']
        self.assertEqual(m,{'measured':2,'missing':1,'min':0.,'max':31.,'last':31.})
        self.assertEqual(value['metrics']['network']['rx_mib_per_second']['max'],0)
    def test_duplicate_observation_not_double_counted(self):
        source=fixture();first=copy.deepcopy(source['events'][0]);first['id']='copy';first['kind']='internal_state_input'
        source['events'].append(first);source['events_sha256']=sha(canonical(source['events']))
        self.assertEqual(S.aggregate(source)[0]['sample_count'],3)
    def test_projection_preserves_coverage_and_runtime_goal_attribution(self):
        source=fixture();episodes,coverage=S.project(source,lambda _: ([],{}))
        self.assertEqual(set(coverage),{e['id'] for e in source['events']})
        self.assertIn('非历史用户原话',episodes[0]['input']['原始记录'][0]['类型'])
    def test_tampered_source_rejected(self):
        source=fixture();source['events'][0]['body']='{}'
        with self.assertRaises(AssertionError):S.project(source,lambda _: ([],{}))
    def test_no_adapter_stacking_and_other_arguments_retained(self):
        args=D.command(['server','--ctx-size','8192','--cpu-mask','555','--lora','old','--lora-scaled','old2:0'], 'new','helper')
        self.assertEqual(args,['server','--ctx-size','8192','--cpu-mask','555','--lora-scaled','new:0','--lora-scaled','helper:0'])
        self.assertEqual(D.selection(True),[{'id':0,'scale':1.},{'id':1,'scale':0.}])
        self.assertEqual(D.selection(False),[{'id':0,'scale':0.},{'id':1,'scale':1.}])
    def test_no_same_helper_main(self):
        with self.assertRaises(AssertionError):D.command(['server'],'same','same')
    def test_oversize_answer_not_truncated(self):
        with self.assertRaises(ValueError):T.make_row('x','q',{},'x'*241,{},Tokenizer(),512)
    def test_complete_conditions_not_truncated(self):
        with self.assertRaises(ValueError):T.make_row('x','q',{'condition':'x'*1000},'ok',{},Tokenizer(),1)
    def test_bad_legacy_format_is_not_replay(self):
        self.assertFalse(T.qualifies({'origin':{'format':'teacher-only-decision-v1'},'messages':[]}))
    def test_helper_claim_is_not_gold(self):
        source=fixture();episode=S.project(source,lambda _: ([],{}))[0][0]
        proof={'source_events_sha256':source['events_sha256'],'helper_audit_sha256':'h','helper_adapter_sha256':'a'}
        replayed={'record':{'影响':'已经全部上线且速度提高十倍'},'source_trace':{'normalized_selection_plan':{'结果':['记录1']}}}
        rows=T.rows_for_episode(source,episode,replayed,proof,Tokenizer(),512)
        self.assertEqual(len(rows),9)
        self.assertTrue(all(T.qualifies(r) for r in rows))
        self.assertTrue(all('十倍' not in r['messages'][-1]['content'] for r in rows))
        self.assertTrue(any('null' in r['messages'][-1]['content'] for r in rows))
    def test_assistant_statement_is_not_tool_receipt(self):
        e={'id':'m','seq':1,'task_id':'t','window_id':'w','role':'primary','kind':'assistant_reply','body':canonical({'text':'全部测试通过'}),'created':'now'}
        source={'events':[e]}
        ep={'key':['t','w','g'],'input':{'原始记录':[{'编号':'用户要求','类型':'用户原文','内容':'test'},
          {'编号':'记录1','类型':'模型已输出','内容':{'text':'全部测试通过'}}]},'source_anchors':{'记录1':{'event_id':'m'}}}
        result={'source_trace':{'normalized_selection_plan':{'条件':['用户要求'],'发生':[],'结果':['记录1']}}}
        self.assertEqual(T.rows_for_episode(source,ep,result,{},Tokenizer(),512),[])
    def test_technical_failure_does_not_become_empty_training_success(self):
        with self.assertRaises(OSError):T.verify_episode({}, {}, {'status':'technical_error'}, {})
    def verified_fixture(self,directory):
        source=fixture();ep=S.project(source,lambda _: ([],{}))[0][0]
        plan={'条件':[],'发生':['记录1'],'结果':['记录1'],'影响':{'说明':'只记录本段实际采样范围，未测项保持未知。','依据':['记录1']}}
        response={'choices':[{'message':{'content':json.dumps(plan,ensure_ascii=False)},'finish_reason':'stop'}]}
        result=T.H.finish(ep['input'],response)
        self.assertEqual(result['status'],'proposal')
        meta={'model':'Qwen3-30B-A3B-Q4_K_M','chat_adapter_id':0,'adapter_id':1,'max_output_tokens':1024}
        path=Path(directory)/'audit.json'
        audit={'input':ep['input'],'request':T.H.request_for(ep['input'],meta),'response':response}
        path.write_text(json.dumps(audit,ensure_ascii=False),encoding='utf-8')
        result.update(audit_path=str(path),helper_adapter_sha256='helper-sha')
        options={'audit_dir':directory,'helper_manifest':meta,'helper_adapter_sha256':'helper-sha'}
        return source,ep,{'status':'proposal','result':result},options,path
    def test_full_source_and_actual_request_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            source,ep,stored,options,path=self.verified_fixture(directory)
            replayed,proof=T.verify_episode(source,ep,stored,options)
            self.assertEqual(replayed['status'],'proposal')
            self.assertEqual(proof['helper_audit_sha256'],sha(path.read_bytes()))
    def test_changed_adapter_request_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source,ep,stored,options,path=self.verified_fixture(directory)
            value=json.loads(path.read_text(encoding='utf-8'));value['request']['lora'][0]['scale']=1
            path.write_text(json.dumps(value,ensure_ascii=False),encoding='utf-8')
            with self.assertRaises(AssertionError):T.verify_episode(source,ep,stored,options)
    def test_changed_state_statistics_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source,ep,stored,options,path=self.verified_fixture(directory)
            source['events'][0]['body']=source['events'][0]['body'].replace('null','99')
            with self.assertRaises(AssertionError):T.verify_episode(source,ep,stored,options)

if __name__=='__main__':unittest.main()
