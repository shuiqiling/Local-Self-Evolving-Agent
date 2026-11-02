from pathlib import Path
import sys,json,unittest,copy,tempfile
E=Path(__file__).resolve().parent
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'runtime'),str(Path(__file__).resolve().parents[1]/'app')]
import behavior_targets as B
import auto_evidence_training as T
from auto_memory_data import canonical,sha

class Tokenizer:
    eos_token='<eos>'
    def encode(self,text,**kw):return list(text)
    def apply_chat_template(self,messages,**kw):return list(canonical(messages))

def source(events):return {'events':events,'events_sha256':sha(canonical(events)),'start_exclusive_seq':0,'end_inclusive_seq':len(events),'handoffs':10}
def event(seq,role,kind,body,task='t'):
    return {'id':'e'+str(seq),'seq':seq,'role':role,'kind':kind,'body':canonical(body),'created':'2026-11-01T00:00:00Z','task_id':task,'window_id':'w'}
def decision(tool,args):return {'action':'tool','tool':tool,'arguments':args,'response':''}
def test_source(passed=True):
    args={'name':'trim_edges','revision':'v2'}
    return source([
        event(1,'user','message',{'text':'请开发 trim_edges，修改后测试当前版本，通过才发布。'}),
        event(2,'tool','tool_result',{'ok':True,'tool':'dev.submit','value':{'name':'trim_edges','revision':'v2','code_sha256':'code-v2','status':'saved','tested':False}}),
        event(3,'primary','primary_decision',decision('dev.test',args)),
        event(4,'primary','tool_request',{'tool':'dev.test','arguments':{**args,'test_fingerprint':'suite-v2'},'operation_id':'op1'}),
        event(5,'tool','tool_result',{'ok':True,'tool':'dev.test','operation_id':'op1','value':{
            **args,'fingerprint':'suite-v2','code_sha256':'code-v2','passed':passed,'own':{'passed':passed,'cases':[{'passed':passed,'actual':'AFTER_ACTION_SECRET'}]}}})])
def publish_source():
    s=test_source();v=B.body(s['events'][-1]);v['value'].update(validation_scope='self_and_owner_tests',independent={'passed':True,'cases':[{'actual':'ok','passed':True}]})
    s['events'][-1]['body']=canonical(v);args={'name':'trim_edges','revision':'v2'}
    s['events'] += [event(6,'primary','primary_decision',decision('dev.publish',args)),
        event(7,'primary','tool_request',{'tool':'dev.publish','arguments':{**args,'test_fingerprint':'suite-v2'},'operation_id':'op2'}),
        event(8,'tool','tool_result',{'ok':True,'tool':'dev.publish','operation_id':'op2','value':{**args,'status':'published','code_sha256':'code-v2'}})]
    return source(s['events'])
def web_source(open_page=False):
    tool='web.open' if open_page else 'web.search'
    args={'url':'https://docs.python.org/3/library/stdtypes.html','query':''} if open_page else {'query':'Python str.strip'}
    goal='读取 '+args['url'] if open_page else '搜索 Python str.strip'
    result={'url':args['url'],'text':'External document','retrieved':True} if open_page else {'query':args['query'],'retrieved':True,'results':[{'id':'p1','title':'str.strip'}]}
    return source([event(1,'user','message',{'text':goal}),event(2,'primary','primary_decision',decision(tool,args)),
        event(3,'primary','tool_request',{'tool':tool,'arguments':args,'operation_id':'op'}),
        event(4,'tool','tool_result',{'tool':tool,'operation_id':'op','ok':True,'value':result})])
def memory_source():
    question='请用 read_memory 核对原文。';args={'request_id':'q1','card_ids':['M1'],'question':question}
    return source([event(1,'user','message',{'text':question}),
        event(2,'retriever','retrieval_result',{'request_id':'q1','memory_candidates':[{'card_id':'card-a','summary':'本次端口更正'}]}),
        event(3,'primary','primary_decision',{'action':'read_memory','tool':'','arguments':args,'response':''}),
        event(4,'runtime','memory_read_selection',{'request_id':'q1','card_aliases':{'M1':'card-a'},'selected_card_ids':['card-a']}),
        event(5,'archive_reader','archive_read_result',{'request_id':'q1','archive_read':True,'answer_status':'retrieved','evidence':[{'quote':'AFTER_ACTION_SECRET'}]})])
def update(s,index,mutate):
    v=B.body(s['events'][index]);mutate(v);s['events'][index]['body']=canonical(v)
    s['events_sha256']=sha(canonical(s['events']));return s
def rows(s,limit=4096):
    out=[]
    for ep in B.project(s):
        plan={'source_trace':{'normalized_selection_plan':{'结果':[ep['behavior']['receipt_ref']]}}}
        proof={'source_anchors':ep['source_anchors'],'source_events_sha256':s['events_sha256'],'helper_audit_sha256':'audit','helper_adapter_sha256':'helper'}
        out+=T.rows_for_episode(s,ep,plan,proof,Tokenizer(),limit)
    return out

class Behavior(unittest.TestCase):
    def test_generation_contract_matches_result_presence_and_impact_limit(self):
        ep=B.project(test_source())[0]
        schema=T.H.request_for(ep['input'],{'model':'m','chat_adapter_id':0,'adapter_id':1,'max_output_tokens':1024})['response_format']['json_schema']['schema']
        self.assertEqual(schema['properties']['结果']['minItems'],1)
        self.assertEqual(schema['properties']['影响']['properties']['说明']['maxLength'],100)
    def test_bad_impact_length_does_not_turn_valid_evidence_into_bad_action(self):
        s=test_source();ep=B.project(s)[0];receipt=ep['behavior']['receipt_ref']
        plan={'条件':['记录1'],'发生':['记录2'],'结果':[receipt],
            '影响':{'说明':''.join(chr(0x4e00+i) for i in range(110)),'依据':[receipt]}}
        raw={'choices':[{'message':{'content':canonical(plan)},'finish_reason':'stop'}]}
        result=T.H.finish(ep['input'],raw)
        self.assertEqual(result['status'],'needs_review')
        self.assertEqual(len(result['review_reasons']),1)
        self.assertTrue(result['review_reasons'][0].startswith('overlong_impact:'))
        meta={'model':'Qwen3-30B-A3B-Q4_K_M','chat_adapter_id':0,'adapter_id':1,'max_output_tokens':1024}
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'audit.json';p.write_text(json.dumps({'input':ep['input'],'request':T.H.request_for(ep['input'],meta),'response':raw}),encoding='utf-8')
            result.update(audit_path=str(p),helper_adapter_sha256='helper')
            replayed,proof=T.verify_episode(s,ep,{'status':'needs_review','result':result},
                {'audit_dir':d,'helper_manifest':meta,'helper_adapter_sha256':'helper'})
            sample=T.rows_for_episode(s,ep,replayed,proof,Tokenizer(),4096)[0]
            self.assertTrue(T.qualifies(sample));self.assertNotIn(plan['影响']['说明'],canonical(sample['messages']))
            self.assertEqual(proof['helper_record_status'],'needs_review')
            self.assertTrue(proof['helper_impact_excluded_from_gold'])
            raw['choices'][0]['finish_reason']='length'
            p.write_text(json.dumps({'input':ep['input'],'request':T.H.request_for(ep['input'],meta),'response':raw}),encoding='utf-8')
            result=T.H.finish(ep['input'],raw);result.update(audit_path=str(p),helper_adapter_sha256='helper')
            with self.assertRaises(ValueError):T.verify_episode(s,ep,{'status':'needs_review','result':result},
                {'audit_dir':d,'helper_manifest':meta,'helper_adapter_sha256':'helper'})
    def test_submission_needs_actual_independent_same_code_pass(self):
        s=publish_source();args={'name':'trim_edges','description':'trim','code':'def run(arguments):\n    return 1',
            'tests':[{'input':{'x':i},'expected':1} for i in range(3)]}
        s['events'][1:1]=[event(1.5,'primary','primary_decision',decision('dev.submit',args)),
            event(1.6,'primary','tool_request',{'tool':'dev.submit','arguments':args,'operation_id':'submit-op'})]
        for i,e in enumerate(s['events']):e.update(seq=i+1,id='e'+str(i+1))
        sb=B.body(s['events'][3]);sb.update(operation_id='submit-op');sb['value']['code_sha256']=sha(args['code']);s['events'][3]['body']=canonical(sb)
        for e in s['events']:
            b=B.body(e)
            if e['kind']=='tool_result' and b.get('tool') in ('dev.test','dev.publish'):
                b['value']['code_sha256']=sha(args['code'])
                if b['tool']=='dev.test':b['value']['independent']['cases']=[{'passed':True,'actual':1}]*3
                e['body']=canonical(b)
        s['events_sha256']=sha(canonical(s['events']))
        self.assertIn('dev.submit',[json.loads(r['messages'][-1]['content'])['tool'] for r in rows(s)])
        for e in s['events']:
            if e['kind']=='tool_result' and B.body(e).get('tool')=='dev.test':
                b=B.body(e);b['value']['passed']=False;e['body']=canonical(b)
        s['events_sha256']=sha(canonical(s['events']))
        self.assertNotIn('dev.submit',[json.loads(r['messages'][-1]['content'])['tool'] for r in rows(s)])
    def test_real_failed_test_is_correct_test_action(self):
        r=rows(test_source(False));self.assertEqual(len(r),1);self.assertEqual(json.loads(r[0]['messages'][-1]['content'])['tool'],'dev.test');self.assertTrue(T.qualifies(r[0]))
    def test_success_is_not_leaked_into_pre_action_input(self):
        for s in (test_source(),memory_source()):
            r=rows(s);self.assertEqual(len(r),1);self.assertNotIn('AFTER_ACTION_SECRET',canonical(r[0]['messages']))
    def test_wrong_receipt_operation_id(self):
        self.assertEqual(rows(update(test_source(),4,lambda v:v.update(operation_id='other'))),[])
    def test_model_success_claim_is_not_execution(self):
        s=test_source();s['events'][-1]['role']='primary';s['events_sha256']=sha(canonical(s['events']));self.assertEqual(rows(s),[])
    def test_stale_code_revision(self):
        self.assertEqual(rows(update(test_source(),1,lambda v:v['value'].update(revision='v3'))),[])
    def test_test_contract_mismatch(self):
        self.assertEqual(rows(update(test_source(),4,lambda v:v['value'].update(fingerprint='other-suite'))),[])
    def test_duplicate_receipt_not_positive_action(self):
        self.assertEqual(rows(update(test_source(),4,lambda v:v.update(deduplicated=True))),[])
    def test_negative_user_instruction(self):
        self.assertEqual(rows(update(test_source(),0,lambda v:v.update(text='不要测试或发布这个版本'))),[])
    def test_checked_publish(self):self.assertEqual([json.loads(r['messages'][-1]['content'])['tool'] for r in rows(publish_source())],['dev.test','dev.publish'])
    def test_self_test_only_cannot_teach_publish(self):
        s=update(publish_source(),4,lambda v:v['value'].update(validation_scope='selftests_only',independent=None));self.assertEqual(len(rows(s)),1)
    def test_publish_after_failed_tests_excluded(self):
        s=update(publish_source(),4,lambda v:v['value'].update(passed=False));self.assertEqual(len(rows(s)),1)
    def test_requested_search(self):self.assertEqual(len(rows(web_source())),1)
    def test_search_success_does_not_prove_query_relevance(self):
        s=update(web_source(),0,lambda v:v.update(text='请搜索另一件事'));self.assertEqual(rows(s),[])
    def test_unavailable_search_excluded(self):
        self.assertEqual(rows(update(web_source(),3,lambda v:v['value'].update(retrieved=False))),[])
    def test_read_requested_url(self):self.assertEqual(len(rows(web_source(True))),1)
    def test_http_success_without_body_excluded(self):
        self.assertEqual(rows(update(web_source(True),3,lambda v:v['value'].update(text='',status=200))),[])
    def test_real_memory_read(self):self.assertEqual(len(rows(memory_source())),1)
    def test_wrong_memory_alias(self):
        self.assertEqual(rows(update(memory_source(),3,lambda v:v.update(card_aliases={'M1':'other'}))),[])
    def test_no_archive_evidence(self):
        self.assertEqual(rows(update(memory_source(),4,lambda v:v.update(evidence=[]))),[])
    def test_helper_omitted_actual_receipt(self):
        s=test_source();ep=B.project(s)[0];self.assertEqual(T.rows_for_episode(s,ep,{'source_trace':{'normalized_selection_plan':{'结果':[]}}},{},Tokenizer(),4096),[])
    def test_budget_never_drops_conditions(self):
        with self.assertRaises(ValueError):rows(test_source(),20)
    def test_future_input_anchor_rejected(self):
        r=rows(test_source())[0];r['origin']['proof']['source_anchors']['用户要求']['seq']=99;self.assertFalse(T.qualifies(r))
    def test_tampered_frozen_source(self):
        s=test_source();s['events'][0]['body']='{}'
        with self.assertRaises(AssertionError):B.project(s)
    def test_changed_goal_between_decision_and_receipt(self):
        s=test_source();s['events'][3]=event(4,'user','message',{'text':'停止'});s['events_sha256']=sha(canonical(s['events']));self.assertEqual(rows(s),[])

if __name__=='__main__':unittest.main()
