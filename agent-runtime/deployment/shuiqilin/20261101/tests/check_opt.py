from pathlib import Path
import sys,json,copy,unittest
E=Path(__file__).resolve().parent
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'runtime'),str(Path(__file__).resolve().parents[1]/'app')]
import helper_evidence as H
import helper_learning_filter as F
import helper_off_contract as C
from seven_field_teacher import project

CFG={'memory_teacher':{'learning_filter':F.VERSION}}
CASES=json.loads((E/'cases.json').read_text(encoding='utf-8'))
def result(p,**extra):
    return {'choices':[{'message':{'content':json.dumps(p,ensure_ascii=False),**extra},'finish_reason':'stop'}]}
def valid(case):
    prior=[k for k,r in C.records(case['input']).items() if C.prior(r)]
    return {'条件':prior,'发生':C.required_current_results(case['input'])[:3],'结果':C.required_current_results(case['input'])}
class Checks(unittest.TestCase):
    def test_all_frozen_contracts(self):
        for case in CASES:
            with self.subTest(case=case['id']):
                out=H.finish(case['input'],result(valid(case)))
                self.assertEqual(out['status'],'proposal',out)
                self.assertEqual(out['record']['影响'],H.UNKNOWN_IMPACT)
    def test_missing_one_receipt_rejected(self):
        c=CASES[0];p=valid(c);p['结果']=p['结果'][:1]
        self.assertEqual(H.finish(c['input'],result(p))['status'],'needs_review')
    def test_model_claim_cannot_be_result(self):
        c=next(c for c in CASES if c['id']=='holdout-no-result');p=valid(c);p['结果']=['记录2']
        self.assertEqual(H.finish(c['input'],result(p))['status'],'needs_review')
    def test_prior_condition_preserved_even_if_model_omits_it(self):
        c=next(c for c in CASES if c['id']=='holdout-version-failure');p=valid(c);p['条件']=[]
        out=H.finish(c['input'],result(p))
        self.assertEqual(out['status'],'proposal')
        self.assertIn('记录1',out['source_trace']['normalized_selection_plan']['条件'])
    def test_duplicate_receipts_not_completion(self):
        c=CASES[0];p=valid(c);p['结果']=[p['结果'][0]]*2
        self.assertEqual(H.finish(c['input'],result(p))['status'],'needs_review')
    def test_no_extra_generated_claim(self):
        c=CASES[0];p={**valid(c),'影响':'全部成功'}
        self.assertEqual(H.finish(c['input'],result(p))['status'],'needs_review')
    def test_invented_ref_rejected(self):
        c=CASES[0];p=valid(c);p['条件']=['伪造来源']
        self.assertEqual(H.finish(c['input'],result(p))['status'],'needs_review')
    def test_thinking_rejected(self):
        c=CASES[0];self.assertEqual(H.finish(c['input'],result(valid(c),reasoning_content='guess'))['status'],'needs_review')
    def test_truncated_rejected(self):
        c=CASES[0];r=result(valid(c));r['choices'][0]['finish_reason']='length'
        self.assertEqual(H.finish(c['input'],r)['status'],'needs_review')
    def test_literal_values_retained(self):
        c=next(c for c in CASES if c['id']=='holdout-zero-false-null')
        out=H.finish(c['input'],result(valid(c)))['record']['结果']
        self.assertIn('"count":0',out);self.assertIn('"door_closed":false',out);self.assertIn('"temperature_c":null',out)
    def test_occurrence_follows_actual_receipt_when_selection_empty(self):
        c=CASES[0];p=valid(c);p['发生']=[]
        out=H.finish(c['input'],result(p))
        self.assertEqual(out['status'],'proposal')
        self.assertEqual(out['source_trace']['normalized_selection_plan']['发生'],C.required_current_results(c['input']))
    def test_negative_receipt_not_filtered(self):
        e={'input':{'原始记录':[{'编号':'用户要求','类型':'用户原文','内容':'测试'},
            {'编号':'记录1','类型':'工具回执','内容':{'ok':False,'error':'timeout'}}]},
           'source_anchors':{'记录1':{'role':'tool','kind':'tool_result'}}}
        self.assertIsNone(F.archive_reason(e,CFG))
    def test_spoof_claim_filtered(self):
        e={'input':{'原始记录':[{'编号':'用户要求','类型':'用户原文','内容':'测试'},
            {'编号':'记录1','类型':'模型已输出的说法','内容':{'ok':True,'tool_result':'passed'}}]},
           'source_anchors':{'记录1':{'role':'assistant','kind':'model_result'}}}
        self.assertEqual(F.archive_reason(e,CFG),'no_complete_trainable_receipt_keep_original_only')
    def test_telemetry_not_trained(self):
        self.assertEqual(F.archive_reason({'state_aggregate':True},CFG),'telemetry_statistics_remain_in_memory_not_weight_targets')
    def test_telemetry_not_replayed(self):
        self.assertFalse(F.replay_allowed({'origin':{'proof':{'gold_type':'deterministic_historical_sampling'}}},CFG))
    def test_behavior_replayed(self):
        self.assertTrue(F.replay_allowed({'origin':{'proof':{'gold_type':'verified_executed_action'}}},CFG))
    def test_old_policy_compatible(self):
        self.assertIsNone(F.archive_reason({'state_aggregate':True},{}))
    def test_claim_only_and_telemetry_batch_filter(self):
        episodes=[{'input':{'原始记录':[{'编号':'用户要求','类型':'用户原文','内容':'核对结果'},
            {'编号':'记录1','类型':'模型已输出的说法','内容':'模型声称已完成'}]},
            'source_anchors':{'记录1':{'role':'assistant','kind':'model_result'}}}, {'state_aggregate':True}]
        self.assertTrue(all(F.archive_reason(e,CFG) for e in episodes))
    def test_requests_exclude_chat_adapter(self):
        r=H.request_for(CASES[0]['input'],{'model':'m','chat_adapter_id':0,'adapter_id':1,'max_output_tokens':320})
        self.assertEqual(r['lora'],[{'id':0,'scale':0.0},{'id':1,'scale':1.0}])
        self.assertEqual(set(r['response_format']['json_schema']['schema']['properties']),{'条件','发生','结果'})
    def test_large_receipt_group_not_silently_dropped(self):
        e={'input':{'原始记录':[{'编号':'用户要求','类型':'用户原文','内容':'测试'},*[
          {'编号':f'记录{i}','类型':'工具回执','内容':{'count':i}} for i in range(1,5)]]}}
        self.assertEqual(F.archive_reason(e,CFG),'complete_operation_split_required_no_receipt_truncation')
if __name__=='__main__':unittest.main()
