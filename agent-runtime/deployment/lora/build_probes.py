"""Counterfactual holdouts for constant-value biases spotted in authored data."""
import hashlib
import json
from build_data import HERE,PRIMARY,dump

cases=[
 ('红杉泵的维护时间？',[('红杉泵维护安排在周三晚上。','user_statement')],'周三晚上。',['周三','晚上'],['周一']),
 ('白桦阀放在哪？',[('白桦阀放在A区7号架。','user_statement')],'A区7号架。',['A区7号架'],['C区']),
 ('刚才的搬运成功了吗？',[('模型推测搬运失败。','model_conclusion'),('工具实测：搬运成功，物体已到目标位置。','tool_observation')],'成功，工具实测物体已到目标位置。',['成功','目标位置'],['实测失败']),
 ('红杉泵检修员的邮箱？',[('红杉泵检修员邮箱为hans@example.invalid，仅为合成测试地址。','user_statement')],'hans@example.invalid。',['hans@example.invalid'],['不知道','未记录']),
 ('银杏测试台已实际运行了吗？',[('工具实测：银杏测试台已完成验收并正在运行。','tool_observation')],'已实际运行，工具实测正在运行。',['运行'],['无法确认','未运行']),
 ('白桦阀的新压力上限？',[('白桦阀旧上限25kPa，已更正为35kPa，旧值作废。','user_statement')],'35kPa。',['35kPa'],['上限25kPa']),
]
rows=[]
for i,(question,evidence,answer,required,forbidden) in enumerate(cases):
    payload={'current_request':question,'goal':'回答当前请求','constraints':['简短回答，不虚构'],
             'last_result':{'context_kind':'original_record_excerpts','memory_candidates':[],
                 'evidence':[{'event_ref':f'E{j+1}','quote':q,'source_type':source,'recorded_at':f'2026-10-{20+j}T10:00:00Z','excerpt_only':False} for j,(q,source) in enumerate(evidence)]},
             'records':[],'tools':{},'environment':{},'answer_max_chars':200,'memory_queries_remaining':0,'memory_reads_remaining':0}
    expected={'action':'finish','response':answer,'tool':'','arguments':{}}
    rows.append({'id':f'counterfactual-{i}','category':'counterfactual','split':'test','expected':expected,'required':required,'forbidden':forbidden,
                 'messages':[{'role':'system','content':PRIMARY},{'role':'user','content':dump(payload)},{'role':'assistant','content':dump(expected)}]})
path=HERE/'data/probes.jsonl';path.write_text(''.join(dump(row)+'\n' for row in rows),encoding='utf-8',newline='\n')
print(json.dumps({'rows':len(rows),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'reason':'Data-bias probes designed before v2 dev/heldout generation; never training data.'}))
