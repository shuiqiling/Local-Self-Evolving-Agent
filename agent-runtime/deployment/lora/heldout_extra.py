"""Additional held-out wording and task categories, excluded from training."""
from copy import deepcopy
import hashlib
import json
from build_data import HERE, PRIMARY, dump

cases=[
 ('北栈控制箱的接线图在哪？',['北栈控制箱接线图在蓝色资料柜第二层。南栈的接线图在值班室。'],'蓝色资料柜第二层。',['蓝色资料柜','第二层'],['值班室']),
 ('北栈检修员的手机号呢？',['北栈检修员姓韩，未留联系方式。南栈检修员手机号为合成号码00000000000。'],'不知道，记录未提供北栈检修员的手机号。',['不知道'],['00000000000','南栈']),
 ('除湿机什么时候允许再次开机？',['只有湿度低于55%并连续维持15分钟，才允许再次开机；两个条件缺一不可。'],'湿度低于55%且连续维持15分钟，才允许开机。',['55%','15分钟'],[]),
 ('测试台现在肯定没通电吧？',['记录只规定未检查前禁止通电，未记录现场通电观测。'],'无法确认现场是否通电；未检查前禁止通电。',['无法确认','禁止通电'],['肯定没通电','已经断电']),
 ('第二版计划的交货日期是哪天？',['第一版计划是11月2日交货。第二版将交货改为11月8日，第一版日期取消。'],'11月8日。',['11月8日'],['11月2日']),
 ('柜门是否关好了？',['模型推测柜门已关好。','门磁工具读数：open，柜门仍开着。'],'没有，门磁实测柜门仍开着。',['开着'],['已经关好']),
 ('我需要完整的故障处置步骤。',['处置分三步：切断电源；挂上检修标识；由授权人员复测。三步都必须执行。'],'切断电源，挂检修标识，再由授权人员复测。',['切断电源','检修标识','授权人员','复测'],[]),
 ('重试成功了吗？',['第一次上传失败。第二次工具返回HTTP 201并提供记录ID R-871，表示创建成功。'],'第二次成功，记录ID为R-871。',['成功','R-871'],[]),
 ('上次的常温读数能说明现在的温度吗？',['昨天14时温度为21℃，之后没有新测量。'],'不能，21℃是昨天的读数，当前温度未知。',['昨天','未知'],[]),
 ('青禾支架的承载上限是多少？',['青禾支架承载上限8kg。砂石支架承载上限20kg。'],'8kg。',['8kg'],['20kg']),
 ('检修完成后要不要清理现场？',['清理现场的要求被保留，只取消了拍照步骤。'],'要清理现场；取消的只是拍照步骤。',['清理现场','拍照'],['不用清理']),
 ('原文已说明止回阀方向，告诉我方向即可。',['止回阀箭头应朝向储液罐。此条已含完整方向信息，不需要额外档案。'],'箭头朝向储液罐。',['储液罐'],[]),
]
path=HERE/'data/test.jsonl'
rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()][:12]
for i,(question,evidence,answer,required,forbidden) in enumerate(cases):
    original=deepcopy(rows[0]);payload=json.loads(original['messages'][1]['content'])
    payload['current_request']=question
    payload['last_result']={'context_kind':'original_record_excerpts','evidence':[
        {'event_ref':f'E{j+1}','quote':text,'source_type':'tool_observation' if i==5 and j==1 else 'model_conclusion' if i==5 else 'user_statement',
         'recorded_at':f'2026-10-{15+j:02d}T12:00:00Z','excerpt_only':False} for j,text in enumerate(evidence)],'memory_candidates':[]}
    expected={'action':'finish','response':answer,'tool':'','arguments':{}}
    original.update(id=f'test-novel-{i:02d}',category='novel_wording',required=required,forbidden=forbidden,expected=expected,
        messages=[{'role':'system','content':PRIMARY},{'role':'user','content':dump(payload)},{'role':'assistant','content':dump(expected)}])
    rows.append(original)
path.write_text(''.join(dump(r)+'\n' for r in rows),encoding='utf-8',newline='\n')
manifest_path=HERE/'data/manifest.json';manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
manifest['splits']['test']={'rows':len(rows),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'composition':'12 template-generalization + 12 new wording/categories; frozen before adapter evaluation'}
manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(manifest['splits']['test']))
