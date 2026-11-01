"""Source/format quality feedback, never an answer or semantic rewrite.

All input episodes use the same predeclared at-most-two-attempt policy. Every
attempt is retained, including errors. Semantic qualification still needs an
independent source audit; these checks do not establish factual correctness.
"""
import json
import re
import helper_off_contract as C

VERSION='helper-source-gate-v3'
MAX_ATTEMPTS=2
BASE_SEED=20261031
MAX_IMPACT_CHARS=100

def quality_issues(payload,plan):
    issues=[]
    try:record,trace=C.render(payload,plan)
    except (AssertionError,KeyError,TypeError,ValueError) as exc:
        return ['source_or_format: '+str(exc)],None,None
    required=set(C.required_current_results(payload))
    if required and not required&set(plan['影响']['依据']):
        issues.append('result_basis_missing: 影响判断没有引用本段真实回执或观测；用户目标和规则本身不能证明执行结果。')
    observations={ref for ref in required if '观测' in C.records(payload)[ref]['类型'].split('（',1)[0]}
    retained=set(trace['normalized_selection_plan']['结果'])
    if observations and not observations&retained:
        issues.append('observation_missing: 本段已有实际观测，却未保留任何观测来源；操作已经执行不能代替其观测结果。')
    text=record['影响']
    goal=record['目标']
    normal=lambda s:re.sub(r'[\s，。；、:：.!！?？]','',s)
    if normal(text)==normal(goal):issues.append('goal_echo: 影响只复述了目标，没有陈述证据支持的结果或未知。')
    if len(text)>MAX_IMPACT_CHARS:issues.append('overlong_impact: 影响超过100字，可能重复或在字符串上限处截断；请精炼而不丢条件。')
    repeated=False
    for size in range(8,len(text)//3+1):
        for start in range(len(text)-size+1):
            phrase=text[start:start+size]
            if text.count(phrase)>=3:
                repeated=True;break
        if repeated:break
    if repeated:issues.append('repeated_impact: 同一长片段至少重复三次。')
    if re.search(r'建议|下一步|需补充|需确认|应当去|请先|你需要',text):
        issues.append('advice_style: 疑似新增建议。只记已有结果；已有业务条件必须注明是条件，不改写为新操作。')
    refs=set(C.records(payload))
    for match in re.finditer(r'(?:用户要求|观测|回执|提交|采样|结果|记录)[（(](编号\d+)[)）]',text):
        if match.group(1) not in refs:
            issues.append('invented_alias: 叙述中的来源别名不在原始编号中；不要用数组位置冒充来源编号。')
    if re.search(r'(?:编号|来源).{0,24}(?:已选入|选入|被选|已选择).{0,30}(?:证明|目标|满足)',text):
        issues.append('selection_as_proof: 把本次提炼的来源选择过程当成了原经历或目标成功依据。')
    raw=json.dumps(payload,ensure_ascii=False)
    if not payload.get('任务背景'):
        for phrase in ('长期目标','整体上线','项目上线','总体验收'):
            if phrase in text and phrase not in raw:
                issues.append('unprovided_scope: 提到了原记录没有提供的其他目标或验收范围。');break
    return issues,record,trace

def repair_feedback(issues):
    return ('这是提炼程序的校验反馈，不是新增经历，也不改变首条用户消息中的当时目标和原始记录。'
            '以下问题需在候选JSON中修正：'+json.dumps(issues,ensure_ascii=False,separators=(',',':'))+
            '。重新核对首条原始记录后输出同一四字段JSON。只陈述已有结果或未知，不写建议；'
            '已给出直接对照时依据对照判断，不凭旧版本或来源被选中就宣称满足要求。'
            '校验反馈不提供正确答案，不能把反馈当作原经历。')

def next_messages(messages,raw_content,issues):
    return [*messages,{'role':'assistant','content':raw_content or '{}'},
            {'role':'user','content':repair_feedback(issues)}]
