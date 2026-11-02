"""Window-source bridge to the online no-thinking auxiliary extractor.

Old cause/advice caches are never reused. Whole oversized task segments and
segments without a historical goal stay review-pending, with originals retained.
"""
from pathlib import Path
from collections import OrderedDict
import json,sqlite3,time,urllib.error
from auto_memory_data import canonical,sha,save
from auto_memory_eval import request
from helper_off_core import VERSION

KINDS={'tool_request':'调用请求','tool_attempt':'调用请求','tool_result':'工具回执',
 'retrieval_result':'检索索引回执（不是原文答案）','archive_read_result':'读档回执（历史引用，不是实时观测）',
 'assistant_reply':'模型已输出的说法（实际答复）','primary_decision':'模型已输出的说法（意图不代表实际执行）',
 'model_result':'模型已输出的说法（不能替代操作回执）','job_error':'运行时回执',
 'internal_state_observation':'观测（本段状态采样）','internal_state_result':'模型已输出的说法（状态反馈）',
 'message':'用户原文'}
def decoded(value):
    try:return json.loads(value) if isinstance(value,str) else value
    except ValueError:return value

def historical_goal(source,event):
    path=source.get('source_db');seq=event.get('seq')
    if not path or not isinstance(seq,int):return None
    with sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        row=db.execute("SELECT * FROM events WHERE task_id=? AND seq<? AND role='user' AND kind='message' ORDER BY seq DESC LIMIT 1",(event['task_id'],seq)).fetchone()
    return dict(row) if row else None

def ordinary_project(source):
    assert sha(canonical(source['events']))==source['events_sha256']
    groups=OrderedDict();goals={};coverage={}
    # Preserve event order from the frozen source; state observations appended
    # by the snapshot remain distinct and do not become invented user goals.
    for event in source['events']:
        kind=event['kind'];body=decoded(event['body']);task=event['task_id']
        if event['role']=='user' and kind=='message':goals[task]=event
        if kind not in KINDS:
            coverage[event['id']]={'status':'archive_only_runtime_metadata'};continue
        if kind=='message' and event['role']!='user':
            coverage[event['id']]={'status':'archive_only_nonuser_message'};continue
        goal=goals.get(task)
        if goal is None:goal=historical_goal(source,event);goals[task]=goal
        key=(task,event.get('window_id'),goal['id'] if goal else None)
        entry=groups.setdefault(key,{'goal':goal,'events':[]})
        entry['events'].append(event);coverage[event['id']]={'status':'task_segment','segment':list(key)}
    episodes=[]
    for key,group in groups.items():
        goal=group['goal'];events=group['events']
        if goal is None:
            episodes.append({'key':list(key),'input':None,'event_ids':[e['id'] for e in events],
              'review_reason':'原记录未提供可核对的当时用户目标；原文保留，未替用户编造目标。'});continue
        body=decoded(goal['body']);goal_type='受控验收要求（不是用户原话）' if isinstance(body,dict) and body.get('source_kind')=='deployment_acceptance_contract' else '用户原文'
        rows=[{'编号':'用户要求','类型':goal_type,'内容':body}]
        anchors={'用户要求':{'event_id':goal['id'],'body_sha256':sha(goal['body'])}}
        seen=set()
        for e in events:
            if e['id']==goal['id']:continue
            value=decoded(e['body']);kind=KINDS[e['kind']]
            # An exact model-result/primary-decision duplicate is bookkeeping,
            # not a second successful action. Do not merge distinct receipts.
            duplicate=(e['role'],kind,canonical(value))
            if e['kind'] in ('model_result','primary_decision') and duplicate in seen:continue
            seen.add(duplicate);ref='记录'+str(len(rows))
            rows.append({'编号':ref,'类型':kind,'内容':value,'归属':'本段归档记录','时间':e.get('created')})
            anchors[ref]={'event_id':e['id'],'body_sha256':sha(e['body']),'role':e['role'],'kind':e['kind']}
        episodes.append({'key':list(key),'input':{'原始记录':rows},'source_anchors':anchors,'event_ids':[e['id'] for e in events]})
    assert set(coverage)=={e['id'] for e in source['events']}
    return episodes,coverage

def project(source):
    from seven_projection import project as projection
    episodes,coverage=projection(source,ordinary_project)
    from behavior_targets import project as action_project
    return episodes+action_project(source),coverage

def build_notes(work,source,config,tokenizer,call=request,before_call=None):
    options=config['memory_teacher'];assert options['projection_version']==VERSION and options['enable_thinking'] is False
    cache=Path(options['cache_dir'])/(Path(work).name+'-'+source['events_sha256'][:16]);cache.mkdir(parents=True,exist_ok=True)
    directory=Path(work)/'teacher'/VERSION;directory.parent.mkdir(parents=True,exist_ok=True)
    if not directory.exists():directory.symlink_to(cache,target_is_directory=True)
    assert directory.resolve()==cache.resolve(),'Auxiliary cache directory changed; preserve old records separately'
    episodes,coverage=project(source);save(directory/'projection.json',{'episodes':episodes,'coverage':coverage,'source_events_sha256':source['events_sha256']})
    notes=[];audits=[];started=time.monotonic()
    endpoint=options.get('extractor_api','http://127.0.0.1:18815');budget=int(options.get('input_token_budget',6000))
    for i,episode in enumerate(episodes):
        path=directory/f'episode-{i:05d}.json';identity=sha(canonical(episode))
        if path.exists() and json.loads(path.read_text()).get('status')!='technical_error':
            audit=json.loads(path.read_text());assert audit['episode_sha256']==identity and audit['thinking_enabled'] is False
            assert audit['helper_adapter_sha256']==options['helper_adapter_sha256']
            assert audit['runtime_manifest_sha256']==options['runtime_manifest_sha256']
        else:
            audit={'episode_sha256':identity,'helper_adapter_sha256':options['helper_adapter_sha256'],'thinking_enabled':False,'training_eligible':False,'runtime_manifest_sha256':options['runtime_manifest_sha256']}
            payload=episode['input']
            from helper_learning_filter import archive_reason
            reason=archive_reason(episode,config)
            if reason:audit.update(status='archive_only',archive_reason=reason)
            elif payload is None:audit.update(status='needs_review',review_reasons=[episode['review_reason']])
            elif len(payload['原始记录'])>64 or len(tokenizer.encode(canonical(payload),add_special_tokens=False))>budget:
                audit.update(status='needs_review',review_reasons=['完整任务片段超限，需按关联操作拆分；本次未截断或拼造上下文。'])
            else:
                if before_call:before_call()
                try:
                    while True:
                        try:
                            audit['result']=call(endpoint,'/api/internal/helper-extract',{'input':payload},timeout=options.get('request_timeout_seconds',300));audit['status']=audit['result']['status'];break
                        except urllib.error.HTTPError as error:
                            if error.code not in (409,503):raise
                            from reasoning_policy import busy
                            if not busy(options.get('idle_databases',[])):raise
                            time.sleep(5)
                except (OSError,ValueError,KeyError) as error:audit.update(status='technical_error',error=str(error))
            save(path,audit)
        if audit.get('result',{}).get('record'):
            result=audit['result'];notes.append({'id':result['id'],'seq':result['id'],'task_id':episode['key'][0],
             'window_id':episode['key'][1],'role':'memory_teacher','kind':'memory_experience_proposal',
             'body':canonical({**result,'source_anchors':episode.get('source_anchors',{}),'teacher_lora_scale':1,
                'teacher_adapter_sha256':options['helper_adapter_sha256'],'interpretation_status':'unverified_experience_proposal'}),
             'created':'offline-source-linked-experience'})
        audits.append(audit);save(directory/'progress.json',{'phase':'extracting','completed_episodes':len(audits),'total_episodes':len(episodes),'thinking_enabled':False})
    report={'enabled':True,'completed':True,'version':VERSION,'thinking_enabled':False,
      'helper_adapter_sha256':options['helper_adapter_sha256'],'packets':len(audits),'notes':len(notes),
      'model_calls':sum('result' in a for a in audits),'archive_only':sum(a['status']=='archive_only' for a in audits),
      'packets_with_errors':sum(a['status']=='technical_error' for a in audits),
      'needs_review':sum(a['status']=='needs_review' for a in audits),'seconds':round(time.monotonic()-started,3),
      'source_events_sha256':source['events_sha256'],'all_raw_archives_retained':True,
      'main_behavior_targets_generated':False,'training_eligible':False,'quality_evaluation':False}
    save(directory/'report.json',report);save(directory/'notes.json',notes)
    return notes,report
