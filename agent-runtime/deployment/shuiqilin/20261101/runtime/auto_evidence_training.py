"""Automatically verified source-grounded SFT, without supervising helper judgments.

No correct action is invented from a failed attempt. Gold answers are literal
receipts or deterministic telemetry statistics, explicitly historical. The
helper's free-form impact, cause, advice and model claims are NEVER gold text.
"""
from pathlib import Path
import json
import sqlite3
import behavior_targets as B
from auto_memory_data import canonical, sha, save, format_decision
import helper_off_core as H
import helper_off_contract as C
from seven_projection import aggregate, STATE_GOAL, VERSION as PROJECTION

FORMAT = 'source-verified-experience-decision-v1'
SYSTEM = ('你是水麒麟。按当前请求回答所给历史证据；保留条件、失败、否定和未知，不把历史观测当当前事实。'
 '只输出action、tool、arguments、response四字段JSON；不声称执行证据外的操作。')

def sequence_tokens(messages, tokenizer):
    return len(tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True, enable_thinking=False)) + len(tokenizer.encode(messages[-1]['content']+tokenizer.eos_token, add_special_tokens=False))

def make_row(identifier, request, evidence, response, proof, tokenizer, maximum):
    if len(response) > 240: raise ValueError('complete_answer_exceeds_concise_budget')
    target = format_decision({'action': 'respond', 'tool': '', 'arguments': {}, 'response': response})
    messages = [{'role':'system','content':SYSTEM}, {'role':'user','content':canonical({
      'current_request':request, 'historical_evidence':evidence})}, {'role':'assistant','content':target}]
    length = sequence_tokens(messages, tokenizer)
    if length > maximum: raise ValueError('complete_conditions_and_receipt_exceed_sequence_budget')
    content_hash = sha(canonical(messages))
    return {'id': identifier, 'messages':messages, 'origin':{'format':FORMAT,
      'kind':'source_verified_experience', 'root_kind':'source_verified_experience',
      'target_sha256':sha(target), 'content_sha256':content_hash, 'sequence_tokens':length,
      'verification':'deterministic_source_replay_not_semantic_certification', 'proof':proof}}

def qualifies(row):
    if row.get('origin',{}).get('format') == B.VERSION:return B.qualifies(row)
    try:
        origin = row['origin']; messages = row['messages']
        assert origin['format'] == FORMAT and origin['root_kind'] == 'source_verified_experience'
        assert origin['kind'] in ('source_verified_experience','source_verified_replay')
        assert origin['verification'] == 'deterministic_source_replay_not_semantic_certification'
        assert 'input_ids' not in row and [m['role'] for m in messages] == ['system','user','assistant']
        assert messages[0]['content'] == SYSTEM
        assert sha(messages[-1]['content']) == origin['target_sha256']
        assert sha(canonical(messages)) == origin['content_sha256']
        value = json.loads(messages[-1]['content'])
        assert set(value) == {'action','tool','arguments','response'}
        assert value['action'] == 'respond' and value['tool'] == '' and value['arguments'] == {}
        assert 0 < len(value['response']) <= 240
        proof = origin['proof']
        assert proof['source_events_sha256'] and proof['helper_audit_sha256'] and proof['helper_adapter_sha256']
        assert proof['gold_type'] in ('literal_historical_receipt','deterministic_historical_sampling')
        return True
    except (KeyError, TypeError, AssertionError, ValueError): return False

def original_event(source, identifier):
    values = [e for e in source['events'] if e['id'] == identifier]
    if values: return values[0]
    # A preceding user goal is the only permitted source outside the frozen interval.
    path = source.get('source_db')
    if not path: raise ValueError('missing_source_goal')
    with sqlite3.connect('file:'+str(path)+'?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        row = db.execute("SELECT * FROM events WHERE id=? AND role='user' AND kind='message' AND seq<=?",(identifier,source['start_exclusive_seq'])).fetchone()
    if row is None: raise ValueError('missing_source_goal')
    return dict(row)

def verify_episode(source, episode, stored, options):
    if stored.get('status') == 'technical_error': raise OSError('Auxiliary technical error; this batch must not consume windows')
    if stored.get('status') != 'proposal' and not (episode.get('behavior') and stored.get('status')=='needs_review'): raise ValueError('helper_did_not_propose_a_complete_record')
    result = stored['result']; audit_path = Path(result['audit_path'])
    # The frontend audit is pinned to the same runtime, same original payload and actual response.
    if not audit_path.resolve().is_relative_to(Path(options['audit_dir']).resolve()): raise ValueError('audit_outside_pinned_directory')
    raw = audit_path.read_bytes(); audit = json.loads(raw)
    assert audit['input'] == episode['input']
    expected_request = H.request_for(episode['input'], options['helper_manifest'])
    assert audit['request'] == expected_request
    replayed = H.finish(episode['input'], audit['response'],options['helper_manifest'].get('extraction_mode'))
    issues=replayed.get('review_reasons',[])
    impact_only=(bool(episode.get('behavior')) and replayed['status']=='needs_review' and bool(issues)
      and all(x.startswith('overlong_impact:') for x in issues) and bool(replayed.get('source_trace')))
    if not (replayed['status']=='proposal' or impact_only):raise ValueError('helper_evidence_plan_not_usable')
    assert replayed['status']==result['status'] and replayed['record']==result['record']
    assert replayed.get('review_reasons')==result.get('review_reasons')
    assert replayed['selection_plan'] == result['selection_plan'] and replayed['source_trace'] == result['source_trace']
    assert result['helper_adapter_sha256'] == options['helper_adapter_sha256']
    if episode.get('state_aggregate'):
        value, anchors = aggregate(source)
        assert episode['input']['原始记录'][0]['内容'] == STATE_GOAL
        assert episode['input']['原始记录'][1]['内容'] == value
        assert episode['source_anchors']['记录1']['sources'] == anchors
    else:
        for row in episode['input']['原始记录']:
            anchor = episode['source_anchors'][row['编号']]
            original = original_event(source, anchor['event_id'])
            assert sha(original['body']) == anchor['body_sha256']
            try: content = json.loads(original['body'])
            except ValueError: content = original['body']
            if anchor.get('projection'):
                assert episode.get('behavior'),'Operation projections are limited to behavior episodes'
                content=B.projected_body(original,anchor['projection'])
            assert row['内容'] == content
            if row['编号'] == '用户要求':
                assert original['role'] == 'user' and original['kind'] == 'message'
                assert original['task_id'] == episode['key'][0]
    return replayed, {'source_events_sha256':source['events_sha256'], 'helper_audit_path':str(audit_path),
       'helper_audit_sha256':sha(raw), 'helper_adapter_sha256':options['helper_adapter_sha256'],
       'projection_version':PROJECTION, 'source_anchors':episode['source_anchors'],
       'helper_record_status':replayed['status'],'helper_impact_excluded_from_gold':True,
       'helper_review_reasons':replayed.get('review_reasons',[])}

def rows_for_episode(source, episode, replayed, proof, tokenizer, maximum):
    if episode.get('behavior'):return B.make_rows(episode,replayed,proof,tokenizer,maximum)
    by = C.records(episode['input']); plan = replayed['source_trace']['normalized_selection_plan']
    identifier = sha(canonical(episode))[:24]; result = []
    if episode.get('state_aggregate'):
        assert '记录1' in plan['结果']
        data = by['记录1']['内容']
        # One distinct question per complete metric family; no dropping a null or version.
        for group, family in data['metrics'].items():
          for metric, statistics in family.items():
            metrics={metric:statistics}
            facts = {'sample_count':data['sample_count'], 'first_at':data['first_at'], 'last_at':data['last_at'],
              'measurement':data['measurement'], group:metrics}
            response = '历史采样统计：'+canonical({group:metrics})
            evidence = {'scope':'仅上述历史时段；min/max/last及缺测计数，不代表当前状态；流量不是可用带宽。', **facts}
            try: result.append(make_row(identifier+'-'+group+'-'+metric, '这批历史采样的'+group+'.'+metric+'返回了哪些统计？缺测保持未知。', evidence,
                  response, {**proof,'gold_type':'deterministic_historical_sampling'}, tokenizer, maximum))
            except ValueError: pass
        return result
    # Use actual runtime/tool receipts, never assistant claims or the helper's impact.
    refs = [r for r in plan['结果'] if r != '用户要求']
    context_refs = list(dict.fromkeys(['用户要求',*plan['条件'],*plan['发生']]))
    for ref in refs:
        anchor = episode['source_anchors'][ref]; event = original_event(source, anchor['event_id'])
        if event['role'] not in ('tool','runtime') or event['kind'] not in ('tool_result','archive_read_result','retrieval_result','job_error'): continue
        # Full goal and selected conditions are kept, including versions and request parameters.
        context = [by[r] for r in context_refs if r != ref and r in by]
        audits=[]; body = C.clean(by[ref]['内容'], audits, (), {}, {})
        response = '历史'+by[ref]['类型']+'：'+(body if isinstance(body,str) else canonical(body))
        evidence = {'conditions':context,'receipt':{**by[ref],'内容':body},
          'scope':'这是历史回执；检索/读档只证明取回材料，提交不证明测试，工具成功仅证明该动作。'}
        try: result.append(make_row(identifier+'-'+ref,'按当时目标和条件，准确复述这条历史回执；不扩大为任务成功。',evidence,
          response,{**proof,'gold_type':'literal_historical_receipt','receipt_ref':ref,'receipt_kind':event['kind']},tokenizer,maximum))
        except ValueError: pass
    return result

def build(work, source, config, tokenizer, teacher_report):
    work=Path(work); options=dict(config['memory_teacher']); options['helper_manifest']=H.manifest(options['runtime_manifest'])
    options['audit_dir']=options['helper_manifest']['audit_dir']
    assert H.sha(Path(options['runtime_manifest']).read_bytes())==options['runtime_manifest_sha256']
    directory=work/'teacher'/H.VERSION
    projection=json.loads((directory/'projection.json').read_text())
    assert projection['source_events_sha256']==source['events_sha256']==sha(canonical(source['events']))
    # Reproject the source; a modified cached projection cannot authorize targets.
    from seven_field_teacher import project
    current,_=project(source); assert current==projection['episodes']
    new=[];review=[];seen=set();archived=[]
    from helper_learning_filter import archive_reason,replay_allowed
    for i,episode in enumerate(projection['episodes']):
        stored=json.loads((directory/f'episode-{i:05d}.json').read_text())
        assert stored['episode_sha256']==sha(canonical(episode))
        reason=archive_reason(episode,config)
        if reason:
            assert stored['status']=='archive_only' and stored['archive_reason']==reason
            archived.append({'episode':i,'reason':reason});continue
        try:
            replayed,proof=verify_episode(source,episode,stored,options)
            candidates=rows_for_episode(source,episode,replayed,proof,tokenizer,config['training']['max_sequence_tokens'])
            if not candidates: review.append({'episode':i,'reason':'no_complete_source_grounded_target_within_budget'})
            for row in candidates:
                key=row['origin']['content_sha256']
                if key not in seen:new.append(row);seen.add(key)
        except ValueError as error: review.append({'episode':i,'reason':str(error)})
    # Replay ONLY this source-verified format; never inherited cause/advice notes.
    previous=[];excluded=0;parent=config.get('parent_adapter');replay_path=None;replay_hash=None
    if parent:
        replay_path=Path(parent['peft']).parent/'data/train.jsonl'
        if replay_path.exists():
            old=json.loads((replay_path.parent/'manifest.json').read_text());replay_hash=sha(replay_path.read_bytes());assert replay_hash==old['train_sha256']
            for line in replay_path.read_text().splitlines():
                row=json.loads(line)
                if qualifies(row) and replay_allowed(row,config):previous.append(row)
                else:excluded+=1
    # Identical verified examples do not become new learning just because a window changed.
    prior_hashes={r['origin']['content_sha256'] for r in previous}
    duplicate_new=sum(r['origin']['content_sha256'] in prior_hashes for r in new)
    new=[r for r in new if r['origin']['content_sha256'] not in prior_hashes]
    rows=list(new)
    if new and previous:
        offset=int(source['events_sha256'][:8],16)
        for i in range(len(new)):
            old=previous[(offset+i)%len(previous)]
            rows.append({**old,'id':'verified-replay-'+str(i),'origin':{**old['origin'],'kind':'source_verified_replay',
                'source_id':old['id'],'source_path':str(replay_path),'source_sha256':replay_hash}})
    text=''.join(canonical(r)+'\n' for r in rows); data=work/'data'; data.mkdir(parents=True,exist_ok=True)
    (data/'train.jsonl').write_text(text,encoding='utf-8',newline='\n')
    save(data/'source-review.json',{'review_method':'deterministic_source_replay_not_general_truth_judge',
      'helper_free_form_judgments_supervised':False,'manual_review_required_to_schedule':False,
      'accepted_rows':len(new),'action_counts':action_counts(new),'pending_review':review,'archive_only':archived,'duplicate_new_rows':duplicate_new,'raw_source_retained':True})
    result={'format':FORMAT,'training_content':'teacher_only','data_scope':'all_context_windows','handoffs':source['handoffs'],
      'train_rows':len(rows),'new_rows':len(new),'teacher_rows':len(new),'replay_rows':len(rows)-len(new),
      'eligible_prior_teacher_rows':len(previous),'excluded_non_teacher_replay_rows':excluded,
      'new_archive_rows':0,'raw_archive_training_rows':0,'evaluation_rows':0,'quality_evaluation':False,
      'train_sha256':sha(text),'window_source_sha256':sha((work/'window-source.json').read_bytes()),
      'source_events_sha256':source['events_sha256'],'source_events':len(source['events']),
      'empty_training_set':not rows,'preview_only':False,'source_review_completed':True,
      'rejected_episodes':len(review),'archive_only_episodes':len(archived),'duplicate_new_rows':duplicate_new,'raw_archives_retained':True,'full_source_coverage':False,
      'training_sequence_tokens':sum(r['origin']['sequence_tokens'] for r in rows),
      'requested_replay_fraction':.5,'actual_replay_fraction':(len(rows)-len(new))/len(rows) if rows else 0,
      'memory_teacher':teacher_report,'action_counts':action_counts(new),'behavior_format':B.VERSION,
      'loss':'Source-verified replies and executed actions with checked preconditions; no free-form helper judgments or later outcomes in action inputs'}
    validate_dataset(result,rows);save(data/'manifest.json',result);return result

def validate_dataset(manifest,rows):
    assert manifest['format']==FORMAT and manifest['training_content']=='teacher_only'
    assert manifest['source_review_completed'] is True
    assert manifest['new_archive_rows']==manifest['raw_archive_training_rows']==manifest['evaluation_rows']==0
    assert manifest['train_rows']==len(rows)==manifest['new_rows']+manifest['replay_rows']
    assert len({r['id'] for r in rows})==len(rows) and all(qualifies(r) for r in rows)
    if not rows:
        assert manifest['memory_teacher']['completed'] and manifest['memory_teacher'].get('packets_with_errors',0)==0


def action_counts(rows):
    from collections import Counter
    return dict(Counter((lambda v:v['tool'] if v['action']=='tool' else v['action'])(json.loads(r['messages'][-1]['content'])) for r in rows))
