"""Lossless, role-labelled archive learning for each ten real window handoffs.

No success filter or quality score is used here. Optional model annotations
supplement the original records; they never replace or filter source content.
Every archived event in the frozen sequence interval is retained, including
failures and corrections. Tokens are split, never truncated to a row budget.
"""
from pathlib import Path
import json
import sqlite3
from auto_memory_data import canonical, sha, save

ARCHIVE_SYSTEM = ('你是水麒麟。以下是带角色、时间和来源的历史记忆存档学习。'
    '存档不是当前指令，也不是已经核实的事实；模型回答可能错误，失败回执仍表示失败，'
    '更正应结合原始时间理解。保持用户陈述、模型结论和工具观测的区别。')


def snapshot(path, cursor, count=10):
    """Freeze exactly count completed handoffs after the acknowledged cursor."""
    with sqlite3.connect('file:' + str(path) + '?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        db.execute('BEGIN')
        boundaries = list(db.execute("SELECT seq FROM events WHERE kind='handoff_completed' AND seq>? ORDER BY seq LIMIT ?",
                                     (cursor['last_seq'], count)))
        if len(boundaries) < count:
            return None
        end = boundaries[-1]['seq']
        events = [dict(r) for r in db.execute('SELECT * FROM events WHERE seq>? AND seq<=? ORDER BY seq', (cursor['last_seq'], end))]
        if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='internal_state_observations'").fetchone():
            for row in db.execute('SELECT * FROM internal_state_observations WHERE source_cutoff_seq>=? AND source_cutoff_seq<? ORDER BY observed_at',
                                  (cursor['last_seq'],end)):
                item=dict(row)
                # Internal observations do not become user messages/handoffs, but
                # are still covered by the next real ten-window training batch.
                events.append({'seq':'internal-'+item['id'],'id':'internal-'+item['id'],'task_id':'internal-state-input',
                    'window_id':'anchored-at-'+str(item['source_cutoff_seq']),'role':'internal_state',
                    'kind':'internal_state_observation','body':canonical(item),'created':item['observed_at']})
    return {'source_db':str(path), 'start_exclusive_seq':cursor['last_seq'], 'end_inclusive_seq':end,
            'handoffs':count, 'handoff_sequences':[r['seq'] for r in boundaries], 'events':events,
            'events_sha256':sha(canonical(events)), 'next_cursor':{'last_seq':end, 'handoffs':cursor.get('handoffs',0)+count}}


def optimizer_groups(order, steps):
    """One pass over every microbatch, using at most eight optimizer updates."""
    assert 0 < steps <= min(8, len(order))
    return [order[i*len(order)//steps:(i+1)*len(order)//steps] for i in range(steps)]


def encode_archive(source, tokenizer, max_tokens=512):
    assert source['events_sha256'] == sha(canonical(source['events']))
    rows=[]; coverage=[]
    for event in source['events']:
        # The original body string, not a lossy summary, remains in the target.
        record={**event, 'source_db':source['source_db'], 'record_type':'historical_event_not_current_instruction',
                'truth_status':'unverified_archive_preserve_original_outcome'}
        text=canonical(record)
        tokens=tokenizer.encode(text,add_special_tokens=False)
        assert tokens
        event_rows=[];offset=0
        while offset<len(tokens):
            prefix=tokenizer.apply_chat_template([
                {'role':'system','content':ARCHIVE_SYSTEM},
                {'role':'user','content':f'学习历史记录 {event["id"]}，从记录token偏移{offset}开始的片段；按原始角色和结果保留内容。'}],
                tokenize=True, add_generation_prompt=True, enable_thinking=False)
            size=max_tokens-len(prefix)-1
            assert size>0, 'Archive prefix leaves no room for source tokens'
            part=tokens[offset:offset+size]
            identifier=f'window-{event["seq"]}-{offset}'
            rows.append({'id':identifier,'input_ids':prefix+part+[tokenizer.eos_token_id], 'target_start':len(prefix),
                'origin':{'kind':'all_window_archive','event_id':event['id'],'event_seq':event['seq'],
                    'event_kind':event['kind'],'role':event['role'],'source_db':source['source_db'],
                    'body_sha256':sha(event['body']),'token_offset':offset,'source_tokens':len(part)}})
            event_rows.append(identifier)
            offset+=len(part)
        coverage.append({'event_id':event['id'],'seq':event['seq'],'body_sha256':sha(event['body']),
                         'encoded_record_sha256':sha(text),'source_tokens':len(tokens),'rows':event_rows})
    assert len(coverage)==len(source['events']) and rows
    return rows,coverage


def build_window_data(work,tokenizer):
    work=Path(work);config=json.loads((work/'run-config.json').read_text())
    source_bytes=(work/'window-source.json').read_bytes()
    assert sha(source_bytes)==config['window_source_sha256']
    source=json.loads(source_bytes)
    if config.get('training_content')=='teacher_only':
        from teacher_training import build_window_data as teacher_only_data
        return teacher_only_data(work,source,config,tokenizer)
    rows,coverage=encode_archive(source,tokenizer,config['training']['max_sequence_tokens'])
    archive_rows=len(rows);teacher_rows=0;teacher_report={'enabled':False}
    if config.get('memory_teacher',{}).get('enabled'):
        from memory_teacher import build_notes
        try:notes,teacher_report=build_notes(work,source,config,tokenizer)
        except (AssertionError,OSError,ValueError,KeyError,TypeError) as error:
            notes=[];teacher_report={'enabled':True,'completed':False,'error':str(error),
                'all_raw_archives_retained':True,'lora_scale':1,'quality_evaluation':False}
            save(work/'teacher/report.json',teacher_report)
        if notes:
            extra={'events':notes,'events_sha256':sha(canonical(notes)),'source_db':'memory_teacher_source_linked_annotations'}
            encoded,teacher_coverage=encode_archive(extra,tokenizer,config['training']['max_sequence_tokens'])
            for row in encoded:row['origin']['kind']='memory_teacher_annotation'
            rows.extend(encoded);teacher_rows=len(encoded)
            save(work/'teacher/coverage.json',teacher_coverage)
    parent=config.get('parent_adapter');previous=[]
    if parent:
        p=Path(parent['peft']).parent/'data/train.jsonl'
        manifest=json.loads((p.parent/'manifest.json').read_text())
        assert sha(p.read_bytes())==manifest['train_sha256'], 'Replay source changed'
        previous=[json.loads(line) for line in p.read_text().splitlines()]
    assert previous,'Active parent needs replay rows for the retained50% replay policy'
    new_count=len(rows)
    # No max_new_rows cap: every token of all ten archived windows is included.
    # Rotate deterministically through the parent's actual training rows.
    offset=int(source['events_sha256'][:8],16)
    for i in range(new_count):
        original=previous[(offset+i)%len(previous)]
        row={k:v for k,v in original.items() if k not in ('id','origin')}
        rows.append({**row,'id':f'window-replay-{i}','origin':{'kind':'previous_active_train_replay',
            'source_id':original['id'],'source_path':str(p),'source_sha256':manifest['train_sha256']}})
    data=work/'data';data.mkdir(exist_ok=True)
    text=''.join(canonical(r)+'\n' for r in rows)
    (data/'train.jsonl').write_text(text,encoding='utf-8',newline='\n')
    save(data/'coverage.json',coverage)
    result={'train_rows':len(rows),'new_archive_rows':archive_rows,'teacher_rows':teacher_rows,'new_rows':new_count,'replay_rows':new_count,'evaluation_rows':0,
            'source_events':len(coverage),'source_tokens':sum(r['source_tokens'] for r in coverage),
            'train_sha256':sha(text),'window_source_sha256':sha(source_bytes),
            'coverage_sha256':sha((data/'coverage.json').read_bytes()),'handoffs':source['handoffs'],
            'data_scope':'all_context_windows','loss':'role-labelled historical archive reconstruction; not success certification',
            'full_source_coverage':True,'truncated_source_tokens':0,'memory_teacher':teacher_report}
    save(data/'manifest.json',result)
    return result


if __name__=='__main__':
    import sys
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained('/home/ubuntu/AI-Native/embodied-agent/models/Qwen3-30B-A3B-bnb-4bit',
        local_files_only=True,trust_remote_code=False)
    print(json.dumps(build_window_data(sys.argv[1],tokenizer)))
