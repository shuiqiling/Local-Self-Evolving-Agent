"""Train only attributed helper lessons; archives remain retrieval evidence."""
from pathlib import Path
import json
from auto_memory_data import canonical,sha,save,format_decision

FORMAT='teacher-only-decision-v1'
SYSTEM=('你是水麒麟。依据给出的历史整理材料提出建议，不当作当前事实。原因只是推测，'
 '建议不代表已执行。仅输出action、tool、arguments、response四字段JSON。')

def qualifies(row):
 """No inherited raw/archive/synthetic replay can masquerade as helper SFT."""
 try:
  origin=row['origin'];messages=row['messages']
  assert origin['format']==FORMAT and origin['kind'] in ('teacher_extracted_lesson','teacher_extracted_replay')
  assert origin['root_kind']=='teacher_extracted_lesson' and origin['teacher_note_sha256']
  assert 'input_ids' not in row and [m['role'] for m in messages]==['system','user','assistant']
  assert messages[0]['content']==SYSTEM
  target=messages[-1]['content'];assert sha(target)==origin['target_sha256']
  value=json.loads(target);assert set(value)=={'action','tool','arguments','response'}
  assert value['action']=='respond' and value['tool']=='' and value['arguments']=={}
  assert isinstance(value['response'],str) and 0<len(value['response'])<=200
  return True
 except (KeyError,TypeError,AssertionError,ValueError):return False

def note_rows(note,source,parent,tokenizer,maximum=512):
 if note.get('kind')=='memory_experience_proposal':
  raise ValueError('Seven-field experience proposals are not verified behavior targets; source review and a separate behavior-sample step are required')
 from teacher_compact import resolve,literal
 body=json.loads(note['body']);originals={e['id']:e for e in source['events']}
 assert body['teacher_lora_scale']==1 and body['teacher_adapter_sha256']==parent['gguf_sha256']
 assert body['source_quote_match'] is True and body['interpretation_status']=='unverified_model_interpretation'
 assert body['correction_status']=='proposal_not_executed_or_verified_by_teacher'
 for quote in body['quotes']:
  original=originals[quote['event_id']];assert sha(original['body'])==quote['body_sha256']
  text=literal(resolve(original,quote['path']))
  assert text[quote['start']:quote['end']]==quote['quote'],'Helper quote differs from its archive'
 cause=body['possible_cause'];next_step=body['proposed_next_step']
 assert isinstance(cause,str) and len(cause)<=80 and isinstance(next_step,str) and len(next_step)<=80
 answer=f'可能原因：{cause}；建议：{next_step}（未执行或验证）。'
 if not cause and not next_step:answer='现有整理材料尚不能确定原因或下一步，需要核对历史原文。'
 evidence=[{'role':q['source_role'],'kind':q['source_kind'],'field':'/'.join(map(str,q['path'])),
    'excerpt':q['quote'],'excerpt_may_be_partial':True} for q in body['quotes']]
 result=[]
 def make(response,part,request):
  target=format_decision({'action':'respond','tool':'','arguments':{},'response':response})
  payload={'current_request':request,'historical_evidence':[]}
  def messages():return [{'role':'system','content':SYSTEM},{'role':'user','content':canonical(payload)},
    {'role':'assistant','content':target}]
  def tokens():
   m=messages();prefix=tokenizer.apply_chat_template(m[:-1],tokenize=True,add_generation_prompt=True,enable_thinking=False)
   suffix=tokenizer.encode(target+tokenizer.eos_token,add_special_tokens=False)
   return len(prefix)+len(suffix)
  assert tokens()<=maximum,'Helper lesson target does not fit the sequence budget'
  # Add only whole attributed excerpts that fit; never split a filename into a
  # synthetic status value, and never truncate the helper's actual lesson.
  for item in evidence:
   payload['historical_evidence'].append(item)
   if tokens()>maximum:payload['historical_evidence'].pop()
  row={'id':note['id']+'-'+part,'messages':messages(),'origin':{'kind':'teacher_extracted_lesson',
   'root_kind':'teacher_extracted_lesson','format':FORMAT,'teacher_note_id':note['id'],
   'teacher_note_sha256':sha(note['body']),'teacher_adapter_sha256':body['teacher_adapter_sha256'],
   'packet_sha256':body['packet_sha256'],'target_sha256':sha(target),'source_quotes':body['quotes'],
   'interpretation_status':body['interpretation_status'],'correction_status':body['correction_status'],
   'sequence_tokens':tokens()}}
  assert qualifies(row);return row
 try:result=[make(answer,'lesson','根据这些历史整理材料，简短说明可能原因和下一步建议。')]
 except AssertionError as error:
  if str(error)!='Helper lesson target does not fit the sequence budget':raise
  if cause:result.append(make('可能原因：'+cause+'（尚未验证）。','cause','根据历史整理材料说明可能原因，保留不确定性。'))
  if next_step:result.append(make('建议：'+next_step+'（尚未执行或验证）。','next','根据历史整理材料提出下一步建议，不冒充已执行。'))
 return result

def validate_dataset(manifest,rows):
 if manifest.get('format')=='source-verified-experience-decision-v1':
  from auto_evidence_training import validate_dataset as verified
  return verified(manifest,rows)
 assert manifest['training_content']=='teacher_only'
 assert manifest['new_archive_rows']==0 and manifest['raw_archive_training_rows']==0
 assert manifest['train_rows']==len(rows) and all(qualifies(r) for r in rows),'Non-helper data entered teacher-only training'
 assert len({r['id'] for r in rows})==len(rows)
 assert manifest['train_rows']==manifest['new_rows']+manifest['replay_rows']
 if not rows:
  helper=manifest['memory_teacher']
  assert helper.get('completed') and helper.get('packets_with_errors',0)==0,'Empty technical failure cannot consume windows'

def build_from_notes(work,source,config,tokenizer,notes,teacher_report,*,preview=False):
 work=Path(work);parent=config['parent_adapter'];new=[]
 for note in notes:new.extend(note_rows(note,source,parent,tokenizer,config['training']['max_sequence_tokens']))
 previous=[];excluded=0;replay_path=None;replay_hash=None
 if parent:
  replay_path=Path(parent['peft']).parent/'data/train.jsonl'
  if replay_path.exists():
   m=json.loads((replay_path.parent/'manifest.json').read_text(encoding='utf-8'));replay_hash=sha(replay_path.read_bytes())
   assert replay_hash==m['train_sha256'],'Replay source changed'
   for line in replay_path.read_text(encoding='utf-8').splitlines():
    row=json.loads(line)
    if qualifies(row):previous.append(row)
    else:excluded+=1
 rows=list(new);offset=int(source['events_sha256'][:8],16);replay_count=0
 if previous and new:
  for i in range(len(new)):
   original=previous[(offset+i)%len(previous)]
   rows.append({**original,'id':f'teacher-replay-{i}','origin':{**original['origin'],
    'kind':'teacher_extracted_replay','source_id':original['id'],'source_path':str(replay_path),'source_sha256':replay_hash}})
  replay_count=len(new)
 data=work/'data';data.mkdir(parents=True,exist_ok=True)
 text=''.join(canonical(r)+'\n' for r in rows);(data/'train.jsonl').write_text(text,encoding='utf-8',newline='\n')
 coverage=[{'event_id':e['id'],'body_sha256':sha(e['body']),'disposition':'archive_only_not_a_training_target'} for e in source['events']]
 save(data/'coverage.json',coverage)
 manifest={'training_content':'teacher_only','data_scope':'all_context_windows','handoffs':source['handoffs'],
  'train_rows':len(rows),'new_rows':len(new),'teacher_rows':len(new),'new_archive_rows':0,'raw_archive_training_rows':0,
  'teacher_notes':len(notes),'replay_rows':replay_count,'eligible_prior_teacher_rows':len(previous),
  'excluded_non_teacher_replay_rows':excluded,'requested_replay_fraction':.5,
  'actual_replay_fraction':replay_count/len(rows) if rows else 0,'evaluation_rows':0,
  'train_sha256':sha(text),'window_source_sha256':sha((work/'window-source.json').read_bytes()) if (work/'window-source.json').exists() else None,
  'source_events':len(source['events']),'source_events_sha256':source['events_sha256'],
  'raw_archives_retained':True,'full_source_coverage':False,'raw_source_not_training_by_owner_decision':True,
  'training_sequence_tokens':sum(r['origin']['sequence_tokens'] for r in rows),
  'empty_training_set':not rows,'preview_only':preview,'memory_teacher':teacher_report,'quality_evaluation':False,
  'loss':'Assistant Decision targets derived only from attributed helper lessons; no full archive reconstruction'}
 validate_dataset(manifest,rows);save(data/'manifest.json',manifest);return manifest

def build_window_data(work,source,config,tokenizer):
 assert config['memory_teacher']['enabled'],'Teacher-only training requires the helper'
 from memory_teacher import build_notes
 notes,report=build_notes(work,source,config,tokenizer)
 assert report.get('completed'),'Helper preparation did not complete; raw fallback is forbidden'
 if not notes and report.get('packets_with_errors',0):raise OSError('No helper lessons after technical annotation errors; do not consume this window batch')
 if config['memory_teacher'].get('projection_version')=='seven-field-experiences-v1':
  assert config.get('automatic_source_review',{}).get('enabled') is True,'Automatic source review is required'
  from auto_evidence_training import build as verified_build
  return verified_build(work,source,config,tokenizer,report)
 return build_from_notes(work,source,config,tokenizer,notes,report)
