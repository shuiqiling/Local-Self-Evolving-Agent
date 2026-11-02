"""Pure rules shared by the learner and the human acceptance API."""
import json
import re
import hashlib
import math
from pathlib import Path

def validate_contract(name,cases):
    if not isinstance(name,str) or not re.fullmatch(r'[a-z][a-z0-9_]{1,39}',name):raise ValueError('Invalid tool name')
    if not isinstance(cases,list) or not 3<=len(cases)<=64:raise ValueError('Provide3–64 fixed owner acceptance cases')
    if any(not isinstance(c,dict) or set(c)!={'input','expected'} or not isinstance(c['input'],dict) for c in cases):raise ValueError('Each case needs input object and expected')
    text=json.dumps(cases,ensure_ascii=False,sort_keys=True,allow_nan=False)
    if len(text)>32000:raise ValueError('Acceptance cases too large')
    if len({json.dumps(c['input'],sort_keys=True,allow_nan=False) for c in cases})<3:raise ValueError('At least3 different inputs required')
    return cases

def model_gate(before,after):
    original={r['id']:r for r in before['rows']};current={r['id']:r for r in after['rows']}
    if set(original)!=set(current):raise ValueError('Paired test cases differ')
    regressions=[key for key,row in current.items() if original[key]['passed'] and not row['passed']]
    gain=sum(r['passed'] for r in current.values() if r['kind']!='retention')-sum(r['passed'] for r in original.values() if r['kind']!='retention')
    ready=after['scores']['workflow']['passed']>=8 and after['scores']['retention']['passed']==4 and gain>0 and not regressions
    return {'ready':ready,'decision_gain':gain,'regressed_cases':regressions}

def ensure_procedure(store,task_id,guide):
    if not guide.get('validated') or hashlib.sha256(guide['body'].encode()).hexdigest()!=guide['body_sha256']:raise ValueError('Procedure memory integrity mismatch')
    with store.db() as db:
        original=db.execute('SELECT body FROM events WHERE id=? AND task_id=?',(guide['event_ref'].split(':')[-1],guide['task_id'])).fetchone()
        if not original or json.loads(original[0]).get('text')!=guide['body']:raise ValueError('Procedure does not match its immutable archive')
        exists=db.execute("SELECT 1 FROM events WHERE task_id=? AND kind='tool_result' AND json_extract(body,'$.unified_procedure')=1 AND json_extract(body,'$.source_sha')=?",(task_id,guide['body_sha256'])).fetchone()
    if exists:return False
    store.append(task_id,'tool','tool_result',{'text':'当前工具接口的开发流程记忆，仅说明规则，不代表已执行：'+guide['body'],
        'unified_procedure':True,'memory_kind':'procedure_not_execution_result','source_ref':guide['event_ref'],'source_sha':guide['body_sha256']})
    return True

def current_epoch(observations,config):
    return [o for o in observations if o['source_db']==config['epoch_live_db'] and o.get('latest_test_seq',0)>config['epoch_min_seq']]

def runtime_fingerprint(root):
    root=Path(root)
    values={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ('unified_tools.py','unified_schema.py','unified_auto_agent.py','auto_memory_policy.py')}
    if (root/'compact_primary.py').exists():values['compact_primary.py']=hashlib.sha256((root/'compact_primary.py').read_bytes()).hexdigest()
    if (root/'light_lora_policy.py').exists():values['light_lora_policy.py']=hashlib.sha256((root/'light_lora_policy.py').read_bytes()).hexdigest()
    for name in ('shared_context.py','state_budget.py','active_context.py','active_memory.py','idle_sessions.py','idle_development.py','app/lea/embedding.py','idle_autonomy.py','state_view.py','machine_state.py','idle_state_input.py','app/lea/runtime.py','training_priority.py','auto_memory_worker.py','reasoning_policy.py','memory_teacher.py','teacher_semantic.py','teacher_training.py','window_memory_data.py','light_lora_pipeline.py','auto_memory_train.py','seven_projection.py','auto_evidence_training.py','seven_field_teacher.py','dual_adapter_promotion.py','kv_after_lora.py','behavior_targets.py','helper_off_core.py','continuity_state.py','trace_view.py','execution_recovery.py','budget_prompt.py','receipt_reporting.py','write_receipts.py','unified_agent_v11.py','app/lea/model.py','app/lea/app.py','app/lea/static/index.html','app/lea/tools.py','app/lea/store.py','helper_evidence.py','helper_learning_filter.py','helper_off_routes.py','navigation_contract.py','navigation_runtime.py','app/lea/evidence.py'):
        if (root/name).exists():values[name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
    return hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest()

def validate_runtime_adapter(value,allow_candidate=False):
    """Owner selection is distinct from passing the automatic quality gate."""
    purpose=value.get('purpose')
    if purpose=='load_checked':
        decision=Path(value['mode_decision']);report=Path(value['load_check_report'])
        if hashlib.sha256(decision.read_bytes()).hexdigest()!=value['mode_decision_sha256']:raise ValueError('Light mode decision changed')
        mode=json.loads(decision.read_text())
        if not (mode.get('source')=='explicit_user_request' and mode.get('allow_load_checked_promotion') is True):raise ValueError('Light promotion is not authorized')
        if hashlib.sha256(report.read_bytes()).hexdigest()!=value['load_check_sha256']:raise ValueError('Load check report changed')
        check=json.loads(report.read_text())
        if not (check.get('loadable') is True and check.get('native_loaded') is True and check.get('generation_api_ok') is True and
                check.get('finite_weights') is True and check.get('budget_checked') is True and check.get('gguf_sha256')==value['gguf_sha256']):
            raise ValueError('Adapter has not passed load checks')
        rate=check.get('max_learning_rate',0)
        if not (0<check.get('step_count',0)<=8 and check.get('epochs')==1 and math.isfinite(rate) and 0<rate<=5e-6
                and check.get('peft_sha256')==value.get('peft_sha256')):raise ValueError('Light training budget or provenance mismatch')
        if value.get('accepted') is not False or value.get('quality_evaluated') is not False:raise ValueError('Load checks are not quality evaluation')
    elif purpose=='isolated_evaluation':
        if not allow_candidate:raise ValueError('Unaccepted adapter requires isolated evaluation')
    else:
        report=Path(value['acceptance_report'])
        if hashlib.sha256(report.read_bytes()).hexdigest()!=value['acceptance_sha256']:raise ValueError('Acceptance report changed')
        accepted=json.loads(report.read_text())['accepted']
        if purpose=='owner_selected':
            decision=Path(value['owner_decision'])
            if hashlib.sha256(decision.read_bytes()).hexdigest()!=value['owner_decision_sha256']:raise ValueError('Owner decision changed')
            owner=json.loads(decision.read_text())
            if not (owner.get('source')=='explicit_user_request' and owner.get('action')=='activate_candidate' and
                    owner.get('allow_activation') is True and owner.get('candidate_sha256')==value['gguf_sha256'] and
                    owner.get('acceptance_sha256')==value['acceptance_sha256'] and bool(owner.get('instruction'))):
                raise ValueError('Owner decision does not authorize this adapter')
            if value.get('accepted') is not accepted:raise ValueError('Owner selection must preserve actual quality outcome')
        elif value.get('accepted') is not True or accepted is not True:raise ValueError('Adapter has not passed acceptance')
    if value.get('model')!='Qwen3-30B-A3B-Q4_K_M':raise ValueError('Wrong adapter architecture')
    if hashlib.sha256(Path(value['gguf']).read_bytes()).hexdigest()!=value['gguf_sha256']:raise ValueError('Adapter weights changed')
    return value

def validate_parent(value):
    if value is None:return
    if value.get('purpose')=='load_checked':
        validate_runtime_adapter(value)
        if hashlib.sha256((Path(value['peft'])/'adapter_model.safetensors').read_bytes()).hexdigest()!=value['peft_sha256']:raise ValueError('Parent PEFT adapter changed')
        if value.get('peft_config_sha256') and hashlib.sha256((Path(value['peft'])/'adapter_config.json').read_bytes()).hexdigest()!=value['peft_config_sha256']:raise ValueError('Parent PEFT configuration changed')
        return
    owner_parent=False
    if value.get('accepted') is not True:
        if value.get('purpose')!='owner_selected':raise ValueError('Training parent is not accepted')
        validate_runtime_adapter(value)
        owner=json.loads(Path(value['owner_decision']).read_text())
        if not (value.get('owner_training_parent') is True and owner.get('allow_as_training_parent') is True):
            raise ValueError('Owner-selected parent is not accepted for automatic training')
        owner_parent=True
    for key,digest_key in (('gguf','gguf_sha256'),('acceptance_report','acceptance_sha256')):
        if hashlib.sha256(Path(value[key]).read_bytes()).hexdigest()!=value[digest_key]:raise ValueError('Training parent changed')
    if hashlib.sha256((Path(value['peft'])/'adapter_model.safetensors').read_bytes()).hexdigest()!=value['peft_sha256']:raise ValueError('Parent PEFT adapter changed')
    if not owner_parent and json.loads(Path(value['acceptance_report']).read_text())['accepted'] is not True:raise ValueError('Parent quality gate did not pass')

def export_base_config(source,directory):
    source=Path(source);directory=Path(directory);raw=source.read_bytes();config=json.loads(raw)
    if config.get('architectures')!=['Qwen3MoeForCausalLM']:raise ValueError('Unexpected adapter architecture')
    cleaned={k:v for k,v in config.items() if k!='quantization_config'}
    directory.mkdir(exist_ok=True);target=directory/'config.json'
    if target.exists() and json.loads(target.read_text())!=cleaned:raise ValueError('Export config changed')
    target.write_text(json.dumps(cleaned,ensure_ascii=False,indent=2),encoding='utf-8')
    return {'source_config_sha256':hashlib.sha256(raw).hexdigest(),'export_config_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
        'removed_metadata':['quantization_config'] if 'quantization_config' in config else [],'source_config_unchanged':source.read_bytes()==raw,
        'export_directory':str(directory),'base_weights_merged_or_dequantized':False,'purpose':'LoRA-only export architecture parameters; contains no base model weights'}
