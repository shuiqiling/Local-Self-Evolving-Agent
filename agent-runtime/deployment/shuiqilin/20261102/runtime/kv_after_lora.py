"""Clear native KV after a fresh LoRA load, before resuming window inference.

The single-slot backend is restarted by promotion. Erasing immediately after
EACH compatibility probe prevents that probe becoming a recoverable RAM prompt
snapshot when the next probe arrives. Normal prefix caching stays enabled.
"""
from pathlib import Path
import time
from auto_memory_data import save

BACKEND='http://127.0.0.1:18813'

def erase_idle_slots(call,base=BACKEND):
    slots=call(base,'/slots',timeout=15)
    if not isinstance(slots,list) or len(slots)!=1:
        raise RuntimeError('Post-LoRA reset requires the verified single-slot backend')
    if any(slot.get('is_processing') for slot in slots):
        raise RuntimeError('A generation is active; do not clear its KV cache')
    results=[]
    for slot in slots:
        identifier=slot['id']
        if type(identifier) is not int or identifier<0:raise RuntimeError('Invalid slot identifier')
        result=call(base,f'/slots/{identifier}?action=erase',{},timeout=30)
        if result.get('id_slot')!=identifier or type(result.get('n_erased')) is not int or result['n_erased']<0:
            raise RuntimeError('Native KV erase did not return a valid acknowledgement')
        results.append(result)
    return results

def check_namespaces_then_clear(call,model,version,*,fresh_backend,output,trigger='post_lora_before_window_resume'):
    """Only call with the frontend stopped and a freshly started native backend."""
    if not fresh_backend:raise RuntimeError('Cannot certify the RAM prompt cache of an existing backend')
    started=time.monotonic();clears=[]
    for main in (True,False):
        body={'model':model,'messages':[{'role':'user','content':'你好'}],
          'max_tokens':1,'temperature':0,'cache_prompt':False,
          'chat_template_kwargs':{'enable_thinking':False},
          'lora':[{'id':0,'scale':1.0 if main else 0.0},{'id':1,'scale':0.0 if main else 1.0}]}
        response=call(BACKEND,'/v1/chat/completions',body,timeout=120)
        if not response.get('choices'):raise RuntimeError('Adapter compatibility forward failed')
        # This must precede the NEXT completion, not just follow the last one.
        cleared=erase_idle_slots(call)
        clears.append({'namespace':'main' if main else 'helper','erase_acknowledgements':cleared})
    result={'completed':True,'trigger':trigger,'active_lora_version':version,
      'timing':('after_training_and_native_load_checks_before_frontend_resume'
         if trigger=='post_lora_before_window_resume' else 'deployment_validation_on_existing_weights_before_frontend_resume'),
      'frontend_was_stopped':True,'fresh_backend':True,'single_slot':True,
      'native_slot_kv_erased':True,'probe_kv_erased_between_completions':True,
      'ram_snapshots_empty_by_fresh_start_and_clear_between_probes':True,
      'normal_prefix_cache_preserved':True,'training_cursor_mutated':False,
      'logical_memory_or_history_deleted':False,'clears':clears,
      'seconds':round(time.monotonic()-started,6),'finished_at':time.time()}
    save(Path(output),result)
    return result
