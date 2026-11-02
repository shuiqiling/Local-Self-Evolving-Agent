"""A full real-window batch reserves the background inference lane for LoRA."""
from pathlib import Path
import json,re,sqlite3,subprocess,time


def unit_active(unit):
    if not re.fullmatch(r'lea-shuiqilin-windows-\d{8}-\d{6}(?:\.service)?',unit):
        return False
    result=subprocess.run(['systemctl','--user','is-active',unit],capture_output=True,text=True,timeout=5)
    return result.stdout.strip() in ('active','activating','reloading','deactivating')


def status(auto, config, *, active_reader=unit_active, now=None):
    """Read actual boundaries and pipeline state; no counter or cursor mutation."""
    auto=Path(auto);now=time.time() if now is None else now
    result={'enabled':config.get('background_inference_yields_to_lora',False),
            'hold_background_inference':False,'phase':'normal_background_schedule',
            'foreground_chat_preempts_training':True,'checked_at':now}
    if not result['enabled']:
        return result
    pending=auto/'pending-window-batch.json'
    if pending.exists():
        batch=json.loads(pending.read_text(encoding='utf-8'));work=Path(batch['work'])
        path=work/'pipeline-report.json';progress=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        result.update(unit=batch['unit'],work=str(work),pipeline_phase=progress.get('phase','starting'),
                      next_cursor=batch['next_cursor'])
        if progress.get('completed') and progress.get('activated'):
            return {**result,'hold_background_inference':True,'phase':'completed_batch_awaiting_acknowledgement'}
        if progress.get('completed') and progress.get('skipped') and progress.get('training_content')=='teacher_only' and progress.get('skip_reason')=='no_teacher_lessons':
            return {**result,'hold_background_inference':True,'phase':'skipped_batch_awaiting_acknowledgement'}
        if progress.get('phase')=='failed':
            return {**result,'phase':'training_failed_background_resumed','error':progress.get('error')}
        age=now-batch.get('created_at',pending.stat().st_mtime)
        if age>60:
            try:alive=active_reader(batch['unit'])
            except (OSError,subprocess.SubprocessError):alive=True
            if not alive:
                return {**result,'phase':'training_unit_stopped_background_resumed',
                        'error':'Pending training unit is not active; keep batch and cursor for inspection'}
        return {**result,'hold_background_inference':True,'phase':'window_batch_in_progress'}
    if not config.get('enabled') or config.get('data_scope')!='all_context_windows':
        return result
    cursor=json.loads((auto/'window-cursor.json').read_text(encoding='utf-8'))
    count=config.get('min_new_handoffs',10);assert count==10
    path=Path(config['live_data'])/'memory.sqlite3'
    with sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True) as db:
        boundaries=[r[0] for r in db.execute("SELECT seq FROM events WHERE kind='handoff_completed' AND seq>? ORDER BY seq LIMIT ?",
                                           (cursor['last_seq'],count))]
    result.update(cursor=cursor,required_windows=count,available_windows_up_to_batch=len(boundaries))
    if len(boundaries)==count:
        result.update(hold_background_inference=True,phase='waiting_for_background_inference_to_finish',
                      end_inclusive_seq=boundaries[-1],handoff_sequences=boundaries)
    return result
