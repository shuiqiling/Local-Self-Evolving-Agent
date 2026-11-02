"""Application-side idle consolidation worker. It is not a second decision Agent."""
from dataclasses import asdict
from datetime import datetime,timezone,timedelta
from pathlib import Path
import argparse
import fcntl
import json
import os
import shutil
import sqlite3
import subprocess
import time

from auto_memory_data import mine_sources,build,canonical,sha,save
from lea.consolidation import ConsolidationPolicy,assess
from auto_memory_policy import current_epoch,runtime_fingerprint,validate_parent
from auto_memory_eval import request
from light_lora_policy import LIGHT_TRAINING,fresh_experiences,training_plan
from window_memory_data import snapshot

R=Path('/home/ubuntu/AI-Native/embodied-agent');A=R/'auto-lora-shuiqilin'

def readiness(paths,policy):
    busy=0;timestamps=[];handoffs=0
    for path in paths:
        with sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True) as db:
            busy+=db.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]
            from idle_state_input import external_marker
            event=external_marker(db)[1]
            if event:timestamps.append(datetime.fromisoformat(event).timestamp())
            if path==paths[0]:handoffs=db.execute("SELECT COUNT(*) FROM events WHERE kind='handoff_completed'").fetchone()[0]
    available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))/1024**2
    load=os.getloadavg()[0]/max(1,len(os.sched_getaffinity(0)));idle=time.time()-max(timestamps,default=time.time())
    return {'ready':not busy and idle>=policy.min_idle_seconds and available>=policy.min_available_gib and load<=policy.max_normalized_load,
        'foreground_jobs':busy,'idle_seconds':max(0,idle),'available_gib':available,'normalized_load':load,'handoffs':handoffs}

def cycle():
    config=json.loads((A/'config.json').read_text());policy=ConsolidationPolicy(**{k:v for k,v in config.get('policy',{}).items() if k in ConsolidationPolicy.__dataclass_fields__})
    if not config.get('enabled',False):return {'phase':'disabled'}
    from training_priority import status as training_priority
    priority=training_priority(A,config);save(A/'training-priority.json',priority)
    lock=(A/'learner.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        if priority['hold_background_inference']:
            return {**priority,'background_inference_paused':True}
        return {'phase':'model_or_learner_busy'}
    try:
        if config.get('training_mode')=='light':return light_cycle(config,policy,lock)
        parent=json.loads((A/'active.json').read_text()) if (A/'active.json').exists() else None
        if parent and parent.get('accepted') is not True and not (parent.get('purpose')=='owner_selected' and parent.get('owner_training_parent') is True):
            return {'phase':'waiting_for_quality_accepted_parent','current_adapter':parent['version'],
                'selection':parent.get('purpose'),'note':'Owner-selected serving remains active; automatic training never treats a failed candidate as quality accepted.'}
        ready=readiness([Path(config['live_data'])/'memory.sqlite3',R/'data/memory.sqlite3'],policy)
        cursor=json.loads((A/'cursor.json').read_text())
        result={'phase':'waiting_for_idle','readiness':ready,'policy':asdict(policy),'min_new_handoffs':5,
            'last_completed_run':config.get('last_completed_run'),'training_report':config.get('last_training_report')}
        if runtime_fingerprint(R/'experiments/shuiqilin-unified-20261030-r1')!=config['runtime_fingerprint']:return {**result,'phase':'waiting_for_runtime_reconfiguration'}
        if not ready['ready']:return result
        if ready['handoffs']-cursor['handoffs']<5:return {**result,'phase':'waiting_for_five_new_windows'}
        all_observations,_,_=mine_sources(config['source_paths']);observations=current_epoch(all_observations,config);gate=assess(observations,policy)
        gate['historical_observations_excluded_from_current_runtime_gate']=len(all_observations)-len(observations)
        result.update(phase='waiting_for_independent_evidence',gate=gate)
        if [r for r in gate['reasons'] if r!='parametric_value_not_measured']:return result
        consumed=json.loads((A/'consumed.json').read_text()) if (A/'consumed.json').exists() else []
        fresh=[x for x in gate['support_episode_ids'] if x not in consumed]
        if len(fresh)<policy.min_support:return {**result,'phase':'waiting_for_new_evidence'}
        pending=[o for o in observations if o['id'] in fresh]
        if len({o['scope'] for o in pending if o['scope']!='unclassified'})<policy.min_scopes:return {**result,'phase':'waiting_for_new_scope_diversity'}
        parent=json.loads((A/'active.json').read_text()) if (A/'active.json').exists() else None
        validate_parent(parent)
        profile=request('http://127.0.0.1:18815','/api/model-profile',timeout=5)
        health=request('http://127.0.0.1:18815','/api/health',timeout=10)
        if profile.get('primary_adapter')!=(parent['version'] if parent else None) or not health['model']['ready']:return {**result,'phase':'waiting_for_model_identity'}
        corpus=sha(canonical({'observations':observations,'parent':parent['gguf_sha256'] if parent else 'base'}))
        attempts=json.loads((A/'attempts.json').read_text()) if (A/'attempts.json').exists() else []
        now=datetime.now(timezone(timedelta(hours=8)));today=now.date().isoformat()
        if any(x.get('corpus_sha256')==corpus for x in attempts):return {**result,'phase':'waiting_for_changed_corpus'}
        if sum(x['local_day']==today for x in attempts)>=policy.max_updates_per_day:return {**result,'phase':'daily_update_budget_reached'}
        source=Path(config['source_code']);source_manifest=json.loads((source/'source-manifest.json').read_text())
        for p in source.glob('*.py'):assert sha(p.read_bytes())==source_manifest['files'][p.name],'Learner code snapshot changed during dispatch'
        run=R/'experiments'/('shuiqilin-auto-'+now.strftime('%Y%m%d-%H%M%S'));run.mkdir()
        for p in source.glob('*.py'):shutil.copy2(p,run/p.name)
        shutil.copy2(source/'source-manifest.json',run/'source-manifest.json')
        replay=Path(parent['peft']).parent/'data/train.jsonl' if parent else None
        manifest=build(run,config['source_paths'],bootstrap=False,consumed=consumed,eligible_ids=fresh,replay_path=replay)
        selected={json.loads(line)['origin'].get('observation_id') for line in (run/'data/train.jsonl').read_text().splitlines() if json.loads(line)['origin']['kind']=='verified_memory_policy'}
        assert len(selected)>=policy.min_support,'Training batch did not cover enough distinct certified episodes'
        save(run/'run-config.json',{'mode':'automatic','parent_adapter':parent,'source_paths':config['source_paths'],'selected_episodes':sorted(selected),'corpus_sha256':corpus,'manifest':manifest,'epoch_live_db':config['epoch_live_db'],'epoch_min_seq':config['epoch_min_seq'],'runtime_fingerprint':config['runtime_fingerprint']})
        unit='lea-shuiqilin-auto-'+now.strftime('%Y%m%d-%H%M%S')
        attempts.append({'local_day':today,'unit':unit,'work':str(run),'corpus_sha256':corpus});save(A/'attempts.json',attempts);save(A/'cursor.json',{'handoffs':ready['handoffs']})
        # Release the lock before a separate memory-bounded unit takes it.
        fcntl.flock(lock,fcntl.LOCK_UN)
        subprocess.run(['systemd-run','--user','--unit='+unit,'--property=WorkingDirectory='+str(run),'--property=Nice=19','--property=CPUQuota=500%',
            '--property=MemoryMax=28G','--property=MemorySwapMax=0','--property=RuntimeMaxSec=7200','--property=NoNewPrivileges=true',
            '--setenv=PYTHONPATH='+config['app_pythonpath'],str(R/'venv/bin/python'),'-u',str(run/'auto_memory_pipeline.py')],check=True,capture_output=True,text=True)
        return {**result,'phase':'automatic_probe_then_training_dispatched','work':str(run),'unit':unit}
    finally:lock.close()

def light_cycle(config,policy,lock):
    """Small verified-receipt batch; no paired probes or numerical score gates."""
    decision=Path(config['light_mode_decision'])
    assert sha(decision.read_bytes())==config['light_mode_decision_sha256'],'Light mode authorization changed'
    authorization=json.loads(decision.read_text())
    assert authorization['source']=='explicit_user_request' and authorization['allow_load_checked_promotion'] is True
    if config.get('data_scope')=='all_context_windows':return window_cycle(config,policy,lock)
    ready=readiness([Path(config['live_data'])/'memory.sqlite3',R/'data/memory.sqlite3'],policy)
    result={'phase':'waiting_for_idle','training_mode':'light','promotion_policy':'loadability_only','quality_evaluation':False,
        'training':{**LIGHT_TRAINING,**config.get('light_training',{})},'readiness':ready,'min_new_handoffs':5,
        'last_completed_run':config.get('last_completed_run'),'training_report':config.get('last_training_report')}
    training_plan({'mode':'automatic_light','training':result['training']},1)
    if runtime_fingerprint(R/'experiments/shuiqilin-unified-20261030-r1')!=config['runtime_fingerprint']:return {**result,'phase':'waiting_for_runtime_reconfiguration'}
    if not ready['ready']:return result
    cursor=json.loads((A/'cursor.json').read_text())
    if ready['handoffs']-cursor['handoffs']<5:return {**result,'phase':'waiting_for_five_new_windows'}
    all_rows,_,_=mine_sources(config['source_paths']);observations=current_epoch(all_rows,config)
    consumed=json.loads((A/'consumed.json').read_text()) if (A/'consumed.json').exists() else []
    fresh=fresh_experiences(observations,consumed)
    if not fresh:return {**result,'phase':'waiting_for_new_verified_experience','new_experiences':0}
    parent=json.loads((A/'active.json').read_text()) if (A/'active.json').exists() else None
    validate_parent(parent)
    profile=request('http://127.0.0.1:18815','/api/model-profile',timeout=5)
    health=request('http://127.0.0.1:18815','/api/health',timeout=10)
    if profile.get('primary_adapter')!=(parent['version'] if parent else None) or not health['model']['ready']:return {**result,'phase':'waiting_for_model_identity'}
    corpus=sha(canonical({'fresh':fresh,'parent':parent['gguf_sha256'] if parent else 'base','mode':'light'}))
    attempts=json.loads((A/'attempts.json').read_text()) if (A/'attempts.json').exists() else []
    now=datetime.now(timezone(timedelta(hours=8)));today=now.date().isoformat()
    if any(x.get('corpus_sha256')==corpus for x in attempts):return {**result,'phase':'waiting_for_changed_corpus'}
    if sum(x['local_day']==today for x in attempts)>=policy.max_updates_per_day:return {**result,'phase':'daily_update_budget_reached'}
    source=Path(config['source_code']);source_manifest=json.loads((source/'source-manifest.json').read_text())
    for p in source.glob('*.py'):assert sha(p.read_bytes())==source_manifest['files'][p.name],'Learner source changed'
    run=R/'experiments'/('shuiqilin-light-'+now.strftime('%Y%m%d-%H%M%S'));run.mkdir()
    for p in source.glob('*.py'):shutil.copy2(p,run/p.name)
    shutil.copy2(source/'source-manifest.json',run/'source-manifest.json');shutil.copy2(decision,run/'mode-decision.json')
    training=result['training'];replay=Path(parent['peft']).parent/'data/train.jsonl' if parent else None
    manifest=build(run,config['source_paths'],bootstrap=False,consumed=consumed,eligible_ids=[o['id'] for o in fresh],replay_path=replay,
        max_new_rows=training['max_new_rows'],include_evaluation=False,complete_groups=True,replay_offset=int(sha(run.name)[:8],16))
    selected=sorted({json.loads(line)['origin']['observation_id'] for line in (run/'data/train.jsonl').read_text().splitlines() if json.loads(line)['origin']['kind']=='verified_memory_policy'})
    assert selected and manifest['evaluation_rows']==0
    plan=training_plan({'mode':'automatic_light','training':training},manifest['train_rows'])
    save(run/'run-config.json',{'mode':'automatic_light','training':training,'parent_adapter':parent,'selected_episodes':selected,
        'manifest':manifest,'source_paths':config['source_paths'],'runtime_fingerprint':config['runtime_fingerprint'],
        'mode_decision':str(run/'mode-decision.json'),'mode_decision_sha256':sha(decision.read_bytes()),'source_mode_decision_sha256':sha(decision.read_bytes())})
    unit='lea-shuiqilin-light-'+now.strftime('%Y%m%d-%H%M%S')
    attempts.append({'local_day':today,'unit':unit,'work':str(run),'corpus_sha256':corpus,'mode':'automatic_light'});save(A/'attempts.json',attempts)
    fcntl.flock(lock,fcntl.LOCK_UN)
    subprocess.run(['systemd-run','--user','--unit='+unit,'--property=WorkingDirectory='+str(run),'--property=Nice=19',
        '--property=CPUQuota=300%','--property=MemoryMax=28G','--property=RuntimeMaxSec=3600','--property=NoNewPrivileges=true',
        '--setenv=PYTHONPATH='+config['app_pythonpath'],str(R/'venv/bin/python'),'-u',str(run/'light_lora_pipeline.py')],check=True,capture_output=True,text=True)
    save(A/'cursor.json',{'handoffs':ready['handoffs']})
    return {**result,'phase':'light_training_dispatched','work':str(run),'unit':unit,'planned_steps':plan['max_steps']}


def window_cycle(config,policy,lock):
    """All archived contents, ten handoffs per batch, no daily attempt cap."""
    ready=readiness([Path(config['live_data'])/'memory.sqlite3',R/'data/memory.sqlite3'],policy)
    cursor=json.loads((A/'window-cursor.json').read_text())
    count=config.get('min_new_handoffs',10);assert count==10
    result={'phase':'waiting_for_idle','training_mode':'light','data_scope':'all_context_windows',
        'training_content':config.get('training_content','raw_archive_and_teacher'),
        'promotion_policy':'loadability_only','quality_evaluation':False,'training':config['light_training'],
        'min_new_handoffs':count,'max_updates_per_day':None,'readiness':ready,
        'new_handoffs':max(0,ready['handoffs']-cursor['handoffs']),
        'memory_teacher':config.get('memory_teacher',{'enabled':False}),
        'last_completed_run':config.get('last_completed_run'),'training_report':config.get('last_training_report')}
    from training_priority import status as training_priority
    priority=training_priority(A,config)
    result['background_inference_paused']=priority['hold_background_inference']
    pending=A/'pending-window-batch.json'
    if pending.exists():
        task=json.loads(pending.read_text());w=Path(task['work']);out=w/'pipeline-report.json'
        report=json.loads(out.read_text()) if out.exists() else {}
        # Acknowledgement is idempotent, including a crash just after activation.
        skipped=report.get('skipped') and report.get('training_content')=='teacher_only' and report.get('skip_reason')=='no_teacher_lessons'
        if report.get('completed') and (report.get('activated') or skipped):
            save(A/'window-cursor.json',task['next_cursor']);pending.unlink()
            return {**result,'phase':'window_batch_skipped_no_lessons' if skipped else 'window_batch_acknowledged','work':str(w)}
        failed=priority['phase'] in ('training_failed_background_resumed','training_unit_stopped_background_resumed')
        if failed and ready['ready'] and not report.get('completed'):
            retries=task.get('automatic_retries',0)
            if retries<3 and time.time()-max(report.get('updated_at',0),task.get('last_retry_at',0))>=60:
                from training_priority import unit_active
                if not unit_active(task['unit']):
                    task.update(automatic_retries=retries+1,last_retry_at=time.time());save(pending,task)
                    code=['systemd-run','--user','--unit='+task['unit'],'--property=WorkingDirectory='+str(w),
                        '--property=Nice=19','--property=CPUQuota=300%','--property=MemoryMax=28G',
                        '--property=RuntimeMaxSec=infinity','--property=NoNewPrivileges=true',
                        '--setenv=PYTHONPATH='+config['app_pythonpath'],str(R/'venv/bin/python'),'-u',str(w/'light_lora_pipeline.py')]
                    existing=subprocess.run(['systemctl','--user','show',task['unit'],'-p','LoadState','--value'],capture_output=True,text=True).stdout.strip()
                    fcntl.flock(lock,fcntl.LOCK_UN)
                    if existing=='loaded':
                        subprocess.run(['systemctl','--user','reset-failed',task['unit']],capture_output=True)
                        subprocess.run(['systemctl','--user','start',task['unit']],check=True,capture_output=True)
                    else:subprocess.run(code,check=True,capture_output=True)
                    return {**result,'phase':'window_batch_resumed','work':str(w),'automatic_retries':retries+1}

        return {**result,'phase':'window_batch_requires_attention' if failed or report.get('phase')=='failed' else 'window_batch_in_progress',
                'work':str(w),'unit':task['unit'],'error':report.get('error')}
    if runtime_fingerprint(R/'experiments/shuiqilin-unified-20261030-r1')!=config['runtime_fingerprint']:
        return {**result,'phase':'waiting_for_runtime_reconfiguration'}
    if config.get('memory_capacity',{}).get('enabled'):
        from memory_capacity import capacity
        storage=capacity(config['memory_capacity'])
        if not storage['write_allowed'] or not storage['optional_write_allowed']:
            return {**result,'phase':'waiting_for_memory_capacity','capacity':storage}
    if not ready['ready']:
        if ready['foreground_jobs'] or ready['idle_seconds']<policy.min_idle_seconds:phase='waiting_for_foreground_idle'
        elif ready['available_gib']<policy.min_available_gib:phase='waiting_for_memory'
        else:phase='waiting_for_cpu_load'
        return {**result,'phase':phase}
    source_batch=snapshot(Path(config['live_data'])/'memory.sqlite3',cursor,count)
    if source_batch is None:return {**result,'phase':'waiting_for_ten_new_windows'}
    parent=json.loads((A/'active.json').read_text()) if (A/'active.json').exists() else None
    validate_parent(parent)
    profile=request('http://127.0.0.1:18815','/api/model-profile',timeout=5)
    health=request('http://127.0.0.1:18815','/api/health',timeout=10)
    if profile.get('primary_adapter')!=(parent['version'] if parent else None) or not health['model']['ready']:
        return {**result,'phase':'waiting_for_model_identity'}
    source=Path(config['source_code']);source_manifest=json.loads((source/'source-manifest.json').read_text())
    for p in source.glob('*.py'):assert sha(p.read_bytes())==source_manifest['files'][p.name],'Learner source changed'
    now=datetime.now(timezone(timedelta(hours=8)))
    run=R/'experiments'/('shuiqilin-windows-'+now.strftime('%Y%m%d-%H%M%S'));run.mkdir()
    for p in source.glob('*.py'):shutil.copy2(p,run/p.name)
    shutil.copy2(source/'source-manifest.json',run/'source-manifest.json')
    decision=Path(config['light_mode_decision']);shutil.copy2(decision,run/'mode-decision.json')
    from memory_capacity import prepare_training_cache
    cache=prepare_training_cache(run,config.get('memory_capacity'))
    if cache:
        save(cache/'window-source.json',source_batch)
        (run/'window-source.json').symlink_to(cache/'window-source.json')
    else:save(run/'window-source.json',source_batch)
    save(run/'run-config.json',{'mode':'automatic_light','data_scope':'all_context_windows','training':config['light_training'],
        'training_content':config.get('training_content','raw_archive_and_teacher'),'memory_capacity':config.get('memory_capacity'),
        'parent_adapter':parent,'selected_episodes':[],'runtime_fingerprint':config['runtime_fingerprint'],
        'mode_decision':str(run/'mode-decision.json'),'mode_decision_sha256':sha(decision.read_bytes()),
        'source_mode_decision_sha256':sha(decision.read_bytes()),'window_source_sha256':sha((run/'window-source.json').read_bytes()),
        'next_window_cursor':source_batch['next_cursor'],'memory_teacher':config.get('memory_teacher',{'enabled':False}),
        'automatic_source_review':config.get('automatic_source_review',{}),'kv_reset_after_lora':config.get('kv_reset_after_lora',{})})
    unit='lea-shuiqilin-windows-'+now.strftime('%Y%m%d-%H%M%S')
    task={'unit':unit,'work':str(run),'next_cursor':source_batch['next_cursor'],'created_at':time.time()}
    save(pending,task)
    attempts=json.loads((A/'attempts.json').read_text()) if (A/'attempts.json').exists() else []
    attempts.append({**task,'local_day':now.date().isoformat(),'mode':'all_context_windows',
        'corpus_sha256':source_batch['events_sha256']});save(A/'attempts.json',attempts)
    # Keep a durable pending record before dispatch. Never advance the cursor on failure.
    fcntl.flock(lock,fcntl.LOCK_UN)
    subprocess.run(['systemd-run','--user','--unit='+unit,'--property=WorkingDirectory='+str(run),'--property=Nice=19',
        '--property=CPUQuota=300%','--property=MemoryMax=28G','--property=RuntimeMaxSec=infinity','--property=NoNewPrivileges=true',
        '--setenv=PYTHONPATH='+config['app_pythonpath'],str(R/'venv/bin/python'),'-u',str(run/'light_lora_pipeline.py')],
        check=True,capture_output=True,text=True)
    return {**result,'phase':'window_training_dispatched',**task,'source_events':len(source_batch['events'])}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');args=p.parse_args()
    while True:
        try:
            state=cycle();save(A/'worker-status.json',{**state,'checked_at':time.time()})
            if state['phase']!='learning_in_progress':save(A/'state.json',{k:v for k,v in state.items() if k!='gate'})
        except BaseException as error:save(A/'worker-error.json',{'error':repr(error),'time':time.time()})
        if args.once:break
        time.sleep(20)
