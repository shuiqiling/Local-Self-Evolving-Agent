"""Bounded small updates. No model-quality scores or held-out questions."""
import math

LIGHT_TRAINING={'max_steps':8,'max_new_rows':4,'replay_fraction':0.5,'epochs':1,'learning_rate':5e-6,
    'gradient_clip':0.5,'max_sequence_tokens':512,'threads':4,'checkpoint_every':4,'batch_size':1}

def training_plan(config,rows):
    if config.get('mode')!='automatic_light':
        return dict(max_steps=rows,epochs=1,learning_rate=3e-5,gradient_clip=1.,max_sequence_tokens=512,
            threads=6,checkpoint_every=8,batch_size=1,warmup_steps=3,light=False)
    p={**LIGHT_TRAINING,**config.get('training',{})}
    all_windows=config.get('data_scope')=='all_context_windows'
    if p['epochs']!=1 or p['batch_size']!=1 or not 1<=p['max_steps']<=8:raise ValueError('Light training requires one pass and at most8steps')
    if not 0<rows or (not all_windows and rows>p['max_steps']):raise ValueError('Dataset exceeds light step budget')
    if not math.isfinite(p['learning_rate']) or not 0<p['learning_rate']<=5e-6:raise ValueError('Light learning rate exceeds5e-6')
    if not 0<p['gradient_clip']<=.5 or not 1<=p['threads']<=4:raise ValueError('Light gradient/CPU budget exceeded')
    if not 1<=p['max_sequence_tokens']<=512 or (not all_windows and not 1<=p['max_new_rows']<=4) or p['replay_fraction']!=.5:raise ValueError('Light data budget exceeded')
    if not 1<=p['checkpoint_every']<=4:raise ValueError('Light checkpoint interval exceeded')
    steps=min(rows,p['max_steps'])
    p.update(max_steps=steps,warmup_steps=min(2,steps),light=True)
    if all_windows:p.update(max_new_rows=None,micro_batches=rows,data_scope='all_context_windows',gradient_accumulation=True)
    return p

def fresh_experiences(observations,consumed):
    """Executed tool tests and immutable receipts stay required; no score gate."""
    used=set(consumed);seen=set();selected=[]
    for row in observations:
        key=row['fingerprint']
        if row['id'] in used or key in seen:continue
        if (row.get('independently_verified') and row.get('policy_followed') and row.get('creation_job_completed')
            and not row.get('critical_violation') and not row.get('failed_outcome')):
            selected.append(row);seen.add(key)
    return selected

def load_checked_manifest(candidate,report_path,report_sha,decision_path,decision_sha):
    return {**candidate,'purpose':'load_checked','accepted':False,'quality_evaluated':False,
        'loadability_passed':True,'load_check_report':str(report_path),'load_check_sha256':report_sha,
        'mode_decision':str(decision_path),'mode_decision_sha256':decision_sha,'id':0,'loaded_ids':[0]}
