"""Light training → native load check → activate, with technical rollback.

No paired benchmark, memory-value probe, answer judgement or quality threshold.
"""
from pathlib import Path
import fcntl
import json
import os
import shlex
import sqlite3
import subprocess
import time
from auto_memory_data import save,sha
from auto_memory_eval import request
from auto_memory_policy import validate_parent,validate_runtime_adapter,runtime_fingerprint,export_base_config
from light_lora_policy import training_plan,load_checked_manifest

R=Path('/home/ubuntu/AI-Native/embodied-agent');A=R/'auto-lora-shuiqilin'
U=R/'experiments/shuiqilin-unified-20261030-r1';DATA=R/'experiments/qwen30b-memory-20261030-r1/data'

def system(*args,check=True):return subprocess.run(['systemctl','--user',*args],check=check,capture_output=True,text=True,timeout=180)
def idle():
    for p in (DATA/'memory.sqlite3',R/'data/memory.sqlite3'):
        with sqlite3.connect('file:'+str(p)+'?mode=ro',uri=True) as db:
            if db.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]:return False
    return True
def wait_idle():
    while not idle():time.sleep(10)
def wait_backend():
    deadline=time.monotonic()+900
    while time.monotonic()<deadline:
        try:
            if request('http://127.0.0.1:18813','/health',timeout=3).get('status')=='ok':return
        except OSError:pass
        time.sleep(3)
    raise TimeoutError('Native backend did not start within900s')
def wait_agent(version):
    deadline=time.monotonic()+100
    while time.monotonic()<deadline:
        try:
            p=request('http://127.0.0.1:18815','/api/model-profile',timeout=5)
            h=request('http://127.0.0.1:18815','/api/health',timeout=8)
            if p['primary_adapter']==version and h['model']['ready']:return p
        except OSError:pass
        time.sleep(2)
    raise TimeoutError('Agent failed to start with selected adapter')

def promote(work,candidate,parent,config,train,finite):
    if config.get('memory_teacher',{}).get('projection_version')=='seven-field-experiences-v1':
        import sys
        from dual_adapter_promotion import promote as dual_promote
        return dual_promote(work,candidate,parent,config,train,finite,api=sys.modules[__name__])
    w=Path(work);wait_idle()
    assert runtime_fingerprint(U)==config['runtime_fingerprint'],'Runtime changed during training'
    current=json.loads((A/'active.json').read_text()) if (A/'active.json').exists() else None
    assert (current['gguf_sha256'] if current else None)==(parent['gguf_sha256'] if parent else None),'Serving parent changed'
    live_config=json.loads((A/'config.json').read_text())
    assert live_config.get('training_mode')=='light' and sha(Path(live_config['light_mode_decision']).read_bytes())==config['source_mode_decision_sha256'],'Mode changed during training'
    drop=Path.home()/'.config/systemd/user/lea-qwen30b-trial.service.d/80-memory-lora.conf'
    old_drop=drop.read_text() if drop.exists() else None;old_active=(A/'active.json').read_bytes() if (A/'active.json').exists() else None
    save(w/'rollback-state.json',{'backend_dropin':old_drop,'active_manifest':json.loads(old_active) if old_active else None})
    pid=system('show','lea-qwen30b-trial.service','-p','MainPID','--value').stdout.strip()
    argv=[x.decode() for x in Path('/proc',pid,'cmdline').read_bytes().split(b'\0') if x]
    clean=[];i=0
    while i<len(argv):
        if argv[i] in ('--lora','--lora-scaled'):i+=2
        elif argv[i]=='--lora-init-without-apply':i+=1
        else:clean.append(argv[i]);i+=1
    clean+=['--lora-scaled',candidate['gguf']+':0']
    try:
        system('stop','lea-shuiqilin.service')
        drop.parent.mkdir(exist_ok=True);drop.write_text('[Service]\nExecStart=\nExecStart='+shlex.join(clean).replace('%','%%')+'\n')
        system('daemon-reload');system('restart','lea-qwen30b-trial.service');wait_backend()
        loaded=request('http://127.0.0.1:18813','/lora-adapters',timeout=10)
        assert len(loaded)==1 and loaded[0]['path']==candidate['gguf'],'Wrong adapter loaded'
        # One-token forward checks runtime compatibility, never answer correctness.
        result=request('http://127.0.0.1:18813','/v1/chat/completions',{'model':'Qwen3-30B-A3B-Q4_K_M',
            'messages':[{'role':'user','content':'你好'}],'max_tokens':1,'temperature':0,'cache_prompt':False,
            'chat_template_kwargs':{'enable_thinking':False},'lora':[{'id':0,'scale':1.}]},timeout=120)
        assert result.get('choices'),'Generation API failed'
        check={'completed':True,'loadable':True,'native_loaded':True,'generation_api_ok':True,'finite_weights':finite['finite_weights'],
            'budget_checked':True,'step_count':train['step_count'],'max_learning_rate':max(x['lr'] for x in train['steps']),
            'epochs':1,'gguf_sha256':candidate['gguf_sha256'],'peft_sha256':candidate['peft_sha256'],'quality_evaluated':False,
            'scope':'File checks, native load, one-token API forward and startup only; no expected answer or score'}
        save(w/'load-check-report.json',check)
        manifest=load_checked_manifest(candidate,w/'load-check-report.json',sha((w/'load-check-report.json').read_bytes()),
            config['mode_decision'],config['mode_decision_sha256'])
        validate_runtime_adapter(manifest);save(A/'active.json',manifest)
        system('start','lea-shuiqilin.service');profile=wait_agent(candidate['version'])
        save(w/'activation-report.json',{'completed':True,'activated':True,'profile':profile,'quality_evaluation':False})
        return manifest
    except BaseException:
        system('stop','lea-shuiqilin.service',check=False)
        if old_drop is None:drop.unlink(missing_ok=True)
        else:drop.write_text(old_drop)
        if old_active is None:(A/'active.json').unlink(missing_ok=True)
        else:(A/'active.json').write_bytes(old_active)
        system('daemon-reload');system('restart','lea-qwen30b-trial.service');wait_backend()
        system('start','lea-shuiqilin.service');wait_agent(current['version'] if current else None)
        save(w/'rollback-completed.json',{'completed':True,'basis':'Technical load/startup error, not a model-quality score'})
        raise

def run(work):
    w=Path(work);out=w/'pipeline-report.json';report=json.loads(out.read_text()) if out.exists() else {'completed':False,'activated':False,'work':str(w),'quality_evaluation':False}
    assert not report['completed'],'Already completed; do not duplicate'
    def progress(phase,**values):
        report.update(phase=phase,updated_at=time.time(),**values);save(out,report)
        save(A/'state.json',{'phase':phase,'work':str(w),'training_mode':'light','quality_evaluation':False,'promotion_policy':'loadability_only','training_report':str(w/'training-report.json')})
    with (A/'learner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            config=json.loads((w/'run-config.json').read_text());assert config['mode']=='automatic_light'
            window_mode=config.get('data_scope')=='all_context_windows'
            if window_mode and not (w/'data/manifest.json').exists():
                progress('preparing_all_window_tokens')
                with (w/'data-build.log').open('w') as log:subprocess.run([str(R/'training/concise-v2/venv/bin/python'),str(w/'window_memory_data.py'),str(w)],
                    env={**os.environ,'PYTHONPATH':str(R/'experiments/qwen30b-nf4-20261030-r1/deps'),'HF_HUB_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'},
                    stdout=log,stderr=subprocess.STDOUT,check=True,timeout=None if config.get('memory_teacher',{}).get('enabled') else 600)
            data=json.loads((w/'data/manifest.json').read_text());assert data['evaluation_rows']==0 and not (w/'data/evaluation.json').exists()
            if window_mode:
                assert data['handoffs']==10
                if config.get('training_content')=='teacher_only':
                    from teacher_training import validate_dataset
                    assert not data.get('preview_only'),'Preview data cannot consume production windows'
                    assert data['window_source_sha256']==config['window_source_sha256']==sha((w/'window-source.json').read_bytes()),'Teacher dataset source differs from the frozen windows'
                    rows=[json.loads(line) for line in (w/'data/train.jsonl').read_text().splitlines()]
                    validate_dataset(data,rows)
                    if not rows:
                        save(A/'window-cursor.json',config['next_window_cursor'])
                        progress('skipped_no_teacher_lessons',completed=True,activated=False,skipped=True,
                            training_content='teacher_only',skip_reason='no_teacher_lessons',gradient_training_started=False)
                        return report
                else:assert data['full_source_coverage'] and data['truncated_source_tokens']==0
            plan=training_plan(config,data['train_rows']);parent=config.get('parent_adapter');validate_parent(parent)
            assert runtime_fingerprint(U)==config['runtime_fingerprint'],'Runtime changed before training'
            assert sha(Path(config['mode_decision']).read_bytes())==config['mode_decision_sha256']
            env={**os.environ,'PYTHONPATH':str(R/'experiments/qwen30b-nf4-20261030-r1/deps'),
                'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false','CUDA_VISIBLE_DEVICES':'',
                'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':str(plan['threads'])}
            if not (w/'preflight.log').exists():
                with (w/'preflight.log').open('w') as log:subprocess.run([str(R/'training/concise-v2/venv/bin/python'),str(w/'auto_memory_train.py'),'--preflight'],env=env,cwd=w,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
            while not (w/'training-report.json').exists() or not json.loads((w/'training-report.json').read_text()).get('completed'):
                wait_idle();progress('light_training',training_plan=plan)
                command=[str(R/'training/concise-v2/venv/bin/python'),'-u',str(w/'auto_memory_train.py')]
                if (w/'latest-checkpoint.json').exists():command+=['--resume']
                with (w/'training.log').open('a') as log:code=subprocess.run(command,env=env,cwd=w,stdout=log,stderr=subprocess.STDOUT,timeout=None if window_mode else 1800).returncode
                if code==75:progress('yielded_to_foreground');time.sleep(10);continue
                assert code==0,'Training failed; checkpoint retained'
            train=json.loads((w/'training-report.json').read_text());assert 0<train['step_count']<=plan['max_steps']<=8
            assert max(x['lr'] for x in train['steps'])<=5e-6
            wait_idle();progress('exporting_adapter')
            base=export_base_config(R/'models/Qwen3-30B-A3B-bnb-4bit/config.json',w/'export-base-config');save(w/'export-report.json',base)
            gguf=w/'adapter-f16.gguf'
            if not gguf.exists():
                with (w/'conversion.log').open('w') as log:subprocess.run(['/home/ubuntu/ai/hf-venv/bin/python','/home/ubuntu/ai/tools/llama.cpp/convert_lora_to_gguf.py',str(w/'adapter'),
                    '--base',base['export_directory'],'--outtype','f16','--outfile',str(gguf)],check=True,stdout=log,stderr=subprocess.STDOUT,timeout=300)
            with (w/'finite-check.log').open('w') as log:subprocess.run(['/home/ubuntu/ai/hf-venv/bin/python',str(w/'light_lora_check.py'),str(w)],
                env={**os.environ,'PYTHONPATH':'/home/ubuntu/ai/tools/llama.cpp/gguf-py'},check=True,stdout=log,stderr=subprocess.STDOUT,timeout=120)
            finite=json.loads((w/'finite-check.json').read_text());assert finite['finite_weights']
            candidate={'model':'Qwen3-30B-A3B-Q4_K_M','version':'light-'+w.name,'gguf':str(gguf),'gguf_sha256':sha(gguf.read_bytes()),
                'peft':str(w/'adapter'),'peft_sha256':sha((w/'adapter/adapter_model.safetensors').read_bytes()),
                'peft_config_sha256':sha((w/'adapter/adapter_config.json').read_bytes()),'base_merged':False}
            save(w/'candidate.json',candidate);progress('native_load_and_startup_check')
            manifest=promote(w,candidate,parent,config,train,finite)
            consumed=json.loads((A/'consumed.json').read_text()) if (A/'consumed.json').exists() else []
            save(A/'consumed.json',sorted(set(consumed+config['selected_episodes'])))
            settings=json.loads((A/'config.json').read_text());settings.update(last_completed_run=str(w),last_training_report=str(w/'training-report.json'));save(A/'config.json',settings)
            if window_mode:save(A/'window-cursor.json',config['next_window_cursor'])
            progress('completed',completed=True,activated=True,candidate=manifest,summary='light_update_loaded_without_quality_benchmark')
        except BaseException as error:progress('failed',error=repr(error));raise
    return report

if __name__=='__main__':run(Path(__file__).resolve().parent)
