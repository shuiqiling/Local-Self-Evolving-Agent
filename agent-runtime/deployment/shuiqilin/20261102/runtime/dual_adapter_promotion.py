"""Transactional main-adapter replacement preserving the independent off helper."""
from pathlib import Path
import json, shlex, time
from auto_memory_data import sha,save

def command(argv, main, helper):
    clean=[];i=0
    while i<len(argv):
        if argv[i] in ('--lora','--lora-scaled'):i+=2
        elif argv[i]=='--lora-init-without-apply':i+=1
        else:clean.append(argv[i]);i+=1
    assert main!=helper
    return clean+['--lora-scaled',str(main)+':0','--lora-scaled',str(helper)+':0']

def selection(main=True):
    return [{'id':0,'scale':1.0 if main else 0.0},{'id':1,'scale':0.0 if main else 1.0}]

def promote(work,candidate,parent,config,train,finite,*,api):
    w=Path(work);A=api.A;api.wait_idle()
    assert api.runtime_fingerprint(api.U)==config['runtime_fingerprint']
    current=json.loads((A/'active.json').read_text())
    assert current['gguf_sha256']==parent['gguf_sha256'],'Serving parent changed'
    live_config=json.loads((A/'config.json').read_text())
    assert live_config['training_mode']=='light'
    assert sha(Path(live_config['light_mode_decision']).read_bytes())==config['source_mode_decision_sha256']
    helper_path=Path(live_config['memory_teacher']['runtime_manifest'])
    helper=json.loads(helper_path.read_text())
    assert helper['enable_thinking'] is False and helper['adapter_id']==1 and helper['chat_adapter_id']==0
    assert sha(Path(helper['gguf']).read_bytes())==helper['gguf_sha256']==config['memory_teacher']['helper_adapter_sha256']
    assert helper['chat_gguf']==current['gguf']
    drop=Path.home()/'.config/systemd/user/lea-qwen30b-trial.service.d/99-auto-training-dual.conf'
    paths=[drop,A/'active.json',helper_path,A/'config.json',A/'kv-reset-status.json']
    previous={str(p):(p.read_bytes() if p.exists() else None) for p in paths}
    save(w/'promotion-backups.json',{str(p): {'existed':previous[str(p)] is not None,
      'sha256':sha(previous[str(p)]) if previous[str(p)] is not None else None} for p in paths})
    for i,p in enumerate(paths):
        if previous[str(p)] is not None:(w/f'promotion-backup-{i}.bin').write_bytes(previous[str(p)])
    pid=api.system('show','lea-qwen30b-trial.service','-p','MainPID','--value').stdout.strip()
    argv=[s.decode() for s in Path('/proc',pid,'cmdline').read_bytes().split(b'\0') if s]
    args=command(argv,candidate['gguf'],helper['gguf'])
    tick=time.monotonic()
    try:
        api.system('stop','lea-shuiqilin.service')
        drop.parent.mkdir(parents=True,exist_ok=True)
        drop.write_text('[Service]\nExecStart=\nExecStart='+shlex.join(args).replace('%','%%')+'\n')
        api.system('daemon-reload');api.system('restart','lea-qwen30b-trial.service');api.wait_backend()
        loaded=api.request('http://127.0.0.1:18813','/lora-adapters',timeout=15)
        assert len(loaded)==2 and {x['id']:x['path'] for x in loaded}=={0:candidate['gguf'],1:helper['gguf']}
        assert all(x['scale']==0 for x in loaded)
        # Training finished, fresh native load succeeded, frontend still stopped.
        from kv_after_lora import check_namespaces_then_clear
        new_pid=api.system('show','lea-qwen30b-trial.service','-p','MainPID','--value').stdout.strip()
        kv_reset=check_namespaces_then_clear(api.request,candidate['model'],candidate['version'],
            fresh_backend=bool(new_pid and new_pid!='0' and new_pid!=pid),output=w/'kv-reset-report.json')
        check={'completed':True,'loadable':True,'native_loaded':True,'generation_api_ok':True,
          'finite_weights':finite['finite_weights'],'budget_checked':True,'step_count':train['step_count'],
          'max_learning_rate':max(x['lr'] for x in train['steps']),'epochs':1,
          'gguf_sha256':candidate['gguf_sha256'],'peft_sha256':candidate['peft_sha256'],
          'quality_evaluated':False,'independent_helper_preserved':True,'helper_gguf_sha256':helper['gguf_sha256'],
          'main_namespace':selection(True),'helper_namespace':selection(False),
          'scope':'Source-verified data, finite weights and two isolated one-token forwards; not an answer quality score'}
        save(w/'load-check-report.json',check)
        manifest=api.load_checked_manifest(candidate,w/'load-check-report.json',sha((w/'load-check-report.json').read_bytes()),
            config['mode_decision'],config['mode_decision_sha256'])
        manifest['loaded_ids']=[0,1]
        manifest.update(kv_cleared_before_window_resume=True,kv_reset_report=str(w/'kv-reset-report.json'),
            kv_reset_report_sha256=sha((w/'kv-reset-report.json').read_bytes()))
        api.validate_runtime_adapter(manifest);save(A/'active.json',manifest)
        save(A/'kv-reset-status.json',kv_reset)
        helper['chat_gguf']=candidate['gguf'];helper['chat_gguf_sha256']=candidate['gguf_sha256']
        save(helper_path,helper)
        live_config['memory_teacher']['runtime_manifest_sha256']=sha(helper_path.read_bytes())
        save(A/'config.json',live_config)
        api.system('start','lea-shuiqilin.service');profile=api.wait_agent(candidate['version'])
        status=api.request('http://127.0.0.1:18815','/api/helper-profile',timeout=20)
        assert status['adapter_sha256']==helper['gguf_sha256'] and status['thinking_enabled'] is False
        save(w/'activation-report.json',{'completed':True,'activated':True,'profile':profile,
           'helper_profile':status,'quality_evaluation':False,'seconds':round(time.monotonic()-tick,3)})
        return manifest
    except BaseException:
        api.system('stop','lea-shuiqilin.service',check=False)
        for p in paths:
            value=previous[str(p)]
            if value is None:p.unlink(missing_ok=True)
            else:p.write_bytes(value)
        api.system('daemon-reload');api.system('restart','lea-qwen30b-trial.service');api.wait_backend()
        api.system('start','lea-shuiqilin.service');api.wait_agent(current['version'])
        save(w/'rollback-completed.json',{'completed':True,'main_and_helper_restored':True})
        raise
