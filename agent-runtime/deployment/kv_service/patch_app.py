"""Apply only the native-session feature to installed and local source variants."""
from pathlib import Path
import shutil

here=Path(__file__).resolve().parent
root=here.parents[1]
stage=here/'app/lea';stage.mkdir(parents=True,exist_ok=True)

def replace_once(text,old,new):
    assert text.count(old)==1,old
    return text.replace(old,new)

def patch(name,text):
    if name=='config.py':
        text+='\n    native_kv_url: str = os.environ.get("LEA_NATIVE_KV_URL", "")\n'
    elif name=='model.py':
        text=replace_once(text,'from .store import dump','from .store import dump\nfrom .native_kv import NativeSessionClient')
        text=replace_once(text,'        self.queue = InferenceQueue()',
            '        self.queue = InferenceQueue()\n        self.native_kv = NativeSessionClient(settings,store) if settings.native_kv_url else None')
        text=replace_once(text,'        await self.http.aclose()',
            '        await self.http.aclose()\n        if self.native_kv:await self.native_kv.close()')
        text=replace_once(text,'            return value',
            '''            if self.native_kv:
                value['native_kv'] = await self.native_kv.health()
                value['active_primary_backend'] = 'native_kvzip'
                value['ready'] = value['ready'] and value['native_kv']['ready']
                if 'prefix_cache' in value:
                    value['prefix_cache']['scope'] = 'auxiliary_and_legacy_only'
                    value['prefix_cache']['primary_slot'] = None
            return value''')
        text=replace_once(text,'        budget = budget or self.settings.working_tokens',
            '''        if role == 'primary' and self.native_kv:
            return await self.native_kv.call(task_id,messages,schema,output_schema,limit,self.queue)
        budget = budget or self.settings.working_tokens''')
    elif name=='runtime.py':
        text=replace_once(text,'from .store import Conflict, Store',
            'from .store import Conflict, Store\nfrom .native_kv import native_handoff, reconcile_windows')
        text=replace_once(text,'    async def handoff(self, task_id, reason):\n',
            '''    async def handoff(self, task_id, reason):
        if getattr(self.model,'native_kv',None):
            return await native_handoff(self,task_id,reason)
''')
        if '            task, records = self.context(task_id, after=after)' in text:
            text=replace_once(text,'            task, records = self.context(task_id, after=after)',
                '            task, records, native_cutoff = self.context(task_id, after=after, with_cutoff=True)')
        else:
            text=replace_once(text,'            task, records = self.context(task_id)',
                '            task, records, native_cutoff = self.context(task_id, with_cutoff=True)')
        anchor='            prompt = messages('
        assert text.count(anchor)==1
        text=text.replace(anchor,'''            if getattr(self.model,'native_kv',None):
                payload['_native_through_seq']=native_cutoff
                payload['current_request_operations']=[
                    {'id':op['id'],'tool':op['tool'],'status':op['status'],'path':op['arguments'].get('path'),
                     'evidence_ref':(op.get('result') or {}).get('evidence_ref')}
                    for op in self.store.operations(task_id) if op['user_event']==body.get('user_event')]
'''+anchor)
        text=replace_once(text,'            if await self.model.tokens(prompt) > self.settings.working_tokens:',
            "            if not getattr(self.model,'native_kv',None) and await self.model.tokens(prompt) > self.settings.working_tokens:")
        marker='            self.store.append(task_id, "primary", "primary_decision", decision.model_dump())'
        text=replace_once(text,marker,'''            if getattr(self.model,'native_kv',None):
                metadata=self.model.native_kv.take_result(task_id)
                try:
                    task=reconcile_windows(self.store,task_id,task['window_id'],task['constraint_version'],metadata)
                except Conflict:
                    last_result={'warning':'Effective constraints or task state changed; replan with the latest state'}
                    continue
'''+marker)
    return text

for name in ['config.py','model.py','runtime.py']:
    original=(here/'baseline/lea'/name).read_text(encoding='utf-8')
    (stage/name).write_text(patch(name,original),encoding='utf-8')
    local=root/'lea'/name
    content=local.read_text(encoding='utf-8')
    if 'native_kv_url' not in content and 'from .native_kv import' not in content:
        local.write_text(patch(name,content),encoding='utf-8')
shutil.copy2(root/'lea/native_kv.py',stage/'native_kv.py')
print(stage)
