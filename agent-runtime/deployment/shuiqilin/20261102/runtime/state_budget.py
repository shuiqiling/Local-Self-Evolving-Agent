"""Fit the final state messages, preserving source data and execution evidence."""
import copy
import json
from types import SimpleNamespace

VERSION='final-state-budget-v1'


async def fit(runtime,task,prompt,cutoff,render):
    """Existing source fitting sees the exact prompt that will be sent."""
    async def count(raw):return await runtime.model.tokens(render(raw))
    if await count(prompt)<=runtime.settings.working_tokens:return render(prompt)
    candidate=copy.deepcopy(prompt);payload=json.loads(candidate[-1]['content'])
    # Latest receipt stays intact. Older full receipts remain archived by ref;
    # their selection/status/source pointer are retained in the active view.
    recent=payload.get('initiative_context',{}).get('recent',[])
    for entry in recent[1:]:
        if entry.get('evidence_ref') and entry.get('receipt_excerpt') is not None:
            entry.pop('receipt_excerpt')
            entry['receipt_details_omitted']=True
            entry['receipt_note']='旧回执详情按evidence_ref读取；executed不代表内容正确或测试通过。'
    candidate[-1]['content']=json.dumps(payload,ensure_ascii=False,separators=(',',':'))
    from continuity_state import fit_prompt
    # A proxy avoids mutating the shared model/token counter while foreground
    # requests may be running in another task.
    proxy=SimpleNamespace(store=runtime.store,settings=runtime.settings,model=SimpleNamespace(tokens=count))
    fitted=await fit_prompt(proxy,task,candidate,cutoff,None)
    return render(fitted)


def useful_records(records):
    return any(r.get('kind') not in ('handoff_requested','handoff_completed') for r in records)
