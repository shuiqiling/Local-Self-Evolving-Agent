"""Bounded prompt views of archived execution traces; never rewrite evidence."""
import json

VERSION='archived-trace-view-v1'

def compact_records(store, task_id, records, cutoff, last_result):
    """Only compact already-archived model intentions and old tool receipts.

    Current request, constraints, latest tool result and host observations are
    owned by the caller and never passed through this function. Unresolved or
    mismatching references remain intact rather than becoming a trusted view.
    """
    result=[]
    for item in records:
        kind=item.get('kind')
        if kind not in ('primary_decision','tool_result'):
            result.append(item);continue
        try:
            e=store.resolve(task_id,item['ref'])
        except (KeyError,ValueError):
            result.append(item);continue
        if e['seq']>cutoff or e['kind']!=kind or e['role']!=item.get('role'):
            result.append(item);continue
        if kind=='primary_decision' and e['role']!='primary':
            result.append(item);continue
        if kind=='tool_result' and e['role']!='tool':
            result.append(item);continue
        body=e['body']
        if not isinstance(body,dict):
            result.append(item);continue
        # A verbatim returned receipt remains in last_result, so its extra copy
        # can disappear only when it is the same archived body.
        if kind=='tool_result' and last_result and last_result.get('evidence_ref')==e['ref']:
            comparable={k:v for k,v in last_result.items() if k not in ('evidence_ref','deduplicated')}
            if comparable==body:continue
        if len(item.get('text',''))<=480:
            result.append(item);continue
        view={'ref':e['ref'],'role':e['role'],'kind':kind,'excerpt_only':True,
              'details':{'tool':'context.read','arguments':{'ref':e['ref']}},'source_chars':len(e['text'])}
        if kind=='primary_decision':
            view['scope']='模型此前的操作意图，不代表已执行；代码及参数保存在引用原文。'
            view['action']=body.get('action');view['tool']=body.get('tool')
            args=body.get('arguments')
            if isinstance(args,dict):
                view['target']={k:v for k,v in args.items() if k in ('name','path','revision') and isinstance(v,str) and len(v)<=100}
        else:
            # A procedure note is not an execution receipt; do not collapse it.
            if 'tool' not in body or 'ok' not in body:
                result.append(item);continue
            view['scope']='历史工具回执定位，非完整正文；需要输出细节时读取引用。ok不等于测试通过。'
            view.update(tool=body['tool'],ok=body['ok'],operation_id=body.get('operation_id'))
            value=body.get('value')
            if isinstance(value,dict):
                view['receipt']={k:v for k,v in value.items()
                    if k in ('name','revision','status','passed','tested','path','bytes','retrieved','validation_scope')
                    and (v is None or type(v) in (bool,int,float) or isinstance(v,str) and len(v)<=120)}
            if body.get('error'):view['has_error']=True
        # Never enlarge a small record or substitute a model-generated summary.
        result.append(view if len(json.dumps(view,ensure_ascii=False))<len(json.dumps(item,ensure_ascii=False)) else item)
    return result
