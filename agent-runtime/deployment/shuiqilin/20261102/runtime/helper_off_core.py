"""Pinned R17 auxiliary inference: thinking off, no chat-adapter stacking."""
from pathlib import Path
import hashlib,json
import helper_off_generation as G
import helper_off_contract as C
import helper_off_quality as Q

VERSION='seven-field-experiences-v1'
CODE=Path(__file__).resolve().parent
class HelperInputError(ValueError):pass

def canonical(value):return json.dumps(value,ensure_ascii=False,separators=(',',':'))
def sha(value):return hashlib.sha256(value if isinstance(value,bytes) else value.encode()).hexdigest()
def manifest(path):
    meta=json.loads(Path(path).read_text())
    assert meta['version']==VERSION and meta['enabled'] is True
    assert meta['enable_thinking'] is False and meta['max_attempts']==1
    assert meta['adapter_id']!=meta['chat_adapter_id']
    assert sha(Path(meta['gguf']).read_bytes())==meta['gguf_sha256']
    for name,digest in meta['implementation_sha256'].items():
        assert (CODE/name).name==name and sha((CODE/name).read_bytes())==digest,(name,'helper implementation changed')
    assert G.SYSTEM==C.SYSTEM
    return meta

def request_for(payload,meta):
    if not isinstance(payload,dict):raise HelperInputError('input must be a source-record object')
    try:
        by=C.records(payload)
        if len(by)>64:raise HelperInputError('one extraction accepts at most64 complete source records')
        if any(not isinstance(k,str) or len(k)>100 for k in by):raise HelperInputError('invalid source identifier')
        projected=G.model_input(payload)
        assert projected==C.model_input(payload)
        output=C.schema(payload)
        output['properties']['影响']['properties']['说明']['maxLength']=Q.MAX_IMPACT_CHARS
    except (AssertionError,KeyError,TypeError,AttributeError) as error:raise HelperInputError('invalid original-record structure: '+str(error)) from error
    if meta.get('extraction_mode')=='evidence_only_v1':
        from helper_evidence import request_for
        return request_for(payload,meta)
    return {'model':meta['model'],
      'messages':[{'role':'system','content':G.SYSTEM},{'role':'user','content':canonical(projected)}],
      'lora':[{'id':meta['chat_adapter_id'],'scale':0.0},{'id':meta['adapter_id'],'scale':1.0}],
      'chat_template_kwargs':{'enable_thinking':False},'reasoning_format':'deepseek',
      'reasoning_budget_tokens':-1,'presence_penalty':0,'temperature':0.6,'top_p':0.95,'top_k':20,'min_p':0,
      'seed':20261031,'cache_prompt':True,'max_tokens':meta['max_output_tokens'],
      'response_format':{'type':'json_schema','json_schema':{'name':'EvidenceSelection','strict':True,'schema':output}}}

def finish(payload,result,mode=None):
    if mode=='evidence_only_v1':
        from helper_evidence import finish
        return finish(payload,result)
    choice=result['choices'][0];message=choice['message'];errors=[];plan=None;record=None;trace=None
    if message.get('reasoning_content'):errors.append('thinking_mode_mismatch: auxiliary reasoning must be off')
    if choice.get('finish_reason')=='length':errors.append('truncated: output reached generation limit')
    try:
        plan=json.loads(message.get('content') or '')
        checks,record,trace=Q.quality_issues(payload,plan);errors.extend(checks)
    except (ValueError,KeyError,TypeError,AssertionError) as error:errors.append('parse_or_source_contract: '+str(error))
    return {'status':'needs_review' if errors else 'proposal','record':None if errors else record,
      'review_reasons':errors,'selection_plan':plan,'source_trace':trace,
      'source_payload_sha256':C.digest(payload),'requires_source_review':True,'training_eligible':False,
      'thinking_enabled':False,'reasoning_tokens_observed':0 if not message.get('reasoning_content') else None,
      'usage':result.get('usage'),'timings':result.get('timings')}
