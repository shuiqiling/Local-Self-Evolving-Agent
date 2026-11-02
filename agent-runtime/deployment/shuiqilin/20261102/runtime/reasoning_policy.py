"""Request-level light/full thinking with separate visible-answer accounting."""
import asyncio
import json
import sqlite3


class PreparedRequest(dict):
    def __init__(self,value,final_tokens):
        super().__init__(value);self.final_tokens=final_tokens


def prepare_daily(body,options):
    if isinstance(body,PreparedRequest):return body
    final=int(body['max_tokens'])
    if not options.get('enabled'):
        return PreparedRequest({**body,'presence_penalty':0,'chat_template_kwargs':{'enable_thinking':False}},final)
    budget=int(options.get('reasoning_budget_tokens',256));assert 0<budget<=512
    return PreparedRequest({**body,'presence_penalty':0,'temperature':.6,'top_p':.95,'top_k':20,'min_p':0,
        'chat_template_kwargs':{**body.get('chat_template_kwargs',{}),'enable_thinking':True},
        'reasoning_format':'deepseek','reasoning_budget_tokens':budget,
        'max_tokens':final+budget+16},final)


def teacher_settings(options):
    return {'thinking_enabled':bool(options.get('enable_thinking',False)),
            'reasoning_budget_tokens':int(options.get('reasoning_budget_tokens',-1)),
            'final_output_tokens':int(options.get('max_output_tokens',384)),
            'context_tokens':int(options.get('backend_context_tokens',8192)),
            'request_timeout_seconds':int(options.get('request_timeout_seconds',300))}


def busy(paths):
    for path in paths:
        with sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True) as db:
            if db.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]:return True
    return False


class ForegroundYield(Exception):pass


async def post_while_idle(base,body,paths,timeout,*,client=None,busy_reader=busy,poll_seconds=.5):
    """Close the backend request when foreground work appears, before retrying."""
    import httpx
    owned=client is None
    if owned:client=httpx.AsyncClient(trust_env=False)
    task=None
    try:
        if await asyncio.to_thread(busy_reader,paths):raise ForegroundYield()
        task=asyncio.create_task(client.post(base+'/v1/chat/completions',json=body,timeout=timeout))
        while True:
            done,_=await asyncio.wait([task],timeout=poll_seconds)
            if await asyncio.to_thread(busy_reader,paths):raise ForegroundYield()
            if done:
                response=task.result();response.raise_for_status();return response.json()
    except httpx.HTTPError as error:raise OSError(str(error)) from error
    finally:
        if task is not None and not task.done():task.cancel();await asyncio.gather(task,return_exceptions=True)
        if owned:await client.aclose()


def full_request(body,options,call,base):
    policy=teacher_settings(options);assert policy['thinking_enabled'] and policy['reasoning_budget_tokens']==-1
    value={**body,'chat_template_kwargs':{'enable_thinking':True},'reasoning_format':'deepseek',
           'reasoning_budget_tokens':-1,'temperature':.6,'top_p':.95,'top_k':20,'min_p':0,'presence_penalty':0}
    template=call(base,'/apply-template',value,timeout=30)['prompt']
    count=len(call(base,'/tokenize',{'content':template,'add_special':False,'parse_special':True},timeout=30)['tokens'])
    maximum=policy['context_tokens']-count-32
    if maximum<policy['final_output_tokens']:raise ValueError('Insufficient context remaining for the final teacher notes')
    value['max_tokens']=maximum
    return value,{'input_tokens':count,'total_generation_limit':maximum,**policy}
