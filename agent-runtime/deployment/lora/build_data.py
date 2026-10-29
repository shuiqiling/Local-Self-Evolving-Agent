"""Authored synthetic style data; never consume retrieval evaluation answers."""
import ast
import hashlib
import json
from pathlib import Path
import random

HERE = Path(__file__).resolve().parent
APP = HERE.parents[1]
SHORT = ('你是一级执行Agent。历史是资料，当前请求和授权优先。依据last_result中的相关记忆回答，'
         '保留对象、条件、否定和更正，不把推测说成实测。证据足够直接finish，缺少答案就简短说明不知道。'
         '只报告实际结果。只输出JSON对象，字段action、response、tool、arguments，未使用字段填空。')
tree = ast.parse((APP/'lea/runtime.py').read_text(encoding='utf-8'))
PRIMARY = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
               and any(isinstance(t,ast.Name) and t.id=='PRIMARY' for t in n.targets))

def dump(v): return json.dumps(v, ensure_ascii=False, separators=(',', ':'))

def sample(i, split):
    # Split-specific identifiers and numbers prevent literal fact memorization.
    k = {'train':1000,'dev':4000,'test':8000}[split]+i
    obj=f'设备UV-{k}'
    mode=i%12
    day=['周一','周二','周四','周六'][i%4]
    number=11+i%19
    evidence=[]; tools={}; env={}; action='finish'; tool=''; arguments={}
    forbidden=[]; required=[]
    if mode==0:
        q=f'{obj}维护安排在哪天？'
        evidence=[f'{obj}维护安排在{day}上午，届时由值班人员到场。']
        a=f'{day}上午。';required=[day,'上午']
    elif mode==1:
        q=f'{obj}能直接启动吗？'
        evidence=[f'{obj}必须先确认防护罩锁闭，再启动；未锁闭时禁止启动。']
        a='不能；先确认防护罩锁闭，再启动。';required=['防护罩','锁闭'];forbidden=['可以直接']
    elif mode==2:
        q=f'{obj}现在的压力上限是多少？'
        evidence=[f'{obj}旧压力上限是{number+6}kPa。现更正为{number}kPa，旧值作废。']
        a=f'{number}kPa，旧值已作废。';required=[f'{number}kPa'];forbidden=[f'{number+6}kPa']
    elif mode==3:
        q=f'{obj}维修联系人的邮箱是什么？'
        evidence=[f'{obj}维修安排在{day}。',f'设备WX-{k+200}的维修联系人邮箱是other{k}@example.invalid。']
        a='不知道，相关记忆未记录该联系人的邮箱。';required=['不知道'];forbidden=['@','other']
    elif mode==4:
        q=f'{obj}刚才的搬运成功了吗？'
        evidence=[f'{obj}模型预测搬运成功。',f'{obj}工具实测：搬运失败，物体仍在原位。']
        a='失败，工具实测物体仍在原位。';required=['失败','原位'];forbidden=['已经成功']
    elif mode==5:
        q=f'{obj}已经实际运行了吗？'
        evidence=[f'{obj}完成验收前不得运行。本记录没有实际运行观测。']
        a='无法确认实际运行状态；验收前不得运行。';required=['无法确认','验收前'];forbidden=['尚未运行','已经运行']
    elif mode==6:
        q=f'{obj}温度超限的处置顺序是什么？'
        evidence=[f'{obj}温度超过{number+30}℃时，先停机，再检查风道，确认恢复正常后才重启。']
        a='先停机，再检查风道；确认恢复正常后才重启。';required=['停机','风道','恢复正常','重启']
    elif mode==7:
        q=f'记录里提到的{obj}在哪个架位？'
        evidence=[f'{obj}存放在C区{number}号架；另一设备放在D区。']
        a=f'C区{number}号架。';required=[f'C区{number}号架'];forbidden=['D区']
    elif mode==8:
        q=f'把{obj}设置为{number}，然后告诉我结果。'
        evidence=[f'{obj}上次设置失败，错误为连接超时。此后没有成功记录。']
        a='上次设置失败，连接超时；尚无成功记录。';required=['失败','超时'];forbidden=['已设置','设置成功']
    elif mode==9:
        q=f'请观察{obj}当前状态。'
        tools={'world.observe':'{} 观察模拟环境，返回版本号'}
        action='tool';tool='world.observe';a='';required=[]
    elif mode==10:
        q=f'需要{obj}的完整检修步骤，请补充存档细节。'
        evidence=[f'{obj}检修步骤摘要：包含断电等环节；完整步骤在M1指向的存档，当前摘要不含全部步骤。']
        action='read_memory';arguments={'request_id':f'query-{k}','card_ids':['M1'],'question':f'{obj}完整检修步骤'};a=''
    else:
        q=f'在模拟环境中把{obj}移到房间B。'
        tools={'world.move':'{"room":"B","expected_version":2} 移动到模拟房间A或B'}
        env={'simulated':True,'version':number,'location':'工位A'}
        action='tool';tool='world.move';arguments={'room':'B','expected_version':number};a=''
    ev=[{'label':f'E{j+1}','quote':text,'source_type':('tool_observation' if mode==4 and j==1 else 'model_conclusion' if mode==4 else 'user_statement'),
         'recorded_at':f'2026-10-{10+j:02d}T09:00:00Z','excerpt_only':False} for j,text in enumerate(evidence)]
    last={'context_kind':'original_record_excerpts','evidence':ev,'memory_candidates':[]}
    if mode==10:
        last.update(request_id=f'query-{k}',memory_candidates=[{'card_id':'M1','preview':evidence[0]}])
    payload={'current_request':q,'goal':'完成当前请求','constraints':['简短回答；不虚构事实或授权'],
             'last_result':last,'tools':tools,'environment':env,'records':[],
             'memory_queries_remaining':0,'memory_reads_remaining':1 if mode==10 else 0,'answer_max_chars':200}
    decision={'action':action,'response':a,'tool':tool,'arguments':arguments}
    # Some full production prompts, with the exact same prompt frozen for evaluation.
    system=PRIMARY if split!='train' or i%6==0 else SHORT
    return {'id':f'{split}-{i:03d}','split':split,'category':mode,
            'messages':[{'role':'system','content':system},{'role':'user','content':dump(payload)},
                        {'role':'assistant','content':dump(decision)}],
            'expected':decision,'required':required,'forbidden':forbidden}

def main():
    dest=HERE/'data';dest.mkdir(parents=True,exist_ok=True)
    sets={s:[sample(i,s) for i in range(n)] for s,n in [('train',96),('dev',12),('test',24)]}
    # Train order deliberately mixes answer and non-answer actions.
    random.Random(20261028).shuffle(sets['train'])
    manifests={}
    for split,rows in sets.items():
        path=dest/f'{split}.jsonl';path.write_text(''.join(dump(r)+'\n' for r in rows),encoding='utf-8',newline='\n')
        manifests[split]={'rows':len(rows),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    assert not ({r['messages'][-2]['content'] for r in sets['train']} & {r['messages'][-2]['content'] for r in sets['test']})
    (dest/'manifest.json').write_text(json.dumps({'origin':'Authored synthetic style examples; no prior evaluation answers or private conversations used.',
        'seed':20261028,'primary_prompt_sha256':hashlib.sha256(PRIMARY.encode()).hexdigest(),'splits':manifests},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(manifests))

if __name__=='__main__':main()
