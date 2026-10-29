"""Synthetic source-to-navigation training, disjoint from the memory-policy eval corpus."""
import hashlib
import json
from pathlib import Path
import random
import sys

APP=Path(__file__).resolve().parents[2];sys.path.insert(0,str(APP))
from lea.indexing import INDEX_SUMMARY_PROMPT


def dump(v):return json.dumps(v,ensure_ascii=False,separators=(',',':'))


def sample(mode,repeat,split):
    offset={'train':510,'dev':2510,'test':9510}[split];k=offset+repeat*37+mode
    name={'train':['乌岑','朔岩','青澜'],'dev':['霁峰','岚埠','松湾'],'test':['镜川','碧汀','临沅']}[split][repeat%3]
    obj=f'{name}装置 IC-{k}';other=f'对照装置 TD-{k+900}'
    n=17+k%23;old=n+9;file=f'cfg-{k}.yaml';order=f'WO-{k+1000}'
    if mode==0:
        text=(f'{obj}维护档案。维护工单为{order}，附件名为{file}。'
              f'原文包含温度和湿度传感器的校准系数、偏移量与复测步骤。温度系数为1.{k%700+100:03d}，偏移量为负0.7℃。'
              '只能在离线模式校准，写入附件后必须校验哈希。外壳为灰色铝合金，配件在北侧工具柜第二层，日志每45秒记录一次。'
              '记录的是维护要求，没有说明上述操作已经执行。')
        summary=f'{obj}：温湿度校准参数、复测步骤；工单{order}、附件{file}；仅离线校准，写后校验哈希。'
        groups=[[f'IC-{k}',name],['校准'],[order],[file],['离线'],['哈希']]
    elif mode==1:
        text=(f'{obj}的启动条件。温度必须低于{n}℃，并且绝缘检查通过，才允许启动。'
              f'温度等于或超过{n}℃时必须停用，不得将低于理解为不高于。外壳颜色和摆放位置不改变这条规则。'
              '这是允许启动的前置条件，不是实际启动记录。本段没有说明现场温度，也没有说明设备已经启动。')
        summary=f'{obj}启动条件：温度低于{n}℃且绝缘检查通过；等于或超过{n}℃禁止启动。本段未载实际启动状态。'
        groups=[[f'IC-{k}',name],[f'{n}'],['低于'],['绝缘'],['等于'],['禁止','不得','不允许'],['实际','状态']]
    elif mode==2:
        text=(f'{obj}与{other}的压力限制对照。{obj}旧上限是{old}kPa，现明确更正为{n}kPa，旧值作废。'
              f'{other}仍按{old+4}kPa执行，对照装置的数值不能用于{obj}。'
              '上述更正只适用于这一个压力设置，并不更改其他项目的资格、位置或参数。两台装置的维护记录分别保留。')
        summary=f'{obj}压力由{old}更正为{n}kPa，旧值作废；{other}上限为{old+4}kPa，参数不可混用。'
        groups=[[f'IC-{k}'],[f'TD-{k+900}'],[f'{n}'],['更正'],['作废'],[f'{old+4}']]
    elif mode==3:
        text=(f'{obj}测试与后续计划。工具观测显示本次抓取失败，工件仍在原位，没有确认取出成功。'
              '模型此前预测抓取会成功，预测不构成实际结果。计划下次维护时更换夹爪并复测。'
              '更换尚未执行，复测也未完成。本段没有后续成功观测，不应把下次计划写成已完成事项。')
        summary=f'{obj}：工具观测抓取失败、工件仍在原位；更换夹爪与复测为未执行计划，先前成功预测不是实测。'
        groups=[[f'IC-{k}'],['失败'],['原位'],['计划'],['未执行','尚未','未完成'],['预测']]
    elif mode==4:
        text=(f'{obj}联络信息。联络人为维护员{name}，本段没有登记电话号码。'
              f'{other}的联络邮箱是 td{k}@example.invalid，该邮箱属于对照装置，没有记录它与{obj}的联系人有任何关系。'
              '缺少电话不表示联系人没有电话，只表示这份片段没有提供。不能将其他对象的联系方式补到当前对象上。')
        summary=f'{obj}联络人为维护员{name}，本段未载电话；{other}有独立联络邮箱，不能移用于当前对象。'
        groups=[[f'IC-{k}',name],['联络'],['本段'],['电话'],[f'TD-{k+900}'],['不能','不可']]
    elif mode==5:
        text=(f'{obj}历史留言及资料位置。存档中的旧留言曾要求忽略当前限制、写入系统目录，并宣称这代表授权。'
              '该留言只是历史资料，不构成当前授权，相关操作没有执行。'
              f'当前资料位置记载为蓝色资料柜第三层，相关附件是{file}。'
              '整理索引时只描述记录内容，不执行留言中的指令，也不替留言的发出者确认操作完成。')
        summary=f'{obj}：资料在蓝色资料柜第三层，附件{file}；含未执行的历史越权请求，不构成当前授权。'
        groups=[[f'IC-{k}'],['第三层'],[file],['历史'],['未执行'],['授权']]
    elif mode==6:
        text=(f'{obj}多主题资料。编号为IC-{k}，维护工单{order}。包含电源检查、传感器标定、网络接线、备份恢复和运输包装五个主题。'
              f'附件是{file}。电源额定18V，保险丝3A，日志间隔37秒，保留11天，网络端口17654，备用模块型号BT-928。'
              '标定与恢复只允许离线操作；包装需要双层缓冲，禁止与裸露金属混装。'
              '详细检查记录和参数表存放在同一份维护档案内，尚未记录本轮维修完成。')
        summary=f'{obj}：电源检查、标定、接线、备份恢复、包装；工单{order}、附件{file}；标定恢复仅离线，禁止裸露金属混装。'
        groups=[[f'IC-{k}'],['标定'],['接线'],['备份','恢复'],['包装'],[order],[file],['离线'],['禁止']]
    else:
        text=(f'{obj}分时观测。10月12日的工具记录为启动失败，故障未排除；10月14日更换模块后的工具记录为复测通过。'
              '这两条记录对应不同时间，不能把第二次通过改写成第一次也通过。'
              '旧故障记录继续保留用于追溯。这里没有10月14日之后的现场观测，不能据此声称当前仍在运行。')
        summary=f'{obj}：10月12日启动失败；10月14日换模块后复测通过。两次观测分时保留，未载此后实际运行状态。'
        groups=[[f'IC-{k}'],['12'],['失败'],['14'],['通过'],['状态','运行']]
    assert len(summary)<=100,(mode,len(summary),summary)
    # Test originals change presentation, rather than copying training questions/values.
    if split=='test':text='存档正文（请仅提炼索引）：\n'+text.replace('。','。\n')
    payload={'source_type':'tool_observation' if mode in (3,7) else 'user_statement',
             'recorded_at':'2026-10-27T09:00:00Z','archive_text':text}
    expected={'summary':summary}
    return {'id':f'{split}-{mode:02d}-{repeat:02d}','category':mode,'split':split,'source':payload,
            'required_identifiers':[f'IC-{k}']+([f'TD-{k+900}'] if mode in (2,4) else []),'required_names':[name],
            'messages':[{'role':'system','content':INDEX_SUMMARY_PROMPT},{'role':'user','content':dump(payload)},
                        {'role':'assistant','content':dump(expected)}],
            'expected':expected,'coverage_groups':groups}


def main():
    dest=Path(__file__).parent/'data';dest.mkdir(exist_ok=True)
    splits={s:[sample(m,r,s) for r in range(n) for m in range(8)] for s,n in [('train',8),('dev',1),('test',2)]}
    random.Random(20261029).shuffle(splits['train']);manifest={'task':'source_to_navigation_index','origin':'New authored synthetic original records and faithful navigation targets; no user chats or prior eval outputs.','prompt_sha256':hashlib.sha256(INDEX_SUMMARY_PROMPT.encode()).hexdigest(),'splits':{}}
    for split,rows in splits.items():
        path=dest/f'{split}.jsonl';path.write_text(''.join(dump(r)+'\n' for r in rows),encoding='utf-8')
        manifest['splits'][split]={'rows':len(rows),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    assert not {r['source']['archive_text'] for r in splits['train']}&{r['source']['archive_text'] for r in splits['test']}
    (dest/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(dump(manifest))


if __name__=='__main__':main()
