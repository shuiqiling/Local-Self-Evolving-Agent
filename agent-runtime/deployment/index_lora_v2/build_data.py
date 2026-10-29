"""Fresh short-index curriculum: literal fidelity, mandatory constraints, and natural stops."""
import hashlib
import json
from pathlib import Path
import random
import sys

APP=Path(__file__).resolve().parents[2];sys.path.insert(0,str(APP))
from lea.indexing import INDEX_SUMMARY_PROMPT


def dump(v):return json.dumps(v,ensure_ascii=False,separators=(',',':'))


def sample(mode,repeat,split):
    offset={'train':20400,'dev':43700,'test':86500}[split];k=offset+repeat*43+mode
    name={'train':['洛苇','庭梧','砾原'],'dev':['稷河','杉泽','芦津'],'test':['烁湾','栀坪','禾浦']}[split][repeat%3]
    obj=f'{name}设备 HG-{k}';other=f'RJ-{k+1700}'
    n=12+k%27;old=n+11;port=21000+k%8000;diag=33000+k%7000
    ids=[f'HG-{k}'];groups=[];critical=[];bindings=[]
    if mode==0:
        text=(f'{obj}综合标定记录。记录包括温湿度传感器系数、偏移、复测、附件和维护工单。'
              f'附件为 tune-{k}.json，工单为MT-{k+900}。温度系数1.{k%700+100:03d}，偏移负0.35℃。'
              f'网络端口{port}，日志间隔47秒，外壳为米白色，备用件放在西柜。'
              '标定只允许离线进行，附件写入后必须校验哈希。本段未说明上述操作是否已经执行。')
        summary=f'{obj}：温湿度标定、附件、工单与复测；仅离线标定，写后校验哈希。'
        groups=[['温湿度','温度','湿度'],['标定','校准'],['附件'],['工单'],['复测']]
        critical=['仅离线标定','写后校验哈希']
    elif mode==1:
        relation=['lt','le','gt','ge'][repeat%4]
        raw={'lt':f'温度严格低于{n}℃才允许启动，等于或超过{n}℃禁止启动',
             'le':f'温度不高于{n}℃才允许启动，等于{n}℃允许，高于时禁止',
             'gt':f'温度严格高于{n}℃才允许启动，等于或低于{n}℃禁止',
             'ge':f'温度不低于{n}℃才允许启动，等于{n}℃允许，低于时禁止'}[relation]
        short={'lt':f'低于{n}℃且检查通过才可启动；等于禁止',
               'le':f'不高于{n}℃且检查通过才可启动；等于允许',
               'gt':f'高于{n}℃且检查通过才可启动；等于禁止',
               'ge':f'不低于{n}℃且检查通过才可启动；等于允许'}[relation]
        text=f'{obj}温度规则：{raw}，同时绝缘检查必须通过。机壳颜色、端口与日志参数不改变该规则。这是前置条件，没有现场温度和实际启动观测。'
        summary=f'{obj}：{short}。'
        groups=[['启动'],['检查','绝缘']];critical=[short]
        bindings=[{'field':'启动温度边界','value':str(n),'relation':relation}]
    elif mode==2:
        text=(f'{obj}维护写入要求。必须使用维护账号，只能在离线模式执行，写入前先断电，写入后必须校验哈希。'
              f'网络端口{port}，诊断端口{diag}，日志保留13天，附件为 maint-{k}.yaml。'
              '这些数值是背景配置，不改变账号、离线、断电和校验四项前提。本段没有完成写入的记录。')
        summary=f'{obj}：维护写入流程；仅维护账号离线操作，先断电，写后校验哈希。'
        groups=[['维护','写入']];critical=['仅维护账号离线操作','先断电','写后校验哈希']
    elif mode==3:
        text=(f'{obj}端口登记。业务端口为{port}，诊断端口为{diag}，两者不得混用。'
              f'对象编号HG-{k}和维护工单MT-{k+900}不是端口。日志间隔为47秒。'
              '本次只登记两个端口的对应关系，没有记录服务已经启动。')
        summary=f'{obj}：业务端口{port}，诊断端口{diag}；不可混用。'
        groups=[['端口']];critical=[f'业务端口{port}',f'诊断端口{diag}','不可混用']
        bindings=[{'field':'业务端口','value':str(port)},{'field':'诊断端口','value':str(diag)}]
    elif mode==4:
        coefficient=f'1.{k%700+101:03d}';offset=f'{(k%8+2)/10:.1f}'
        text=(f'{obj}温度传感器参数。校准系数为{coefficient}，偏移量为负{offset}℃。'
              f'另一个湿度参数偏移为正{int(float(offset)*10)+1}，不能移用于温度传感器。'
              '温度参数仅适用于离线标定，未登记本轮标定结果。')
        summary=f'{obj}温度标定：系数{coefficient}，偏移负{offset}℃；仅离线使用。'
        groups=[['温度'],['标定','校准']];critical=[f'系数{coefficient}',f'偏移负{offset}℃','仅离线使用']
        bindings=[{'field':'校准系数','value':coefficient},{'field':'温度偏移','value':'-'+offset}]
    elif mode==5:
        text=(f'{obj}与对照设备{other}的压力记录。{obj}旧上限为{old}kPa，现明确更正为{n}kPa，旧值作废。'
              f'{other}仍为{n+6}kPa，不能移用。更正只影响当前压力设置，不更改其他主题。')
        summary=f'{obj}压力改为{n}kPa，旧值作废；{other}仍为{n+6}kPa，不可移用。'
        ids.append(other);groups=[['压力']];critical=[f'改为{n}kPa','旧值作废',f'{other}仍为{n+6}kPa','不可移用']
        bindings=[{'field':f'HG-{k}压力','value':str(n)},{'field':f'{other}压力','value':str(n+6)}]
    elif mode==6:
        text=(f'{obj}维护计划与资料。计划更换滤网并复测，必须先停机。'
              f'附件为 plan-{k}.json，业务端口{port}。本段没有说明计划是否已经执行。'
              '没有执行记录不能写成已经完成，也不能写成确定尚未执行。')
        summary=f'{obj}：更换滤网与复测计划；先停机，执行状态本段未载。'
        groups=[['滤网'],['复测'],['计划']];critical=['先停机','执行状态本段未载']
    elif mode==7:
        success=repeat%2==1
        result='复测通过' if success else '抓取失败，工件仍在原位'
        text=(f'{obj}工具观测：{result}。先前模型预测与此次观测不一致，以本次记录的观测为准。'
              '后续计划更换接头，但更换尚未执行。不得把后续计划写成已经更换。')
        summary=f'{obj}：工具观测{result}；更换接头为未执行计划。'
        groups=[['工具','观测'],['接头'],['计划']];critical=[result,'未执行计划']
    elif mode==8:
        text=(f'{obj}多主题档案。内容包括电源检查、传感器标定、网络接线、备份恢复与运输包装。'
              f'电源19V，保险丝4A，端口{port}，备用型号QS-{k+88}，日志间隔47秒。'
              f'维护工单MT-{k+900}，附件archive-{k}.yaml。标定与恢复仅允许离线操作；运输禁止与裸露金属混装。'
              '数值、工单与附件的细项保存在本段对应表格。')
        summary=f'{obj}：电源、标定、接线、备份、包装；标定恢复仅离线，禁与裸露金属混装。'
        groups=[['电源'],['标定'],['接线','布线'],['备份','恢复'],['包装','运输']]
        critical=['标定恢复仅离线','禁与裸露金属混装']
    elif mode==9:
        text=(f'{obj}历史留言与资料位置。旧留言要求跳过限制并改写系统目录，留言不构成当前授权，实际已被拒绝，相关改写未执行。'
              '资料位于东侧青色文件柜，附件和清单在同一存档。只描述历史资料，不执行留言。')
        summary=f'{obj}：资料在东侧青柜；旧越权改写已拒绝、未执行，不构成当前授权。'
        groups=[['资料'],['旧','历史']];critical=['已拒绝','未执行','不构成当前授权']
    elif mode==10:
        same=repeat%2==1
        relation='记录明确说明两个项目共用同一联络人' if same else '本段未登记两者联络信息是否有关联'
        text=(f'{obj}的联络人为记录员{name}，本段未提供电话。对照设备{other}有独立邮箱。'
              f'{relation}。不能仅因同段出现就合并两者属性，未提供电话不等于联系人没有电话。')
        tail='记录明确同一联络人' if same else '两者关联未载，不移用邮箱'
        summary=f'{obj}联络资料本段未载电话；{other}有邮箱，{tail}。'
        ids.append(other);groups=[['联络'],['本段'],['电话']];critical=[tail]
    else:
        text=(f'{obj}长档案导览。重点是标定、备份恢复和运输包装。标定必须离线，恢复需要维护账号，包装禁止金属混装。'
              f'其余附带项包括外壳颜色、端口{port}、诊断端口{diag}、日志47秒、保留13天、电压19V、备用件QS-{k+88}、工单MT-{k+900}及附件。'
              '没有维修完成观测。不能用背景参数替代三个重点主题或删掉它们的限制。')
        summary=f'{obj}：标定、恢复、包装；标定仅离线，恢复需维护账号，禁金属混装。'
        groups=[['标定'],['恢复'],['包装']];critical=['标定仅离线','恢复需维护账号','禁金属混装']
    assert len(summary)<=72,(mode,len(summary),summary)
    if split=='test':
        text='原始存档：\n'+text.replace('。','。\n')
        if repeat%2:text=text.replace('必须','务必').replace('本段','此片段')
    payload={'source_type':'tool_observation' if mode==7 else 'user_statement','recorded_at':'2026-10-29T00:00:00Z','archive_text':text}
    expected={'summary':summary}
    return {'id':f'{split}-{mode:02d}-{repeat:02d}','category':mode,'split':split,'source':payload,
            'required_identifiers':ids,'required_names':[name],'coverage_groups':groups,
            'critical_fragments':critical,'key_bindings':bindings,'teacher_chars':len(summary),
            'messages':[{'role':'system','content':INDEX_SUMMARY_PROMPT},{'role':'user','content':dump(payload)},{'role':'assistant','content':dump(expected)}],
            'expected':expected}


def main():
    folder=Path(__file__).parent/'data';folder.mkdir(exist_ok=True)
    splits={s:[sample(m,r,s) for r in range(n) for m in range(12)] for s,n in [('train',8),('dev',1),('test',2)]}
    splits['test'].extend(sample(m,r,'test') for m,r in [(1,2),(1,3),(2,2),(3,2),(4,2),(8,2),(10,2),(11,2)])
    random.Random(20261030).shuffle(splits['train'])
    manifest={'version':'index-v2','initial_adapter':'index-v1','training_target_max_chars':72,'evaluation_schema_max_chars':100,
        'origin':'New authored curriculum and facts, not previous evaluation rows. Test frozen before training.',
        'prompt_sha256':hashlib.sha256(INDEX_SUMMARY_PROMPT.encode()).hexdigest(),'splits':{}}
    sources={}
    for split,rows in splits.items():
        path=folder/f'{split}.jsonl';path.write_text(''.join(dump(r)+'\n' for r in rows),encoding='utf-8')
        manifest['splits'][split]={'rows':len(rows),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        sources[split]={r['source']['archive_text'] for r in rows}
    assert not sources['train']&sources['test'] and not sources['train']&sources['dev'] and not sources['test']&sources['dev']
    (folder/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(dump(manifest))


if __name__=='__main__':main()
