"""New synthetic memory-policy examples; no user conversations or prior test answers."""
import hashlib
import json
from pathlib import Path
import random
import sys

APP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(APP))
from lea.runtime import PRIMARY_SCOPED

SHORT = ('你是使用本地记忆系统的执行Agent，只输出JSON对象action、response、tool、arguments。'
         'memory_candidates是原文向量召回后提供的片段索引，核对title中的对象和适用条件，不能只选最高分。'
         '摘要或已有evidence足够就直接finish，不重复确认；精确数值已给出也不用读。'
         '缺少关键细节或用户要求原文时，按需read_memory，arguments包含当前request_id、必要card_ids和具体question。'
         '工具仅搜索选中片段；局部未找到不代表全库没有。保留条件、否定和更正，不把规定当实际状态。'
         '遵守memory_queries_remaining、memory_reads_remaining和answer_max_chars，未使用字段填空。')


def dump(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def sample(mode, repeat, split):
    k = {'train': 1200, 'dev': 4500, 'test': 8300}[split] + repeat*31 + mode
    name = {'train': ['映川', '柏湾', '漓浦'], 'dev': ['宁桥', '榆岗', '柳汀'], 'test': ['昭汀', '墨岭', '沅港']}[split][repeat % 3]
    obj = f'{name}设备 PX-{k}'; other = f'另湾设备 QT-{k+701}'
    number = 13 + (k % 23); day = ['周一', '周二', '周四', '周六'][k % 4]
    qid = f'query_{k:016x}'
    last = {'request_id': qid, 'context_kind': 'segment_navigation_only', 'archive_read': False,
            'memory_candidates': [], 'evidence': []}
    cards = last['memory_candidates']; evidence = last['evidence']
    reads = 1; queries = 0; action = 'finish'; answer = ''; arguments = {}; required = []; forbidden = []
    wording = 0 if split == 'train' else (1 if split == 'dev' else 2+repeat%2)

    def card(title, summary, score=.78, **extra):
        value = {'card_id': f'M{len(cards)+1}', 'title': title, 'summary': summary, 'match_score': score,
                 'source_types': ['user_statement'], 'recorded_at': f'2026-10-{11+k%12:02d}T08:00:00Z',
                 'summary_pending': False, 'summary_rejected': False, **extra}
        cards.append(value); return value['card_id']

    def quote(text, source='user_statement'):
        evidence.append({'event_ref': f'E{len(evidence)+1}', 'quote': text, 'source_type': source,
                         'recorded_at': '2026-10-24T08:00:00Z', 'excerpt_only': True})
        last.update(archive_read=True, context_kind='selected_original_excerpts')

    def choose(identifiers, question):
        nonlocal action, arguments
        action = 'read_memory'; arguments = {'request_id': qid, 'card_ids': identifiers, 'question': question}

    if mode == 0:
        q = [f'{obj}什么时候维护？', f'看已有摘要，{obj}的维护时间是？', f'请给出{obj}排定的维护时段。', f'{obj}计划哪天上午检修？'][wording]
        card(f'{obj}维护安排', f'{obj}维护窗口为{day}上午，负责人按计划到场。')
        answer = f'摘要记载为{day}上午。'; required = [day, '上午']
    elif mode == 1:
        q = [f'{obj}温度等于{number}℃时可以启动吗？', f'{obj}达到{number}℃能启动吗，依据摘要回答。', f'温度刚好{number}℃，{obj}符合启动条件吗？', f'请判断：{obj}在{number}℃能否启动？'][wording]
        card(f'{obj}启动条件', f'必须低于{number}℃且完成绝缘检查后才可启动；等于上限不允许启动。')
        answer = f'不可以。摘要要求低于{number}℃，并完成绝缘检查。'; required = ['不可以', f'{number}', '绝缘']; forbidden = ['可以直接']
    elif mode == 2:
        q = [f'{obj}的附件名称是什么？', f'已读证据中{obj}用哪个附件文件？', f'请从现有原文提取{obj}的附件名。', f'{obj}附件叫什么？不用重新查证已有原文。'][wording]
        quote(f'{obj}附件是 drive-{k}.yaml，仅允许离线写入。')
        card(f'{other}附件', f'{other}使用 unrelated-{k}.json。', .91)
        answer = f'drive-{k}.yaml。'; required = [f'drive-{k}.yaml']; forbidden = ['unrelated']
    elif mode == 3:
        q = [f'{obj}当前压力上限是多少？', f'根据更正记录，{obj}现行压力上限是多少？', f'{obj}的旧上限作废后应采用哪个值？', f'请说明{obj}最新有效的压力限制。'][wording]
        card(f'{obj}旧压力设置', f'旧压力上限是{number+8}kPa。')
        card(f'{obj}同一设置的更正', f'明确更正为{number}kPa，原{number+8}kPa作废。')
        answer = f'摘要更正为{number}kPa，旧值已作废。'; required = [f'{number}kPa']; forbidden = [f'{number+8}kPa']
    elif mode == 4:
        q = [f'{obj}校准系数和偏移量分别是多少？', f'请查{obj}的传感器校准参数：系数及偏移量。', f'我需要{obj}的校准系数与偏移量具体数值。', f'{obj}标定档案里的系数和偏移分别是什么？'][wording]
        flag = {} if repeat%3 == 0 else {'summary_pending': repeat%3 == 1, 'summary_rejected': repeat%3 == 2}
        identifier = card(f'{obj}校准档案', '存档包含传感器参数及校准条件，当前索引未提供具体系数和偏移量。', **flag)
        choose([identifier], q)
    elif mode == 5:
        q = [f'{obj}校准系数是多少？', f'请查{obj}自身的校准系数。', f'我问的是{obj}，它的校准系数具体是多少？', f'需要{obj}的系数，不能用另一台设备的数据。'][wording]
        target = (f'{obj}校准档案', '存有校准系数，具体数值需读取。', .92 if repeat%3 == 0 else .76)
        distractor = (f'{other}校准档案', f'{other}系数为1.8，与{obj}不同。', .86)
        pair = [target, distractor] if repeat%2 else [distractor, target]
        chosen = None
        for value in pair:
            identifier = card(*value)
            if value is target: chosen = identifier
        choose([chosen], q)
    elif mode == 6:
        q = [f'请给出{obj}启动要求的原句。', f'我要核对{obj}启动条件，请读取原文并引用。', f'请找出{obj}启动要求在存档中的逐字表述。', f'摘要不算原句，请查{obj}启动规定的原文。'][wording]
        identifier = card(f'{obj}启动规定', '摘要：先检查防护罩锁闭，再启动。')
        choose([identifier], q)
    elif mode == 7:
        q = [f'{obj}本周运行上限到底是哪个值？', f'两条同周记录不一致，请核对{obj}的运行上限。', f'{obj}本周的上限有矛盾，需要确认更正依据。', f'请查原始记录，确认{obj}本周上限的有效值。'][wording]
        one = card(f'{obj}本周运行限制', f'上限{number}，未注明更正。')
        two = card(f'{obj}同周运行限制', f'上限{number+3}，未注明更正。')
        choose([one, two], q)
    elif mode == 8:
        q = [f'{obj}的校准参数是多少，有什么条件？', f'依据读回的原文，说明{obj}的系数、偏移与适用条件。', f'把{obj}的标定值和必要限制简短告诉我。', f'{obj}的系数与偏移是什么，在哪种模式可用？'][wording]
        coefficient = f'1.{k%800+100:03d}'
        quote(f'{obj}校准系数{coefficient}，偏移量负{number/10:.1f}℃，仅适用于离线模式。')
        reads = 0
        answer = f'系数{coefficient}，偏移量负{number/10:.1f}℃，仅适用于离线模式。'; required = [coefficient, f'{number/10:.1f}', '负', '离线']
    elif mode == 9:
        q = [f'{obj}实际上已经运行了吗？', f'这些记忆能证明{obj}实际启用了吗？', f'问的是实际状态：{obj}有没有投入运行？', f'{obj}是否真的已经启用，而非规定是否允许？'][wording]
        quote(f'{obj}未通过验收前禁止运行。本记录未提供实际启用或停用观测。')
        reads = 0
        answer = '无法确认实际运行状态；记录只说明验收前禁止运行。'; required = ['无法确认', '验收']; forbidden = ['尚未运行', '已经运行', '从未运行']
    elif mode == 10:
        q = [f'{obj}联络人的电话是多少？', f'现有资料能确定{obj}负责人的电话号码吗？', f'请告诉我{obj}维护联系人的手机号。', f'{obj}联系人电话能查到吗？'][wording]
        reads = 0
        if repeat%3 == 0:
            quote(f'{obj}联络人是记录员{number}。片段没有提供电话号码。')
            answer = '已读片段未提供联系电话，当前无法确认。'
        elif repeat%3 == 1:
            last.update(context_kind='segment_navigation_only', error='原文读取失败，未取得片段。')
            card(f'{obj}联络信息', '仅为定位预览，当前未取得联系电话。', summary_pending=True)
            answer = '本次原文读取失败，当前无法确认联系电话。'
        else:
            answer = '当前没有可用的相关记忆索引，无法确认联系电话。'
        required = ['无法确认']; forbidden = ['所有记忆都没有', '全库没有']
    else:
        if repeat%3 == 0:
            q = [f'请从历史中找出{obj}的检修步骤。', f'当前没有索引，请检索{obj}的完整检修流程。', f'帮我定位{obj}检修步骤的历史记录。', f'能检索一下{obj}的检修流程入口吗？'][wording]
            queries = 1; reads = 0; action = 'recall'; arguments = {'question': q}
        elif repeat%3 == 1:
            q = [f'{number}+7等于多少？', f'只计算{number}加7。', f'给出{number}与7的和。', f'算一下{number}+7。'][wording]
            card(f'{other}维护档案', '包含旧维护事项，与当前算术无关。')
            answer = str(number+7); required = [answer]
        else:
            q = [f'{obj}放在哪个架位？', f'已有预览明确写了{obj}位置，它在哪？', f'请指出{obj}的存放架位。', f'{obj}的架位是哪个？'][wording]
            card(f'{obj}存放在B区{number}号架', f'{obj}存放在B区{number}号架。', summary_pending=True, preview_only=True)
            answer = f'预览记载在B区{number}号架。'; required = [f'B区{number}号架']
    expected = {'action': action, 'response': answer, 'tool': '', 'arguments': arguments}
    payload = {'current_request': q, 'goal': '回答当前问题', 'constraints': ['依据所给记忆，不编造事实'],
        'last_result': last, 'tools': {}, 'environment': {}, 'records': [],
        'memory_queries_remaining': queries, 'memory_reads_remaining': reads, 'answer_max_chars': 200}
    system = PRIMARY_SCOPED if split != 'train' or (repeat+mode)%4 in (0, 1) else SHORT
    return {'id': f'{split}-{mode:02d}-{repeat:02d}', 'category': mode, 'split': split,
        'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': dump(payload)},
                     {'role': 'assistant', 'content': dump(expected)}],
        'expected': expected, 'required': required, 'forbidden': forbidden}


def main():
    dest = Path(__file__).parent/'data'; dest.mkdir(exist_ok=True)
    splits = {split: [sample(mode, repeat, split) for repeat in range(n) for mode in range(12)]
              for split, n in [('train', 8), ('dev', 1), ('test', 2)]}
    splits['test'].extend(sample(mode, 2, 'test') for mode in (10, 11))
    random.Random(20261029).shuffle(splits['train'])
    manifest = {'origin': 'New authored synthetic memory policy examples, no user conversations or previous evaluation outputs.',
        'spec_sha256': hashlib.sha256((APP/'prompts/memory_lora_review.txt').read_bytes()).hexdigest(),
        'production_prompt_sha256': hashlib.sha256(PRIMARY_SCOPED.encode()).hexdigest(), 'splits': {}}
    texts = []
    for split, rows in splits.items():
        path = dest/(split+'.jsonl'); path.write_text(''.join(dump(row)+'\n' for row in rows), encoding='utf-8')
        manifest['splits'][split] = {'rows': len(rows), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                   'actions': {a: sum(r['expected']['action']==a for r in rows) for a in ('finish','read_memory','recall')}}
        texts.append({r['messages'][1]['content'] for r in rows})
    assert not texts[0]&texts[1] and not texts[0]&texts[2] and not texts[1]&texts[2]
    (dest/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(dump(manifest))


if __name__ == '__main__': main()
