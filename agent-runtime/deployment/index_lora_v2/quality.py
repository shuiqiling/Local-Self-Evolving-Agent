"""Literal checks with whole identifiers and signed-number equivalence, not semantic proof."""
from decimal import Decimal
import re

IDENTIFIER=re.compile(r'(?<![A-Za-z0-9_.-])[A-Za-z]{2,}(?:[-_][A-Za-z0-9]+)+(?:\.[A-Za-z0-9]+)*(?![A-Za-z0-9_.-])')
NUMBER=re.compile(r'(?P<sign>负|正|[+\-−])?(?P<value>\d+(?:\.\d+)?)')


def atoms(text):
    identifiers=[m.group() for m in IDENTIFIER.finditer(text)]
    masked=list(text)
    for m in IDENTIFIER.finditer(text):masked[m.start():m.end()]=[' ']*(m.end()-m.start())
    numbers=[]
    for m in NUMBER.finditer(''.join(masked)):
        sign='-' if m.group('sign') in ('负','-','−') else ''
        numbers.append((m.group(),str(Decimal(sign+m.group('value')).normalize())))
    return set(identifiers),numbers


def unsupported(source,summary):
    source_ids,source_numbers=atoms(source);ids,numbers=atoms(summary)
    values={v for _,v in source_numbers}
    return {'identifiers':sorted(ids-source_ids),'numbers':[text for text,value in numbers if value not in values]}


def missing_key_values(summary,bindings):
    values={v for _,v in atoms(summary)[1]}
    return [b for b in bindings if str(Decimal(b['value']).normalize()) not in values]


def check(row,summary):
    unknown=unsupported(row['source']['archive_text'],summary)
    missing_critical=[x for x in row['critical_fragments'] if x not in summary]
    missing_topics=[g for g in row['coverage_groups'] if not any(x in summary for x in g)]
    key_missing=missing_key_values(summary,row['key_bindings'])
    return {'identity':all(x in summary for x in row['required_identifiers']+row['required_names']),
            'literal_supported':not unknown['identifiers'] and not unknown['numbers'],
            'critical_exact':not missing_critical,'topic_exact':not missing_topics,
            'key_values_present':not key_missing,'normal_end':summary.endswith(('。','！','？','.','!','?')),
            'at_character_cap':len(summary)==100,'chars':len(summary),
            'unsupported':unknown,'missing_critical':missing_critical,'missing_topics':missing_topics,
            'missing_key_values':key_missing}
