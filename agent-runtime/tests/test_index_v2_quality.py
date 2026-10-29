import importlib.util
from pathlib import Path

path=Path(__file__).parents[1]/'deployment/index_lora_v2/quality.py'
spec=importlib.util.spec_from_file_location('index_v2_quality',path)
quality=importlib.util.module_from_spec(spec);spec.loader.exec_module(quality)


def test_whole_identifiers_and_numbers_cannot_pass_as_prefixes():
    source='设备HG-20510，业务端口21407，诊断端口33791。'
    bad=quality.unsupported(source,'设备HG-205，业务端口214，诊断端口337910。')
    assert bad['identifiers']==['HG-205']
    assert set(bad['numbers'])=={'214','337910'}


def test_equivalent_negative_forms_and_units_are_not_fabrications():
    source='HG-20510温度偏移为负0.70℃，电源19V，保险丝4A。'
    assert quality.unsupported(source,'HG-20510：偏移-0.7℃，电源19V/4A。')=={'identifiers':[],'numbers':[]}
    assert quality.unsupported(source,'偏移正0.7℃。')['numbers']==['正0.7']


def test_presence_is_not_claimed_to_prove_field_binding_or_sentence_completeness():
    source='HG-20510业务端口21407，诊断端口33791。'
    assert not quality.unsupported(source,'业务端口33791，诊断端口21407。')['numbers']
    # Field swapping and semantic truncation still require source review.
