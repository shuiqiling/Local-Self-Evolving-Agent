"""File/number checks only: no questions, expected answers or model scores."""
from pathlib import Path
import json
import hashlib
import sys
import numpy as np
from safetensors import safe_open
from gguf import GGUFReader

def check(work):
    w=Path(work);cfg=json.loads((w/'adapter/adapter_config.json').read_text())
    assert cfg['r']==8 and set(cfg['target_modules'])=={'q_proj','v_proj'}
    count=0
    with safe_open(w/'adapter/adapter_model.safetensors',framework='np') as file:
        for key in file.keys():
            value=file.get_tensor(key);assert np.issubdtype(value.dtype,np.floating) and np.isfinite(value).all(),key
            count+=1
    reader=GGUFReader(str(w/'adapter-f16.gguf'))
    assert len(reader.tensors)==count==192
    for tensor in reader.tensors:
        assert np.issubdtype(tensor.data.dtype,np.floating) and np.isfinite(tensor.data).all(),tensor.name
    result={'completed':True,'finite_weights':True,'peft_tensors':count,'gguf_tensors':len(reader.tensors),
        'quality_evaluation':False,'scope':'File readability, compatible rank/modules and finite numeric tensors only'}
    (w/'finite-check.json').write_text(json.dumps(result,indent=2));return result

if __name__=='__main__':print(json.dumps(check(sys.argv[1] if len(sys.argv)>1 else Path(__file__).resolve().parent)))
