"""Isolated CPU reference of the official Fast KVzip Weight.forward.

No inference-model replacement. Receives the actual post-attention-norm hidden
states from the pinned native Q4+LoRA process over a local Unix socket.
FP32 gate arithmetic is intentional on this AVX2 CPU; not a claim of bitwise
equivalence to the author's CUDA/BF16 arithmetic.
"""
import ast, json, math, os, socket, struct, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

work = Path(__file__).resolve().parent
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
# The author's training checkpoint also contains numpy RNG state. Allow only
# these concrete numpy data constructors; never fall back to weights_only=False.
allowed = [np.core.multiarray._reconstruct,
           (np.core.multiarray._reconstruct,'numpy.core.multiarray._reconstruct'),
           np.ndarray, np.dtype,
           *[type(np.dtype(k)) for k in ['uint32','uint64','int64','float32','float64']]]
with torch.serialization.safe_globals(allowed):
    checkpoint = torch.load(work/'gate.pt', map_location='cpu', weights_only=True)
weights = checkpoint['module']
assert len(weights) == 36
expected = {'q_proj.weight','q_proj.bias','k_proj.weight','q_norm.weight',
            'k_norm.weight','k_base','b'}
for w in weights:
    assert set(w) == expected
    assert w['q_proj.weight'].shape == (512,2560)
    assert w['k_proj.weight'].shape == (128,2560)
    assert all(torch.isfinite(t).all() for t in w.values())
weights = [{k:v.float().contiguous() for k,v in w.items()} for w in weights]
del checkpoint

def rms(x, w):
    return x * torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6) * w

@torch.inference_mode()
def gate(layer, x):
    w=weights[layer]; n=x.shape[0]
    q=rms(F.linear(x,w['q_proj.weight'],w['q_proj.bias']).view(n,8,4,16),w['q_norm.weight'])
    k=rms(F.linear(x,w['k_proj.weight']).view(n,8,1,16),w['k_norm.weight'])
    q=q.transpose(0,1).transpose(-1,-2);k=k.transpose(0,1)
    logit=torch.matmul(k,q)/4 + w['b'].unsqueeze(2)
    base=torch.matmul(w['k_base'],q)/4
    return (1/(1+torch.exp(base-logit).sum(2,keepdim=True))).mean(-1).squeeze(-1).contiguous()

# Compare the CPU port against the pinned author's actual Weight class, using
# the real first-layer gate weights. Only replace its unavailable HF RMSNorm
# dependency with the equivalent FP32 operation, not its gate forward method.
class Norm(torch.nn.Module):
    def __init__(self,dim):
        super().__init__();self.weight=torch.nn.Parameter(torch.ones(dim))
    def forward(self,x):return rms(x,self.weight)
tree=ast.parse((work/'official_gate.py').read_text())
klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Weight')
namespace={'torch':torch,'nn':torch.nn,'math':math,'Qwen3RMSNorm':Norm}
exec(compile(ast.Module(body=[klass],type_ignores=[]),'pinned-official-Weight','exec'),namespace)
reference=namespace['Weight'](0,2560,16,8,4,torch.float32,sink=16)
reference.load_state_dict(weights[0]);torch.manual_seed(20261029)
test_input=torch.randn(32,2560)
with torch.inference_mode():
    difference=(reference(test_input).squeeze(0)-gate(0,test_input)).abs().max().item()
assert difference<=1e-6,('CPU gate differs from official formula',difference)

def recvall(conn,n):
    parts=[]
    while n:
        b=conn.recv(n)
        if not b:raise EOFError('native gate connection closed')
        parts.append(b);n-=len(b)
    return b''.join(parts)

sockpath=work/'gate.sock'
assert not sockpath.exists(), 'Refuse to replace an existing gate socket'
server=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);server.bind(str(sockpath));server.listen(1)
(work/'gate-ready.json').write_text(json.dumps({'pid':os.getpid(),'torch':torch.__version__,
    'threads':1,'layers':36,'arithmetic':'FP32','weights_only':True,
    'official_formula_fp32_max_diff':difference}))
rows=[]
try:
    conn,_=server.accept()
    with conn:
        while True:
            layer,n,width=struct.unpack('<iii',recvall(conn,12))
            if layer == -1:break
            assert 0<=layer<36 and 0<n<=512 and width==2560
            x=torch.from_numpy(np.frombuffer(recvall(conn,n*width*4),dtype='<f4').copy().reshape(n,width))
            start=time.perf_counter();score=gate(layer,x);elapsed=time.perf_counter()-start
            assert score.shape==(8,n) and torch.isfinite(score).all()
            conn.sendall(score.numpy().astype('<f4',copy=False).tobytes())
            rows.append({'layer':layer,'tokens':n,'math_ms':elapsed*1000})
finally:
    server.close()
    (work/'gate-timings.json').write_text(json.dumps({'calls':rows,'total_math_ms':sum(r['math_ms'] for r in rows)},indent=2))
