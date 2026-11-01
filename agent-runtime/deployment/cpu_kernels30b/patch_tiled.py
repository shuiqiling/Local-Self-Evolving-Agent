"""Apply only to the isolated private 30B experiment, retaining the old path."""
from pathlib import Path
import difflib, hashlib, json

W = Path(__file__).resolve().parent
assert W.name.startswith('cpu-kernels-30b-'), 'Isolated experiment only'
p = W/'source/ggml/src/ggml-cpu/tiled/tiled-kernel.cpp'
before = p.read_text()
assert 'sq_micro_avxvnni' not in before
text = before.replace('#include <cstring>', '#include <cstring>\n#include <cstdlib>\n#include "avxvnni-tiled.inc"')
old = '''#elif defined(__AVX2__)
    if (standard) {'''
new = '''#elif defined(__AVX2__)
#if defined(__AVXVNNI__)
    if (sq_use_avxvnni_tiled()) {
        if (standard) {
            sq_micro_avxvnni<SUBBLK, HAS_MIN, BIAS, 1, ACTBIAS>(src0, src1, i0, j0, num_k, slab, buf, buf_stride);
        } else {
            sq_micro_avxvnni<SUBBLK, HAS_MIN, BIAS, 0, ACTBIAS>(src0, src1, i0, j0, num_k, slab, buf, buf_stride);
        }
        return;
    }
#endif
    if (standard) {'''
assert text.count(old) == 1
text = text.replace(old, new)
old = '''void tiled_repack_src1(tiled_tile_src1 * src1, int row0, int num_k, bool bias) {
    if (!bias) {'''
new = '''void tiled_repack_src1(tiled_tile_src1 * src1, int row0, int num_k, bool bias) {
#if defined(__AVXVNNI__) && !defined(__AVX512VNNI__)
    if (sq_use_avxvnni_tiled()) {
        sq_repack_src1(src1, row0, num_k, bias);
        return;
    }
#endif
    if (!bias) {'''
assert text.count(old) == 1
text = text.replace(old, new)
# Include after intrinsics have been declared.
text = text.replace('#include "avxvnni-tiled.inc"', '')
marker = '// Reference implementation, slower than existing vec_dot approach'
text = text.replace(marker, '#include "avxvnni-tiled.inc"\n\n'+marker)
inc = p.parent/'avxvnni-tiled.inc'
inc.write_bytes((W/'avxvnni-tiled.inc').read_bytes())
p.write_text(text)
(W/'tiled-avxvnni.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True), text.splitlines(True),
    fromfile='a/ggml/src/ggml-cpu/tiled/tiled-kernel.cpp', tofile='b/ggml/src/ggml-cpu/tiled/tiled-kernel.cpp')))
(W/'patch-report.json').write_text(json.dumps({'before_sha256': hashlib.sha256(before.encode()).hexdigest(),
    'after_sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'include_sha256': hashlib.sha256(inc.read_bytes()).hexdigest()}, indent=2))
