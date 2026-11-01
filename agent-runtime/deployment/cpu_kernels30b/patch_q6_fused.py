"""Fuse the signed i16 multiply/add rather than widening Q6 scales to i32."""
from pathlib import Path
import difflib,hashlib,json
W=Path(__file__).resolve().parent;assert W.name.startswith('cpu-kernels-30b-')
p=W/'source/ggml/src/ggml-cpu/arch/x86/quants.c'
original=W.parent/'q4-vnni-20261029/source/ggml/src/ggml-cpu/arch/x86/quants.c'
before=original.read_text();start=before.index('void ggml_vec_dot_q6_K_q8_K(')
end=before.index('\n#elif defined __AVX__',start);body=before[start:end]
old='        __m256i sumi = _mm256_setzero_si256();'
assert body.count(old)==1
body=body.replace(old,'''#if defined(__AVXVNNI__)
        __m256i sq_sum0 = _mm256_setzero_si256(), sq_sum1 = _mm256_setzero_si256();
        __m256i sq_sum2 = _mm256_setzero_si256(), sq_sum3 = _mm256_setzero_si256();
#else
'''+old+'''
#endif''')
old='\n'.join('            p16_%d = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_%d), p16_%d);'%(i,i,i) for i in range(4))
old+='''

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_2, p16_3));'''
assert body.count(old)==1
new='''#if defined(__AVXVNNI__)
'''+ '\n'.join('            sq_sum%d = _mm256_dpwssd_avx_epi32(sq_sum%d, _mm256_cvtepi8_epi16(scale_%d), p16_%d);'%(i,i,i,i) for i in range(4))+'''
#else
'''+old+'''
#endif'''
body=body.replace(old,new)
old='        sumi = _mm256_sub_epi32(sumi, q8sclsub);';assert body.count(old)==1
body=body.replace(old,'''#if defined(__AVXVNNI__)
        __m256i sumi = _mm256_add_epi32(_mm256_add_epi32(sq_sum0, sq_sum1), _mm256_add_epi32(sq_sum2, sq_sum3));
#endif
'''+old)
after=before[:start]+body+before[end:];p.write_text(after)
(W/'q6-dpwssd-avxvnni.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/quants.c',tofile='b/quants.c')))
(W/'q6-fused-patch-report.json').write_text(json.dumps({'before_sha256':hashlib.sha256(before.encode()).hexdigest(),'after_sha256':hashlib.sha256(p.read_bytes()).hexdigest()},indent=2))
