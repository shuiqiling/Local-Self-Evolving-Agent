from pathlib import Path
import difflib,hashlib,json
W=Path(__file__).resolve().parent
assert W.name.startswith('cpu-kernels-30b-')
p=W/'source/ggml/src/ggml-cpu/arch/x86/quants.c'
before=p.read_text();start=before.index('void ggml_vec_dot_q6_K_q8_K(')
end=before.index('\n  #elif defined __AVX__',start) if '\n  #elif defined __AVX__' in before[start:] else before.index('\n#elif defined __AVX__',start)
body=before[start:end]
old='\n'.join('            __m256i p16_%d = _mm256_maddubs_epi16(q4_%d, q8_%d);'%(i,i,i) for i in range(4))
assert body.count(old)==1
new='''#if defined(__AVXVNNI__)
'''+ '\n'.join('            __m256i p16_%d = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), q4_%d, q8_%d);'%(i,i,i) for i in range(4))+'''
#else
'''+old+'''
#endif'''
body=body.replace(old,new)
old='\n'.join('            p16_%d = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_%d), p16_%d);'%(i,i,i) for i in range(4))
assert body.count(old)==1
# Each subscale repeats over 16 products. Select four copies of each scale
# before sign-extending to i32; this preserves each original reduction lane.
new='''#if defined(__AVXVNNI__)
            const __m128i sq_scale_pack = _mm_setr_epi8(0,0,0,0,8,8,8,8,0,0,0,0,0,0,0,0);
'''+ '\n'.join('            p16_%d = _mm256_mullo_epi32(_mm256_cvtepi8_epi32(_mm_shuffle_epi8(scale_%d, sq_scale_pack)), p16_%d);'%(i,i,i) for i in range(4))+'''
#else
'''+old+'''
#endif'''
body=body.replace(old,new);after=before[:start]+body+before[end:];p.write_text(after)
(W/'q6-dot-avxvnni.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/quants.c',tofile='b/quants.c')))
(W/'q6-patch-report.json').write_text(json.dumps({'before_sha256':hashlib.sha256(before.encode()).hexdigest(),'after_sha256':hashlib.sha256(p.read_bytes()).hexdigest()},indent=2))
