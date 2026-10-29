"""Pinned, guarded Q4_K AVX-VNNI experiment. Apply only to isolated exported source."""
import argparse,difflib,hashlib,json,re
from pathlib import Path

REPACK_HELPERS=r'''
// Experimental AVX-VNNI Q4_K dot accumulation. Adjacent 16-bit scales
// are equal, so reducing four u4*s8 products into i32 before multiplying
// by that scale preserves the integer result. u4 in [0,15] ensures the
// legacy pair sums and 16-product i16 accumulators cannot saturate/overflow.
#if defined(__AVX2__)
static inline __m256i lea_q4k_acc(__m256i acc, __m256i u4, __m256i s8) {
#if defined(__AVXVNNI__)
    return _mm256_dpbusd_avx_epi32(acc, u4, s8);
#else
    return _mm256_add_epi16(acc, _mm256_maddubs_epi16(u4, s8));
#endif
}
static inline __m256i lea_q4k_scale(__m256i acc, __m256i scale_pairs) {
#if defined(__AVXVNNI__)
    return _mm256_mullo_epi32(acc, _mm256_srli_epi32(scale_pairs, 16));
#else
    return _mm256_madd_epi16(acc, scale_pairs);
#endif
}
static inline __m256i lea_q4k_join(__m256i a, __m256i b) {
#if defined(__AVXVNNI__)
    return _mm256_add_epi32(a, b);
#else
    return _mm256_add_epi16(a, b);
#endif
}
#endif
'''
DOT_HELPER=r'''
#if defined(__AVX2__)
static inline __m256i lea_q4k_dot_scale(__m256i u4, __m256i s8, __m256i scale_pairs) {
#if defined(__AVXVNNI__)
    const __m256i sum4 = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), u4, s8);
    return _mm256_mullo_epi32(sum4, _mm256_srli_epi32(scale_pairs, 16));
#else
    return _mm256_madd_epi16(scale_pairs, _mm256_maddubs_epi16(u4, s8));
#endif
}
#endif
'''

def region(text,name):
    start=text.index('void '+name+'(')
    end=text.find('\nvoid ',start+5)
    assert end>start
    return start,end,text[start:end]

def patch_repack(text):
    details={}
    for name in ['ggml_gemv_q4_K_8x8_q8_K','ggml_gemm_q4_K_8x8_q8_K']:
        start,end,body=region(text,name)
        pattern=r'(\b\w+) = _mm256_add_epi16\(\1, _mm256_maddubs_epi16\(([^\n]+)\)\);'
        body,acc=re.subn(pattern,lambda m:f'{m[1]} = lea_q4k_acc({m[1]}, {m[2]});',body)
        if name.startswith('ggml_gemm'):
            trees=0
            def tree(m):
                nonlocal trees,acc
                pairs=re.findall(r'_mm256_maddubs_epi16\((\w+), (\w+)\)',m[2])
                assert len(pairs)==4
                value='_mm256_setzero_si256()'
                for a,b in pairs:value=f'lea_q4k_acc({value}, {a}, {b})'
                trees+=1;acc+=4
                return f'__m256i {m[1]} = {value};'
            body=re.sub(r'__m256i (iacc_mat_\w+_sp[12]) = (_mm256_add_epi16[^\n]+);',tree,body)
            body,joins=re.subn(r'(__m256i iacc_mat_\w+ = )_mm256_add_epi16\((iacc_mat_\w+_sp1), (iacc_mat_\w+_sp2)\);',
                lambda m:f'{m[1]}lea_q4k_join({m[2]}, {m[3]});',body)
            assert trees==32 and joins==16,(trees,joins)
        pattern=r'(\b\w+) = _mm256_madd_epi16\(\1, (\w+)\);'
        body,scale=re.subn(pattern,lambda m:f'{m[1]} = lea_q4k_scale({m[1]}, {m[2]});',body)
        assert acc>0 and scale>0 and acc==scale*8,(name,acc,scale)
        details[name]={'accumulation_sites':acc,'scale_sites':scale}
        text=text[:start]+body+text[end:]
    where=text.index('void ggml_gemv_q4_K_8x8_q8_K(')
    return text[:where]+REPACK_HELPERS+'\n'+text[where:],details

def patch_dot(text):
    start,end,body=region(text,'ggml_vec_dot_q4_K_q8_K')
    for suffix in ['l','h']:
        old=f'__m256i p16{suffix} = _mm256_maddubs_epi16(q4{suffix}, q8{suffix});\n            p16{suffix} = _mm256_madd_epi16(scale_{suffix}, p16{suffix});'
        assert body.count(old)==1
        body=body.replace(old,f'__m256i p16{suffix} = lea_q4k_dot_scale(q4{suffix}, q8{suffix}, scale_{suffix});')
    return text[:start]+DOT_HELPER+'\n'+body+text[end:]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True,type=Path);parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args();source=args.source.resolve();out=args.out.resolve()
    assert '/experiments/q4-vnni-20261029/' in str(source),'Only isolated experiment source may be patched'
    # Profiling observed no ordinary Q4_K dot calls. Limit this candidate to
    # the actually selected repack GEMV/GEMM paths; leave quants.c untouched.
    files=[('ggml/src/ggml-cpu/arch/x86/repack.cpp','c69735f83caf854172cac78eed2cbc022ef81c077e5f6aca55a49d3404a02331',patch_repack)]
    changes=[];report={}
    for rel,expected,patch in files:
        p=source/rel;raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==expected,rel
        before=raw.decode();result=patch(before)
        if isinstance(result,tuple):after,details=result
        else:after=result;details={'replaced_scaled_dot_sites':2}
        assert before!=after
        changes.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/'+rel,tofile='b/'+rel))
        p.write_text(after);report[rel]={'before_sha256':expected,'after_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),**details}
    (out/'q4-vnni.patch').write_text(''.join(changes));(out/'patch-report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
