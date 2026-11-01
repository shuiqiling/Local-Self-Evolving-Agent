"""Private four-row kernel retaining the original GEMV FP32 accumulation order."""
from pathlib import Path
import difflib,hashlib,json,re
W=Path(__file__).resolve().parent;assert W.name.startswith('cpu-kernels-30b-')
arch=W/'source/ggml/src/ggml-cpu/arch/x86/repack.cpp';before=arch.read_text()
assert 'sq_q4_moe_gemm_exact' not in before
start=before.index('void ggml_gemm_q4_K_8x8_q8_K(');brace=before.index('{',start);depth=1;end=brace+1
while depth:
 depth+=(before[end]=='{')-(before[end]=='}');end+=1
body=before[start:end]
st=body.rindex('    for (; y < nr / 4; y++) {');brace=body.index('{',st);depth=1;en=brace+1
while depth:
 depth+=(body[en]=='{')-(body[en]=='}');en+=1
loop=body[st:en]
marker='''                // Loop to iterate over the eight sub blocks of a super block - two sub blocks are processed per iteration'''
assert loop.count(marker)==1
loop=loop.replace(marker,'''                __m256i sq_block[4], sq_min_block[4];
                for (int r = 0; r < 4; ++r) {
                    sq_block[r] = _mm256_setzero_si256();
                    sq_min_block[r] = _mm256_setzero_si256();
                }
'''+marker)
float_lines=[]
for prefix,source,target in [('acc_rows','iacc_row_','sq_block'),('acc_min_rows','iacc_row_min_','sq_min_block')]:
 for row in range(4):
  pattern=rf'                    {prefix}\[{row}\] = _mm256_fmadd_ps\([^\n]+;'
  found=re.findall(pattern,loop);assert len(found)==1,(prefix,row,len(found))
  float_lines.append(found[0].replace(source+str(row),target+'['+str(row)+']'))
  loop=loop.replace(found[0],f'                    {target}[{row}] = _mm256_add_epi32({target}[{row}], {source}{row});')
# Keep one FMA per complete 256-element K block, exactly as production GEMV.
old='''                }
            }

            // Store the accumulated values'''
epilogue='''                }
                const __m128 sq_d = _mm_load_ps(a_ptr[b].d);
                const __m256 row_scale_f32 = _mm256_set_m128(sq_d, sq_d);
'''+ '\n'.join(line[4:] for line in float_lines)+'''
            }

            // Store the accumulated values'''
assert loop.count(old)==1;loop=loop.replace(old,epilogue)
# The old per-quarter scale load is no longer used.
loop=loop.replace('''                    const __m128 row_scale_f32_sse = _mm_load_ps(a_ptr[b].d);
                    const __m256 row_scale_f32 = _mm256_set_m128(row_scale_f32_sse, row_scale_f32_sse); //GGML_F32Cx8_REPEAT_LOAD(a_ptrs[rp][b].d, loadMask);''','')
private='''
#if defined(__AVX2__)
extern "C" void sq_q4_moe_gemm_exact(int,float*,size_t,const void*,const void*,int,int);
extern "C" void sq_q4_moe_gemm_exact(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    const int nb = n / QK_K;
    const int64_t b_nb = nb;
    const int64_t xstart = 0;
    int64_t y = 0;
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;
    const block_q4_Kx8 * b_ptr_start = static_cast<const block_q4_Kx8 *>(vx);
    const block_q8_Kx4 * a_ptr_start = static_cast<const block_q8_Kx4 *>(vy);
    const __m256i m4b = _mm256_set1_epi8(0x0f);
    const __m256i requiredOrder = _mm256_set_epi32(3,2,1,0,7,6,5,4);
    assert(n % QK_K == 0 && nr % 4 == 0 && nc % 8 == 0);
'''+loop+'''
}
#endif
'''
after=before[:end]+private+before[end:];arch.write_text(after)
(W/'q4-moe-exact-gemm.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/arch/x86/repack.cpp',tofile='b/arch/x86/repack.cpp')))
generic=W/'source/ggml/src/ggml-cpu/repack.cpp';old=generic.read_text();text=old
marker='class tensor_traits_base : public ggml::cpu::tensor_traits {'
text=text.replace(marker,'extern "C" void sq_q4_moe_gemm_exact(int,float*,size_t,const void*,const void*,int,int);\n\n'+marker)
old_call='''                        gemm<BLOC_TYPE, INTER_SIZE, NB_COLS, PARAM_TYPE>(ne00, result, ncols,
                            src0_cur + src0_cur_start*nb01, packed, 4, ncols);'''
new_call='''                        sq_q4_moe_gemm_exact(ne00, result, ncols,
                            src0_cur + src0_cur_start*nb01, packed, 4, ncols);'''
assert text.count(old_call)==1;text=text.replace(old_call,new_call);generic.write_text(text)
(W/'q4-moe-exact-dispatch.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),text.splitlines(True),fromfile='a/repack.cpp',tofile='b/repack.cpp')))
(W/'exact-patch-report.json').write_text(json.dumps({'arch_before':hashlib.sha256(before.encode()).hexdigest(),'arch_after':hashlib.sha256(arch.read_bytes()).hexdigest(),
 'generic_after':hashlib.sha256(generic.read_bytes()).hexdigest(),'note':'New private GEMM only. Existing dense GEMM unchanged; integer sums combine all subblocks before one FP32 FMA, matching GEMV.'},indent=2))
