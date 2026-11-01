"""Batch four routed Q4 expert rows, preserving the original Q8 quantization."""
from pathlib import Path
import difflib,hashlib,json
W=Path(__file__).resolve().parent;assert W.name.startswith('cpu-kernels-30b-')
assert json.loads((W/'moe-pack-report-v2.json').read_text())['completed']
p=W/'source/ggml/src/ggml-cpu/repack.cpp';before=p.read_text();text=before
assert 'sq_moe_batch' not in text
marker='class tensor_traits_base : public ggml::cpu::tensor_traits {'
helper='''#include "pack-q8k.inc"

static bool sq_moe_gemm_enabled() {
    static const bool enabled = [] {
        const char * p = getenv("SQ_MOE_GEMM");
        return !p || strcmp(p, "0") != 0;
    }();
    return enabled;
}

'''
assert text.count(marker)==1;text=text.replace(marker,helper+marker)
old='''template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS, ggml_type PARAM_TYPE> class tensor_traits : public tensor_traits_base {

    bool work_size(int /* n_threads */, const struct ggml_tensor * op, size_t & size) override {'''
new='''template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS, ggml_type PARAM_TYPE> class tensor_traits : public tensor_traits_base {
#if defined(__AVX2__)
    static constexpr bool sq_moe_batch = std::is_same<BLOC_TYPE, block_q4_K>::value &&
        INTER_SIZE == 8 && NB_COLS == 8 && PARAM_TYPE == GGML_TYPE_Q8_K;
#else
    static constexpr bool sq_moe_batch = false;
#endif

    bool work_size(int n_threads, const struct ggml_tensor * op, size_t & size) override {'''
assert text.count(old)==1;text=text.replace(old,new)
old='''                    size += sizeof_mmid_row_mapping*ne02*(ne12 + 1);

                    return true;'''
new='''                    size += sizeof_mmid_row_mapping*ne02*(ne12 + 1);
                    if constexpr (sq_moe_batch) {
                        if (sq_moe_gemm_enabled()) {
                            const size_t packed_bytes = GGML_PAD(4*ggml_row_size(PARAM_TYPE, op->src[0]->ne[0]), 64);
                            const size_t result_bytes = GGML_PAD(4*op->src[0]->ne[1]*sizeof(float), 64);
                            size = GGML_PAD(size, 64) + n_threads*(packed_bytes + result_bytes);
                        }
                    }

                    return true;'''
assert text.count(old)==1;text=text.replace(old,new)
old='''            for (int ir1 = 0; ir1 < nr1; ir1++) {
                struct mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, ir1);'''
new='''            int64_t ir1 = 0;
            if constexpr (sq_moe_batch) {
                if (sq_moe_gemm_enabled() && nr1 >= 4) {
                    // One bounded private slot per worker, reused across experts.
                    const size_t base_size = GGML_PAD(nbw3, sizeof(int64_t)) + n_as*(ne12+1)*sizeof(mmid_row_mapping);
                    const size_t packed_bytes = GGML_PAD(4*nbw1, 64);
                    const size_t result_bytes = GGML_PAD(4*ne01*sizeof(float), 64);
                    const size_t slot_stride = packed_bytes + result_bytes;
                    GGML_ASSERT(params->wsize >= GGML_PAD(base_size,64) + nth*slot_stride);
                    char * slot = wdata + GGML_PAD(base_size,64) + ith*slot_stride;
                    auto * packed = reinterpret_cast<block_q8_Kx4 *>(slot);
                    float * result = reinterpret_cast<float *>(slot + packed_bytes);
                    const int64_t ncols = src0_cur_end - src0_cur_start;
                    for (; ir1 + 4 <= nr1; ir1 += 4) {
                        const block_q8_K * rows[4];
                        mmid_row_mapping mappings[4];
                        for (int r = 0; r < 4; ++r) {
                            mappings[r] = MMID_MATRIX_ROW(cur_a, ir1+r);
                            rows[r] = reinterpret_cast<const block_q8_K *>(wdata +
                                (mappings[r].i1 % ne11)*nbw1 + mappings[r].i2*nbw2);
                        }
                        sq_pack_q8k_4x8(rows, packed, ne00/QK_K);
                        gemm<BLOC_TYPE, INTER_SIZE, NB_COLS, PARAM_TYPE>(ne00, result, ncols,
                            src0_cur + src0_cur_start*nb01, packed, 4, ncols);
                        for (int r = 0; r < 4; ++r) {
                            char * out = reinterpret_cast<char *>(dst->data) +
                                mappings[r].i1*nb1 + mappings[r].i2*nb2 + src0_cur_start*sizeof(float);
                            memcpy(out, result+r*ncols, ncols*sizeof(float));
                        }
                    }
                }
            }
            for (; ir1 < nr1; ir1++) {
                struct mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, ir1);'''
assert text.count(old)==1;text=text.replace(old,new)
text='#include <type_traits>\n'+text
p.write_text(text);(p.parent/'pack-q8k.inc').write_bytes((W/'pack-q8k.inc').read_bytes())
(W/'q4-moe-batched-gemm.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True),text.splitlines(True),fromfile='a/ggml/src/ggml-cpu/repack.cpp',tofile='b/ggml/src/ggml-cpu/repack.cpp')))
(W/'moe-patch-report.json').write_text(json.dumps({'before_sha256':hashlib.sha256(before.encode()).hexdigest(),'after_sha256':hashlib.sha256(p.read_bytes()).hexdigest()},indent=2))
