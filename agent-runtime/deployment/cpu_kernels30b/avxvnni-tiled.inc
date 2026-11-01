// Private AVX-VNNI experiment. Keep the AVX2 path selectable for comparison.
#if defined(__AVXVNNI__) && !defined(__AVX512VNNI__)
static bool sq_use_avxvnni_tiled() {
    static const bool enabled = [] {
        const char * p = std::getenv("SQ_AVXVNNI_TILED");
        return !p || std::strcmp(p, "0") != 0;
    }();
    return enabled;
}

// Transpose eight activation rows in groups of eight four-byte words.
// Each load in the MAC then supplies eight output columns with no gather.
static void sq_repack_src1(tiled_tile_src1 * src1, int row0, int num_k, bool bias) {
    const int stride = num_k * TILED_TILE_K;
    uint8_t * base = reinterpret_cast<uint8_t *>(src1->q) + row0 * stride;
    const __m256i flip = _mm256_set1_epi8(bias ? (int8_t) 0x80 : 0);
    for (int group = 0; group < TILED_MICRO; group += 8) {
        uint8_t * band = base + group * stride;
        for (int off = 0; off < stride; off += 32) {
            __m256i v[8], t[8], u[8];
            for (int r = 0; r < 8; ++r) {
                v[r] = _mm256_xor_si256(_mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(band + r * stride + off)), flip);
            }
            for (int r = 0; r < 8; r += 2) {
                t[r] = _mm256_unpacklo_epi32(v[r], v[r+1]);
                t[r+1] = _mm256_unpackhi_epi32(v[r], v[r+1]);
            }
            for (int r = 0; r < 8; r += 4) {
                u[r] = _mm256_unpacklo_epi64(t[r], t[r+2]);
                u[r+1] = _mm256_unpackhi_epi64(t[r], t[r+2]);
                u[r+2] = _mm256_unpacklo_epi64(t[r+1], t[r+3]);
                u[r+3] = _mm256_unpackhi_epi64(t[r+1], t[r+3]);
            }
            for (int r = 0; r < 4; ++r) {
                v[r] = _mm256_permute2x128_si256(u[r], u[r+4], 0x20);
                v[r+4] = _mm256_permute2x128_si256(u[r], u[r+4], 0x31);
            }
            for (int r = 0; r < 8; ++r) {
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(band + r * stride + off), v[r]);
            }
        }
    }
}

template <int SUBBLK, bool HAS_MIN, int BIAS, int NK, bool ACTBIAS>
static void sq_micro_avxvnni(const tiled_tile_src0 & src0, const tiled_tile_src1 & src1,
                            int i0, int j0, int num_k, int slab, float * buf, int buf_stride) {
    constexpr int NB = TILED_TILE_K / SUBBLK;
    constexpr int NS = SUBBLK / 16;
    constexpr int NG = SUBBLK / 4;
    const int nkr = NK > 0 ? NK : num_k;
    const int seff = NK > 0 ? 0 : slab;
    const int q_stride = nkr * TILED_TILE_K;
    const int q_off = seff * TILED_TILE_K;
    const int nb_stride = NB * nkr;
    const int nb_off = seff * NB;
    const int bs_stride = nkr == 1 ? TILED_TILE_ROWS : TILED_MICRO;
    const int bs_off = seff * TILED_MICRO;
    const __m256i zero = _mm256_setzero_si256();
    const __m256i bias = _mm256_set1_epi8((int8_t) BIAS);
    for (int band = 0; band < TILED_MICRO; band += 4) {
        for (int col = 0; col < TILED_MICRO; col += 8) {
            __m256i sum[4] = {zero, zero, zero, zero};
            __m256i mins[4] = {zero, zero, zero, zero};
            for (int s = 0; s < NB; ++s) {
                __m256i raw[4] = {zero, zero, zero, zero};
                for (int g = 0; g < NG; ++g) {
                    const int kg = s * NG + g;
                    const int8_t * p = src1.q + (j0+col) * q_stride + q_off
                                      + (kg % 8) * q_stride + (kg / 8) * 32;
                    const __m256i act = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(p));
                    #pragma GCC unroll 4
                    for (int r = 0; r < 4; ++r) {
                        uint32_t word;
                        std::memcpy(&word, src0.q + (i0+band+r)*q_stride + q_off + kg*4, 4);
                        const __m256i w = _mm256_set1_epi32((int32_t) word);
                        if constexpr (BIAS != 0 && ACTBIAS) {
                            raw[r] = _mm256_dpbusd_avx_epi32(raw[r], act, _mm256_sub_epi8(w, bias));
                        } else if constexpr (BIAS != 0) {
                            // Preserve the legacy signed-byte sign convention, including -128.
                            const __m256i sw = _mm256_sub_epi8(w, bias);
                            raw[r] = _mm256_dpbusd_avx_epi32(raw[r], _mm256_abs_epi8(sw),
                                                            _mm256_sign_epi8(act, sw));
                        } else {
                            raw[r] = _mm256_dpbusd_avx_epi32(raw[r], w, act);
                        }
                    }
                }
                __m256i bsum = zero;
                if constexpr (HAS_MIN) {
                    for (int u = 0; u < NS; ++u) {
                        bsum = _mm256_add_epi32(bsum, _mm256_loadu_si256(
                            reinterpret_cast<const __m256i *>(src1.bsums +
                                (bs_off+s*NS+u)*bs_stride + j0+col)));
                    }
                }
                #pragma GCC unroll 4
                for (int r = 0; r < 4; ++r) {
                    const int a = (i0+band+r)*nb_stride + nb_off+s;
                    sum[r] = _mm256_add_epi32(sum[r], _mm256_mullo_epi32(
                        raw[r], _mm256_set1_epi32(src0.scales[a])));
                    if constexpr (HAS_MIN) {
                        mins[r] = _mm256_add_epi32(mins[r], _mm256_mullo_epi32(
                            bsum, _mm256_set1_epi32(src0.mins[a])));
                    }
                }
            }
            const __m256 d1 = _mm256_loadu_ps(src1.d + seff*TILED_MICRO + j0+col);
            #pragma GCC unroll 4
            for (int r = 0; r < 4; ++r) {
                const int row = i0+band+r;
                const int d_off = seff*TILED_MICRO + row;
                if constexpr (BIAS != 0 && ACTBIAS) {
                    sum[r] = _mm256_sub_epi32(sum[r], _mm256_set1_epi32(src0.mins[row*nb_stride+nb_off]));
                }
                __m256 result = _mm256_mul_ps(_mm256_set1_ps(src0.d[d_off]), _mm256_cvtepi32_ps(sum[r]));
                if constexpr (HAS_MIN) {
                    result = _mm256_fnmadd_ps(_mm256_set1_ps(src0.dmin[d_off]), _mm256_cvtepi32_ps(mins[r]), result);
                }
                float * out = buf + row*buf_stride + j0+col;
                _mm256_storeu_ps(out, _mm256_fmadd_ps(result, d1, _mm256_loadu_ps(out)));
            }
        }
    }
}
#endif
