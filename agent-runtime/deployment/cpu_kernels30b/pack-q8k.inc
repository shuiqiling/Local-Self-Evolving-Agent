// Re-layout existing Q8_K codes only: do not quantize the activations again.
static void sq_pack_q8k_4x8(const block_q8_K * const rows[4], block_q8_Kx4 * out, int blocks) {
    for (int b = 0; b < blocks; ++b) {
        for (int r = 0; r < 4; ++r) {
            out[b].d[r] = rows[r][b].d;
            for (int k = 0; k < 32; ++k) {
                memcpy(out[b].qs + k*32 + r*8, rows[r][b].qs + k*8, 8);
            }
            for (int k = 0; k < 16; ++k) {
                out[b].bsums[(k/4)*16 + r*4 + k%4] = rows[r][b].bsums[k];
            }
        }
    }
}
