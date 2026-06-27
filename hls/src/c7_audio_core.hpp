// c7_audio_core.hpp — the C7 audio mask network as a reusable core (no AXI interface).
// Shared by the standalone IP (c7_audio_top) and the monolithic AVSE IP (c7_avse_top), so the audio
// compute lives in exactly one place. video_embed is a flat [B*T_LAT] row-major signal that the
// caller supplies (from m_axi in the standalone, or from the on-chip video encoder in the monolith).
//
// REAL trained weights from c7_weights.hpp (p2-c7-full/best.pt, BN-folded). This is the value-faithful
// deployment that matches the fixed-point emulator tools/c7_fixedpoint.py bit-for-bit (see DEPLOY_PLAN.md):
//  - in_norm / bn1 / bn2 are inline per-channel affines (s,b) in bn_t, applied to the FULL-precision
//    pre-activation then cast once to data_t (PReLU done in acc_t so there is no extra quantization).
//  - mask uses hardsigmoid (the HW choice); decoder uses the correct ConvTranspose(pad) offset.
#ifndef C7_AUDIO_CORE_HPP
#define C7_AUDIO_CORE_HPP

#include "c7_types.hpp"
#include "c7_weights.hpp"

namespace c7 {

static data_t hsig(acc_t x) {            // hardsigmoid ~ clamp(0.2x+0.5, 0, 1)
#pragma HLS INLINE
    acc_t v = (acc_t)0.2 * x + (acc_t)0.5;
    if (v < 0) v = 0; else if (v > 1) v = 1;
    return (data_t)v;
}

// audio_in[T], video_embed[B*T_LAT] (row-major [B][T_LAT]), audio_out[T].
static void audio_core(const sample_t *audio_in,
                       const data_t   *video_embed,
                       sample_t       *audio_out)
{
    using namespace cfg;

    static data_t w [N][T_LAT];     // encoded latent; reused to hold w*mask
    static data_t y [B][T_LAT];     // TCN state (in-place residual)
    static data_t h [H][T_LAT];     // in_conv output (post bn1)
    static data_t hd[H][T_LAT];     // dwconv output (post bn2)
#pragma HLS ARRAY_PARTITION variable=w  dim=1 cyclic factor=2
#pragma HLS ARRAY_PARTITION variable=y  dim=1 cyclic factor=2
#pragma HLS ARRAY_PARTITION variable=h  dim=1 cyclic factor=2
#pragma HLS ARRAY_PARTITION variable=hd dim=1 cyclic factor=2

    // encoder: Conv1d(1->N, k=L, stride=STRIDE, pad=STRIDE)
    ENC: for (int n = 0; n < N; n++)
        for (int t = 0; t < T_LAT; t++) {
#pragma HLS PIPELINE II=2
            acc_t a = 0;
            for (int k = 0; k < L; k++) {
                int s = t * STRIDE + k - STRIDE;
                sample_t x = (s >= 0 && s < T) ? audio_in[s] : (sample_t)0;
                a += (acc_t)(x * wts::Wenc[n][k]);
            }
            w[n][t] = (data_t)a;
        }

    // bottleneck: y = in_norm(w) -> 1x1(N->B) + video.  in_norm inline (wn cast to data_t).
    BOT: for (int b = 0; b < B; b++)
        for (int t = 0; t < T_LAT; t++) {
#pragma HLS PIPELINE II=2
            acc_t a = (acc_t)video_embed[b * T_LAT + t];
            for (int n = 0; n < N; n++) {
                data_t wn = (data_t)(wts::innorm_s[n] * w[n][t] + wts::innorm_b[n]);
                a += (acc_t)(wn * wts::Wbn[n][b]);
            }
            y[b][t] = (data_t)a;
        }

    // 10 dilated dwsep TCN blocks (residual). PReLU in acc_t, then bn affine, then one data_t cast.
    BLOCKS: for (int blk = 0; blk < NBLK; blk++) {
        int dil = 1 << (blk % X);
        IN1x1: for (int hh = 0; hh < H; hh++)
            for (int t = 0; t < T_LAT; t++) {
#pragma HLS PIPELINE II=2
                acc_t a = 0;
                for (int b = 0; b < B; b++) a += (acc_t)(y[b][t] * wts::Win[blk][b][hh]);
                acc_t p = (a >= 0) ? a : (acc_t)(wts::pr1[blk][hh] * a);
                h[hh][t] = (data_t)(wts::bn1_s[blk][hh] * p + wts::bn1_b[blk][hh]);
            }
        DW: for (int hh = 0; hh < H; hh++)
            for (int t = 0; t < T_LAT; t++) {
#pragma HLS PIPELINE II=2
                acc_t a = 0;
                for (int j = 0; j < KD; j++) {
                    int tt = t - (KD - 1 - j) * dil;
                    data_t hv = (tt >= 0) ? h[hh][tt] : (data_t)0;
                    a += (acc_t)(hv * wts::Wdw[blk][hh][j]);
                }
                acc_t p = (a >= 0) ? a : (acc_t)(wts::pr2[blk][hh] * a);
                hd[hh][t] = (data_t)(wts::bn2_s[blk][hh] * p + wts::bn2_b[blk][hh]);
            }
        OUT1x1: for (int b = 0; b < B; b++)
            for (int t = 0; t < T_LAT; t++) {
#pragma HLS PIPELINE II=2
                acc_t a = 0;
                for (int hh = 0; hh < H; hh++) a += (acc_t)(hd[hh][t] * wts::Wout[blk][hh][b]);
                y[b][t] = (data_t)((acc_t)y[b][t] + a);
            }
    }

    // mask + apply (overwrite w with w*mask)
    MASK: for (int n = 0; n < N; n++)
        for (int t = 0; t < T_LAT; t++) {
#pragma HLS PIPELINE II=2
            acc_t a = 0;
            for (int b = 0; b < B; b++) a += (acc_t)(y[b][t] * wts::Wmask[b][n]);
            w[n][t] = (data_t)(w[n][t] * hsig(a));
        }

    // decoder: ConvTranspose1d(N->1, k=L, stride=STRIDE, pad=STRIDE), accumulate. s = t*STRIDE+k-STRIDE.
    static acc_t obuf[T];
    INIT_O: for (int s = 0; s < T; s++) obuf[s] = 0;
    DEC: for (int n = 0; n < N; n++)
        for (int t = 0; t < T_LAT; t++) {
#pragma HLS PIPELINE II=2
            data_t wv = w[n][t];
            for (int k = 0; k < L; k++) {
                int s = t * STRIDE + k - STRIDE;
                if (s >= 0 && s < T) obuf[s] += (acc_t)(wv * wts::Wdec[n][k]);
            }
        }
    CASTO: for (int s = 0; s < T; s++) {
#pragma HLS PIPELINE II=1
        audio_out[s] = (sample_t)obuf[s];
    }
}

}  // namespace c7

#endif  // C7_AUDIO_CORE_HPP
