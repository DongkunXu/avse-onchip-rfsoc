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
    static wgt_t  wdl[N][L];        // O-3b: Wdec staged + partitioned on n for the gather decoder
    static sample_t abuf[T];        // O-3c: cache audio_in on-chip (read once vs 128x DDR), taps in parallel
// O-3: partition the channel dim (the reduction axis) cyclic-16 so the 1x1 channel reductions read
// 16/cycle with the reduction unrolled. w also feeds BOT (II=8) and the gather decoder (II=16).
#pragma HLS ARRAY_PARTITION variable=w   dim=1 cyclic factor=16
#pragma HLS ARRAY_PARTITION variable=y   dim=1 cyclic factor=16
#pragma HLS ARRAY_PARTITION variable=h   dim=1 cyclic factor=16
#pragma HLS ARRAY_PARTITION variable=hd  dim=1 cyclic factor=16
#pragma HLS ARRAY_PARTITION variable=wdl dim=1 cyclic factor=16
#pragma HLS ARRAY_PARTITION variable=abuf     cyclic factor=16

    // O-3b: stage the decoder weights once (partitioned on n) for the gather-form decoder below.
    LDWD: for (int n = 0; n < N; n++)
        for (int k = 0; k < L; k++) {
#pragma HLS PIPELINE II=1
            wdl[n][k] = wts::Wdec[n][k];
        }

    // O-3c: cache audio_in on-chip once (the encoder reads it N=128x; on-board these were un-bursted
    // strided DDR reads), partitioned cyclic-16 so the 32-tap window reads in parallel.
    LDA: for (int i = 0; i < T; i++) {
#pragma HLS PIPELINE II=1
        abuf[i] = audio_in[i];
    }

    // encoder: Conv1d(1->N, k=L, stride=STRIDE, pad=STRIDE). Kernel unrolled (Wenc row -> regs), II=2.
    ENC: for (int n = 0; n < N; n++) {
        wgt_t wr[L];
#pragma HLS ARRAY_PARTITION variable=wr complete
        for (int k = 0; k < L; k++) wr[k] = wts::Wenc[n][k];
        for (int t = 0; t < T_LAT; t++) {
#pragma HLS PIPELINE II=2
            acc_t a = 0;
            for (int k = 0; k < L; k++) {
#pragma HLS UNROLL
                int s = t * STRIDE + k - STRIDE;
                sample_t x = (s >= 0 && s < T) ? abuf[s] : (sample_t)0;
                a += (acc_t)(x * wr[k]);
            }
            w[n][t] = (data_t)a;
        }
    }

    // bottleneck: y = in_norm(w) -> 1x1(N->B) + video.  in_norm inline (wn cast to data_t).
    BOT: for (int b = 0; b < B; b++) {
        wgt_t wr[N];                               // O-3b: Wbn row -> regs, unroll n (II=8)
#pragma HLS ARRAY_PARTITION variable=wr complete
        for (int n = 0; n < N; n++) wr[n] = wts::Wbn[n][b];
        for (int t = 0; t < T_LAT; t++) {
#pragma HLS PIPELINE II=8
            acc_t a = (acc_t)video_embed[b * T_LAT + t];
            for (int n = 0; n < N; n++) {
#pragma HLS UNROLL
                data_t wn = (data_t)(wts::innorm_s[n] * w[n][t] + wts::innorm_b[n]);
                a += (acc_t)(wn * wr[n]);
            }
            y[b][t] = (data_t)a;
        }
    }

    // 10 dilated dwsep TCN blocks (residual). PReLU in acc_t, then bn affine, then one data_t cast.
    BLOCKS: for (int blk = 0; blk < NBLK; blk++) {
        int dil = 1 << (blk % X);
        IN1x1: for (int hh = 0; hh < H; hh++) {
            wgt_t wr[B];                                // O-3: weight row -> regs, unroll b (II=4)
#pragma HLS ARRAY_PARTITION variable=wr complete
            for (int b = 0; b < B; b++) wr[b] = wts::Win[blk][b][hh];
            for (int t = 0; t < T_LAT; t++) {
#pragma HLS PIPELINE II=4
                acc_t a = 0;
                for (int b = 0; b < B; b++) {
#pragma HLS UNROLL
                    a += (acc_t)(y[b][t] * wr[b]);
                }
                acc_t p = (a >= 0) ? a : (acc_t)(wts::pr1[blk][hh] * a);
                h[hh][t] = (data_t)(wts::bn1_s[blk][hh] * p + wts::bn1_b[blk][hh]);
            }
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
        OUT1x1: for (int b = 0; b < B; b++) {
            wgt_t wr[H];                                // O-3: weight row -> regs, unroll hh (II=8)
#pragma HLS ARRAY_PARTITION variable=wr complete
            for (int hh = 0; hh < H; hh++) wr[hh] = wts::Wout[blk][hh][b];
            for (int t = 0; t < T_LAT; t++) {
#pragma HLS PIPELINE II=8
                acc_t a = 0;
                for (int hh = 0; hh < H; hh++) {
#pragma HLS UNROLL
                    a += (acc_t)(hd[hh][t] * wr[hh]);
                }
                y[b][t] = (data_t)((acc_t)y[b][t] + a);
            }
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

    // decoder: ConvTranspose1d(N->1, k=L, stride=STRIDE, pad=STRIDE) in GATHER form (O-3c). Each output
    // s is computed exactly once -> no scatter/RMW hazard (the rolled scatter it replaces lost updates on
    // the real pipeline; D-19). Inverting s = t*STRIDE+k-STRIDE: for a given s the only contributors are
    // (t=s/16+1, k=s%16) and (t=s/16, k=s%16+16) -- both t always in [0,T_LAT) -> no bounds check. The
    // acc_t sum is order-independent (products are exact in acc_t), so this is bit-identical to the scatter.
    DECG: for (int s = 0; s < T; s++) {
#pragma HLS PIPELINE II=16
        int tb = s / STRIDE, k0 = s % STRIDE;      // STRIDE=16 (power of 2) -> shift/mask
        acc_t acc = 0;
        for (int n = 0; n < N; n++) {
#pragma HLS UNROLL
            acc += (acc_t)(w[n][tb + 1] * wdl[n][k0])
                 + (acc_t)(w[n][tb]     * wdl[n][k0 + STRIDE]);
        }
        audio_out[s] = (sample_t)acc;
    }
}

}  // namespace c7

#endif  // C7_AUDIO_CORE_HPP
