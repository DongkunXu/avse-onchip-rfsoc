// c7_audio_core.hpp — the C7 audio mask network as a reusable core (no AXI interface).
// Shared by the standalone IP (c7_audio_top) and the monolithic AVSE IP (c7_avse_top), so the audio
// compute lives in exactly one place. video_embed is a flat [B*T_LAT] row-major signal that the
// caller supplies (from m_axi in the standalone, or from the on-chip video encoder in the monolith).
#ifndef C7_AUDIO_CORE_HPP
#define C7_AUDIO_CORE_HPP

#include "c7_types.hpp"

namespace c7 {

static data_t hsig(acc_t x) {            // hardsigmoid ~ clamp(0.2x+0.5, 0, 1)
#pragma HLS INLINE
    acc_t v = (acc_t)0.2 * x + (acc_t)0.5;
    if (v < 0) v = 0; else if (v > 1) v = 1;
    return (data_t)v;
}
static data_t prelu(acc_t x, wgt_t a) {
#pragma HLS INLINE
    return (data_t)(x >= 0 ? x : (acc_t)(a * x));
}

// audio_in[T], video_embed[B*T_LAT] (row-major [B][T_LAT]), audio_out[T].
static void audio_core(const sample_t *audio_in,
                       const data_t   *video_embed,
                       sample_t       *audio_out)
{
    using namespace cfg;

    static data_t w [N][T_LAT];     // encoded latent; reused to hold w*mask
    static data_t y [B][T_LAT];     // TCN state (in-place residual)
    static data_t h [H][T_LAT];     // in_conv output
    static data_t hd[H][T_LAT];     // dwconv output
#pragma HLS ARRAY_PARTITION variable=w  dim=1 cyclic factor=2
#pragma HLS ARRAY_PARTITION variable=y  dim=1 cyclic factor=2
#pragma HLS ARRAY_PARTITION variable=h  dim=1 cyclic factor=2
#pragma HLS ARRAY_PARTITION variable=hd dim=1 cyclic factor=2

    static wgt_t Wenc[N][L], Wdec[N][L];
    static wgt_t bn_s[N], bn_b[N];
    static wgt_t Wbn[N][B], Wmask[B][N];
    static wgt_t Win [NBLK][B][H];
    static wgt_t Wdw [NBLK][H][KD];
    static wgt_t Wout[NBLK][H][B];
    static wgt_t pr1[NBLK][H], pr2[NBLK][H];

    INIT_W: {
        for (int n = 0; n < N; n++) {
            for (int k = 0; k < L; k++) { Wenc[n][k] = (wgt_t)(((n*7+k*3)%17-8)*0.02);
                                          Wdec[n][k] = (wgt_t)(((n*5+k*11)%17-8)*0.02); }
            bn_s[n] = (wgt_t)1.0; bn_b[n] = (wgt_t)0.0;
            for (int b = 0; b < B; b++) Wbn[n][b] = (wgt_t)(((n+b)%13-6)*0.03);
        }
        for (int b = 0; b < B; b++) for (int n = 0; n < N; n++) Wmask[b][n] = (wgt_t)(((b*3+n)%13-6)*0.03);
        for (int i = 0; i < NBLK; i++) {
            for (int b = 0; b < B; b++) for (int hh = 0; hh < H; hh++) Win[i][b][hh] = (wgt_t)(((i+b+hh)%11-5)*0.03);
            for (int hh = 0; hh < H; hh++) {
                for (int j = 0; j < KD; j++) Wdw[i][hh][j] = (wgt_t)(((i+hh+j)%7-3)*0.05);
                for (int b = 0; b < B; b++)  Wout[i][hh][b] = (wgt_t)(((i+hh+b)%11-5)*0.03);
                pr1[i][hh] = (wgt_t)0.1; pr2[i][hh] = (wgt_t)0.1;
            }
        }
    }

    // encoder: Conv1d(1->N, k=L, stride=STRIDE)
    ENC: for (int n = 0; n < N; n++)
        for (int t = 0; t < T_LAT; t++) {
#pragma HLS PIPELINE II=2
            acc_t a = 0;
            for (int k = 0; k < L; k++) {
                int s = t * STRIDE + k - STRIDE;
                sample_t x = (s >= 0 && s < T) ? audio_in[s] : (sample_t)0;
                a += (acc_t)(x * Wenc[n][k]);
            }
            w[n][t] = (data_t)a;
        }

    // bottleneck: y = BN(w)*Wbn + video
    BOT: for (int b = 0; b < B; b++)
        for (int t = 0; t < T_LAT; t++) {
#pragma HLS PIPELINE II=2
            acc_t a = (acc_t)video_embed[b * T_LAT + t];
            for (int n = 0; n < N; n++) {
                data_t wn = (data_t)(bn_s[n] * w[n][t] + bn_b[n]);
                a += (acc_t)(wn * Wbn[n][b]);
            }
            y[b][t] = (data_t)a;
        }

    // 10 dilated dwsep TCN blocks (residual)
    BLOCKS: for (int blk = 0; blk < NBLK; blk++) {
        int dil = 1 << (blk % X);
        IN1x1: for (int hh = 0; hh < H; hh++)
            for (int t = 0; t < T_LAT; t++) {
#pragma HLS PIPELINE II=2
                acc_t a = 0;
                for (int b = 0; b < B; b++) a += (acc_t)(y[b][t] * Win[blk][b][hh]);
                h[hh][t] = prelu(a, pr1[blk][hh]);
            }
        DW: for (int hh = 0; hh < H; hh++)
            for (int t = 0; t < T_LAT; t++) {
#pragma HLS PIPELINE II=2
                acc_t a = 0;
                for (int j = 0; j < KD; j++) {
                    int tt = t - (KD - 1 - j) * dil;
                    data_t hv = (tt >= 0) ? h[hh][tt] : (data_t)0;
                    a += (acc_t)(hv * Wdw[blk][hh][j]);
                }
                hd[hh][t] = prelu(a, pr2[blk][hh]);
            }
        OUT1x1: for (int b = 0; b < B; b++)
            for (int t = 0; t < T_LAT; t++) {
#pragma HLS PIPELINE II=2
                acc_t a = 0;
                for (int hh = 0; hh < H; hh++) a += (acc_t)(hd[hh][t] * Wout[blk][hh][b]);
                y[b][t] = (data_t)((acc_t)y[b][t] + a);
            }
    }

    // mask + apply (overwrite w with w*mask)
    MASK: for (int n = 0; n < N; n++)
        for (int t = 0; t < T_LAT; t++) {
#pragma HLS PIPELINE II=2
            acc_t a = 0;
            for (int b = 0; b < B; b++) a += (acc_t)(y[b][t] * Wmask[b][n]);
            w[n][t] = (data_t)(w[n][t] * hsig(a));
        }

    // decoder: ConvTranspose1d(N->1, k=L, stride=STRIDE), accumulate
    static acc_t obuf[T];
    INIT_O: for (int s = 0; s < T; s++) obuf[s] = 0;
    DEC: for (int n = 0; n < N; n++)
        for (int t = 0; t < T_LAT; t++) {
#pragma HLS PIPELINE II=2
            data_t wv = w[n][t];
            for (int k = 0; k < L; k++) {
                int s = t * STRIDE + k;
                if (s < T) obuf[s] += (acc_t)(wv * Wdec[n][k]);
            }
        }
    CASTO: for (int s = 0; s < T; s++) {
#pragma HLS PIPELINE II=1
        audio_out[s] = (sample_t)obuf[s];
    }
}

}  // namespace c7

#endif  // C7_AUDIO_CORE_HPP
