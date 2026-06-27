// c7_video.hpp — VALUE-FAITHFUL video encoder for the monolithic AVSE deployment.
// Mirrors LightweightVideoEncoder (video_channels=96, video_layers=3) exactly, with REAL trained weights
// (c7_weights.hpp) and the same quantization points as the emulator tools/c7_fixedpoint.py:
//   per frame: Conv2d(1->64,k7,s2,p3)+BN+ReLU
//     -> 3x DepthwiseSeparableConv2d(stride2): depthwise(k3,no act) -> pointwise+BN+ReLU (main)
//        + shortcut[Conv2d(1x1,s2)+BN] -> residual add
//     -> AvgPool2d(k5,s1) [6x6->2x2] -> feature_proj Conv2d(k2)+ReLU [2x2->1x1]
//     -> temporal_proj Linear + residual
// All BN folds into its preceding conv (W_fold=W*s, bias=b) -> v_*_b biases (bn_t). Looped over 30 frames.
#ifndef C7_VIDEO_HPP
#define C7_VIDEO_HPP

#include "c7_types.hpp"
#include "c7_weights.hpp"

namespace c7 {
namespace vid {
constexpr int TF = 30, IN = 96;
constexpr int C0 = 64, C = 96;
constexpr int S0 = 48, S1 = 24, S2 = 12, S3 = 6;
}

static data_t vrelu(acc_t a) {           // ReLU then cast to data_t
#pragma HLS INLINE
    return (data_t)(a > 0 ? a : (acc_t)0);
}

static void video_encoder(const data_t *video_in, data_t video_feat[vid::C][vid::TF])
{
    using namespace vid;

    // per-frame buffers (reused across 30 frames)
    static data_t fbuf[IN][IN];      // O-1/O-2a: on-chip frame cache, partitioned 7x7 for the conv0 window
#pragma HLS ARRAY_PARTITION variable=fbuf dim=1 cyclic factor=7
#pragma HLS ARRAY_PARTITION variable=fbuf dim=2 cyclic factor=7
    static wgt_t  c0w[C0][7][7];     // O-2a: conv0 weights staged so the unrolled 7x7 kernel reads 49/clk
#pragma HLS ARRAY_PARTITION variable=c0w dim=2 complete
#pragma HLS ARRAY_PARTITION variable=c0w dim=3 complete
    static data_t b0[C0][S0 * S0];   // conv0 out  [64][2304]
    static data_t dw[C][S1 * S1];    // depthwise scratch (max [96][576])
    static data_t main_[C][S1 * S1]; // pointwise+relu main (max [96][576])
    static data_t b1[C][S1 * S1];    // stage1 out [96][576]
    static data_t b2[C][S2 * S2];    // stage2 out [96][144]
    static data_t b3[C][S3 * S3];    // stage3 out [96][36]
    static data_t pooled[C][4];      // avgpool 2x2
    static data_t fpb[C];            // feature_proj out
    // O-2b: partition the reduction-input buffers on the CHANNEL dim (the reduction axis) so the
    // pointwise/shortcut channel reductions read 16 inputs/cycle (II~4-6). Channel->bank,
    // spatial->within-bank address keeps the strided shortcut reads conflict-free (the original 6 h
    // blow-up was strided reads into SPATIALLY-partitioned buffers + complete-partitioned big ROMs;
    // both avoided here). Factor 16 keeps BRAM ~flat (complete would double it via bank rounding).
#pragma HLS ARRAY_PARTITION variable=b0 dim=1 cyclic factor=16
#pragma HLS ARRAY_PARTITION variable=dw dim=1 cyclic factor=16
#pragma HLS ARRAY_PARTITION variable=b1 dim=1 cyclic factor=16
#pragma HLS ARRAY_PARTITION variable=b2 dim=1 cyclic factor=16

    // O-2a: stage the conv0 weights once into a fully-partitioned local buffer so the unrolled 7x7
    // kernel can read all 49 taps in parallel (the namespace ROM wts::v_c0_w can't take a partition
    // pragma here). One-time copy, amortized over 30 frames.
    LDW0: for (int co = 0; co < C0; co++)
        for (int ky = 0; ky < 7; ky++)
            for (int kx = 0; kx < 7; kx++) {
#pragma HLS PIPELINE II=1
                c0w[co][ky][kx] = wts::v_c0_w[co][ky][kx];
            }

    FRAMES: for (int f = 0; f < TF; f++) {
        const data_t *img = video_in + f * IN * IN;

        // ---- O-1: burst-cache this frame on-chip (one contiguous 96x96 DDR read) so the conv reads
        // hit BRAM instead of issuing per-element DDR round-trips (the on-board 4.55x penalty). The
        // values are identical; only the memory staging changes.
        LOADF: for (int y = 0; y < IN; y++)
            for (int x = 0; x < IN; x++) {
#pragma HLS PIPELINE II=1
                fbuf[y][x] = img[y * IN + x];
            }

        // ---- conv0: Conv2d(1->64,k7,s2,p3) + BN(folded) + ReLU ----
        // O-2a: pipeline the spatial loop at II=1 with the 7x7 kernel reduction fully unrolled
        // (49 parallel MACs). fbuf is partitioned 7x7 (49 banks) and c0w is fully partitioned, so the
        // 49 input reads + 49 weight reads all land in distinct banks each cycle.
        CONV0: for (int co = 0; co < C0; co++)
            for (int oy = 0; oy < S0; oy++)
                for (int ox = 0; ox < S0; ox++) {
#pragma HLS PIPELINE II=1
                    acc_t a = (acc_t)wts::v_c0_b[co];
                    for (int ky = 0; ky < 7; ky++)
                        for (int kx = 0; kx < 7; kx++) {
#pragma HLS UNROLL
                            int iy = oy * 2 - 3 + ky, ix = ox * 2 - 3 + kx;
                            data_t v = (iy >= 0 && iy < IN && ix >= 0 && ix < IN) ? fbuf[iy][ix] : (data_t)0;
                            a += (acc_t)(v * c0w[co][ky][kx]);
                        }
                    b0[co][oy * S0 + ox] = vrelu(a);
                }

        // ---- 3 DepthwiseSeparable stages. macro-like inline per stage. ----
        // stage 1: in=b0(C0,S0) -> out b1(C,S1)
        DW1: for (int c = 0; c < C0; c++)
            for (int oy = 0; oy < S1; oy++) for (int ox = 0; ox < S1; ox++) {                acc_t a = 0;                                   // depthwise, NO activation
                for (int ky = 0; ky < 3; ky++) for (int kx = 0; kx < 3; kx++) {
                    int iy = oy * 2 - 1 + ky, ix = ox * 2 - 1 + kx;
                    data_t v = (iy >= 0 && iy < S0 && ix >= 0 && ix < S0) ? b0[c][iy * S0 + ix] : (data_t)0;
                    a += (acc_t)(v * wts::v_dw1_w[c][ky][kx]);
                }
                dw[c][oy * S1 + ox] = (data_t)a;
            }
        PW1: for (int co = 0; co < C; co++) {                  // O-2b: weight row -> regs, unroll ci (II=4)
            wgt_t wr[C0];
#pragma HLS ARRAY_PARTITION variable=wr complete
            for (int ci = 0; ci < C0; ci++) wr[ci] = wts::v_pw1_w[co][ci];
            for (int p = 0; p < S1 * S1; p++) {
#pragma HLS PIPELINE II=4
                acc_t a = (acc_t)wts::v_pw1_b[co];
                for (int ci = 0; ci < C0; ci++) {
#pragma HLS UNROLL
                    a += (acc_t)(dw[ci][p] * wr[ci]);
                }
                main_[co][p] = vrelu(a);                       // pointwise + BN + ReLU
            }
        }
        SC1: for (int co = 0; co < C; co++) {                   // shortcut Conv2d(1x1,s2)+BN, no act
            wgt_t wr[C0];
#pragma HLS ARRAY_PARTITION variable=wr complete
            for (int ci = 0; ci < C0; ci++) wr[ci] = wts::v_sc1_w[co][ci];
            for (int oy = 0; oy < S1; oy++) for (int ox = 0; ox < S1; ox++) {
#pragma HLS PIPELINE II=4
                acc_t a = (acc_t)wts::v_sc1_b[co];
                for (int ci = 0; ci < C0; ci++) {
#pragma HLS UNROLL
                    a += (acc_t)(b0[ci][(oy * 2) * S0 + ox * 2] * wr[ci]);
                }
                int p = oy * S1 + ox;
                b1[co][p] = (data_t)((acc_t)main_[co][p] + (acc_t)(data_t)a);   // q(main) + q(sc)
            }
        }

        // stage 2: in=b1(C,S1) -> out b2(C,S2)
        DW2: for (int c = 0; c < C; c++)
            for (int oy = 0; oy < S2; oy++) for (int ox = 0; ox < S2; ox++) {                acc_t a = 0;
                for (int ky = 0; ky < 3; ky++) for (int kx = 0; kx < 3; kx++) {
                    int iy = oy * 2 - 1 + ky, ix = ox * 2 - 1 + kx;
                    data_t v = (iy >= 0 && iy < S1 && ix >= 0 && ix < S1) ? b1[c][iy * S1 + ix] : (data_t)0;
                    a += (acc_t)(v * wts::v_dw2_w[c][ky][kx]);
                }
                dw[c][oy * S2 + ox] = (data_t)a;
            }
        PW2: for (int co = 0; co < C; co++) {                  // O-2b: weight row -> regs, unroll ci (II=6)
            wgt_t wr[C];
#pragma HLS ARRAY_PARTITION variable=wr complete
            for (int ci = 0; ci < C; ci++) wr[ci] = wts::v_pw2_w[co][ci];
            for (int p = 0; p < S2 * S2; p++) {
#pragma HLS PIPELINE II=6
                acc_t a = (acc_t)wts::v_pw2_b[co];
                for (int ci = 0; ci < C; ci++) {
#pragma HLS UNROLL
                    a += (acc_t)(dw[ci][p] * wr[ci]);
                }
                main_[co][p] = vrelu(a);
            }
        }
        SC2: for (int co = 0; co < C; co++) {
            wgt_t wr[C];
#pragma HLS ARRAY_PARTITION variable=wr complete
            for (int ci = 0; ci < C; ci++) wr[ci] = wts::v_sc2_w[co][ci];
            for (int oy = 0; oy < S2; oy++) for (int ox = 0; ox < S2; ox++) {
#pragma HLS PIPELINE II=6
                acc_t a = (acc_t)wts::v_sc2_b[co];
                for (int ci = 0; ci < C; ci++) {
#pragma HLS UNROLL
                    a += (acc_t)(b1[ci][(oy * 2) * S1 + ox * 2] * wr[ci]);
                }
                int p = oy * S2 + ox;
                b2[co][p] = (data_t)((acc_t)main_[co][p] + (acc_t)(data_t)a);
            }
        }

        // stage 3: in=b2(C,S2) -> out b3(C,S3)
        DW3: for (int c = 0; c < C; c++)
            for (int oy = 0; oy < S3; oy++) for (int ox = 0; ox < S3; ox++) {                acc_t a = 0;
                for (int ky = 0; ky < 3; ky++) for (int kx = 0; kx < 3; kx++) {
                    int iy = oy * 2 - 1 + ky, ix = ox * 2 - 1 + kx;
                    data_t v = (iy >= 0 && iy < S2 && ix >= 0 && ix < S2) ? b2[c][iy * S2 + ix] : (data_t)0;
                    a += (acc_t)(v * wts::v_dw3_w[c][ky][kx]);
                }
                dw[c][oy * S3 + ox] = (data_t)a;
            }
        PW3: for (int co = 0; co < C; co++) {                  // O-2b: weight row -> regs, unroll ci (II=6)
            wgt_t wr[C];
#pragma HLS ARRAY_PARTITION variable=wr complete
            for (int ci = 0; ci < C; ci++) wr[ci] = wts::v_pw3_w[co][ci];
            for (int p = 0; p < S3 * S3; p++) {
#pragma HLS PIPELINE II=6
                acc_t a = (acc_t)wts::v_pw3_b[co];
                for (int ci = 0; ci < C; ci++) {
#pragma HLS UNROLL
                    a += (acc_t)(dw[ci][p] * wr[ci]);
                }
                main_[co][p] = vrelu(a);
            }
        }
        SC3: for (int co = 0; co < C; co++) {
            wgt_t wr[C];
#pragma HLS ARRAY_PARTITION variable=wr complete
            for (int ci = 0; ci < C; ci++) wr[ci] = wts::v_sc3_w[co][ci];
            for (int oy = 0; oy < S3; oy++) for (int ox = 0; ox < S3; ox++) {
#pragma HLS PIPELINE II=6
                acc_t a = (acc_t)wts::v_sc3_b[co];
                for (int ci = 0; ci < C; ci++) {
#pragma HLS UNROLL
                    a += (acc_t)(b2[ci][(oy * 2) * S2 + ox * 2] * wr[ci]);
                }
                int p = oy * S3 + ox;
                b3[co][p] = (data_t)((acc_t)main_[co][p] + (acc_t)(data_t)a);
            }
        }

        // ---- AvgPool2d(k5,s1): 6x6 -> 2x2 ----
        POOL: for (int c = 0; c < C; c++)
            for (int oy = 0; oy < 2; oy++) for (int ox = 0; ox < 2; ox++) {
                acc_t s = 0;
                for (int dy = 0; dy < 5; dy++) for (int dx = 0; dx < 5; dx++)
                    s += (acc_t)b3[c][(oy + dy) * S3 + (ox + dx)];
                pooled[c][oy * 2 + ox] = (data_t)(s / (acc_t)25);
            }

        // ---- feature_proj: Conv2d(96,96,k2) + BN(bias) + ReLU : 2x2 -> 1x1 ----
        // NOT pipelined: this tiny head (96 outputs x 30 frames) has enormous time slack; pipelining
        // forces the 96*2*2 reduction to fully unroll and COMPLETE-partitions the 36864-entry v_fp_w ROM
        // into registers -> the synthesis explodes (hours). Rolled keeps v_fp_w as ROM; latency ~ns.
        FP: for (int co = 0; co < C; co++) {
            acc_t a = (acc_t)wts::v_fp_b[co];
            for (int ci = 0; ci < C; ci++)
                for (int ky = 0; ky < 2; ky++) for (int kx = 0; kx < 2; kx++)
                    a += (acc_t)(pooled[ci][ky * 2 + kx] * wts::v_fp_w[co][ci][ky][kx]);
            fpb[co] = vrelu(a);
        }

        // ---- temporal_proj: Linear(96,96) + residual ----
        // NOT pipelined (same reason as FP): keeps the 9216-entry v_tp_w as a ROM instead of registers.
        TPROJ: for (int o = 0; o < C; o++) {
            acc_t a = (acc_t)wts::v_tp_b[o];
            for (int i = 0; i < C; i++) a += (acc_t)(fpb[i] * wts::v_tp_w[o][i]);
            video_feat[o][f] = (data_t)(a + (acc_t)fpb[o]);
        }
    }
}

}  // namespace c7
#endif  // C7_VIDEO_HPP
