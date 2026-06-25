// c7_video.hpp — compact video encoder for the monolithic AVSE fit check.
// Mirrors the reused LightweightVideoEncoder (video_channels=96, video_layers=3):
//   per frame: Conv2d(1->64,k7,s2,p3) -> 3x DWSep Conv2d(stride2,k3) -> mean-pool -> temporal proj
// Looped over 30 frames -> video_feat[96][30]. Placeholder (index-seeded) weights; the cost, not the
// values, is what the fit check needs. Self-contained; layers inlined (no VLA function params).
#ifndef C7_VIDEO_HPP
#define C7_VIDEO_HPP

#include "c7_types.hpp"

namespace c7 {
namespace vid {
constexpr int TF = 30, IN = 96;
constexpr int C0 = 64, C = 96;
constexpr int S0 = 48, S1 = 24, S2 = 12, S3 = 6;
}

static void video_encoder(const data_t *video_in, data_t video_feat[vid::C][vid::TF])
{
    using namespace vid;

    // ---- placeholder weights ----
    static wgt_t W0[C0][7][7];
    static wgt_t Wdw1[C0][3][3], Wpw1[C0][C];
    static wgt_t Wdw2[C][3][3],  Wpw2[C][C];
    static wgt_t Wdw3[C][3][3],  Wpw3[C][C];
    static wgt_t Wtp[C][C];
    VINIT: {
        for (int o = 0; o < C0; o++) {
            for (int a=0;a<7;a++) for (int b=0;b<7;b++) W0[o][a][b]=(wgt_t)(((o+a*7+b)%13-6)*0.02);
            for (int a=0;a<3;a++) for (int b=0;b<3;b++) Wdw1[o][a][b]=(wgt_t)(((o+a+b)%7-3)*0.05);
        }
        for (int o=0;o<C;o++){
            for (int i=0;i<C0;i++) Wpw1[i][o]=(wgt_t)(((i+o)%11-5)*0.03);
            for (int a=0;a<3;a++) for (int b=0;b<3;b++){ Wdw2[o][a][b]=(wgt_t)(((o+a+b)%7-3)*0.05); Wdw3[o][a][b]=(wgt_t)(((o*2+a+b)%7-3)*0.05);}
            for (int i=0;i<C;i++){ Wpw2[i][o]=(wgt_t)(((i*2+o)%11-5)*0.03); Wpw3[i][o]=(wgt_t)(((i+o*2)%11-5)*0.03); Wtp[i][o]=(wgt_t)(((i+o)%13-6)*0.02);}
        }
    }

    // ---- per-frame buffers (reused across 30 frames) ----
    static data_t b0[C0][S0 * S0];   // [64][2304]
    static data_t dw[C][S1 * S1];    // depthwise scratch (max [96][576])
    static data_t b1[C][S1 * S1];    // [96][576]
    static data_t b2[C][S2 * S2];    // [96][144]
    static data_t b3[C][S3 * S3];    // [96][36]
#pragma HLS ARRAY_PARTITION variable=b0 dim=1 cyclic factor=2
#pragma HLS ARRAY_PARTITION variable=dw dim=1 cyclic factor=2
#pragma HLS ARRAY_PARTITION variable=b1 dim=1 cyclic factor=2
    static data_t pooled[C];

    FRAMES: for (int f = 0; f < TF; f++) {
        const data_t *img = video_in + f * IN * IN;

        // conv0: Conv2d(1->64,k7,s2,p3) + ReLU
        CONV0: for (int co = 0; co < C0; co++)
            for (int oy = 0; oy < S0; oy++)
                for (int ox = 0; ox < S0; ox++) {
#pragma HLS PIPELINE II=2
                    acc_t a = 0;
                    for (int ky = 0; ky < 7; ky++)
                        for (int kx = 0; kx < 7; kx++) {
                            int iy = oy*2-3+ky, ix = ox*2-3+kx;
                            data_t v = (iy>=0&&iy<IN&&ix>=0&&ix<IN) ? img[iy*IN+ix] : (data_t)0;
                            a += (acc_t)(v * W0[co][ky][kx]);
                        }
                    b0[co][oy*S0+ox] = (data_t)(a>0 ? a : (acc_t)0);
                }

        // ---- DWSep stage 1: depthwise(64,k3,s2,p1) b0[48]->dw[24]; pointwise(64->96) -> b1[24] ----
        DW1: for (int c = 0; c < C0; c++)
            for (int oy = 0; oy < S1; oy++) for (int ox = 0; ox < S1; ox++) {
#pragma HLS PIPELINE II=2
                acc_t a=0;
                for (int ky=0;ky<3;ky++) for (int kx=0;kx<3;kx++){ int iy=oy*2-1+ky,ix=ox*2-1+kx;
                    data_t v=(iy>=0&&iy<S0&&ix>=0&&ix<S0)?b0[c][iy*S0+ix]:(data_t)0; a+=(acc_t)(v*Wdw1[c][ky][kx]); }
                dw[c][oy*S1+ox]=(data_t)(a>0?a:(acc_t)0);
            }
        PW1: for (int co = 0; co < C; co++)
            for (int p = 0; p < S1*S1; p++) {
#pragma HLS PIPELINE II=2
                acc_t a=0; for (int ci=0;ci<C0;ci++) a+=(acc_t)(dw[ci][p]*Wpw1[ci][co]);
                b1[co][p]=(data_t)(a>0?a:(acc_t)0);
            }

        // ---- DWSep stage 2: 96->96, [24]->[12] ----
        DW2: for (int c=0;c<C;c++) for (int oy=0;oy<S2;oy++) for (int ox=0;ox<S2;ox++) {
#pragma HLS PIPELINE II=2
            acc_t a=0; for(int ky=0;ky<3;ky++) for(int kx=0;kx<3;kx++){int iy=oy*2-1+ky,ix=ox*2-1+kx;
                data_t v=(iy>=0&&iy<S1&&ix>=0&&ix<S1)?b1[c][iy*S1+ix]:(data_t)0; a+=(acc_t)(v*Wdw2[c][ky][kx]);}
            dw[c][oy*S2+ox]=(data_t)(a>0?a:(acc_t)0);
        }
        PW2: for (int co=0;co<C;co++) for (int p=0;p<S2*S2;p++) {
#pragma HLS PIPELINE II=2
            acc_t a=0; for(int ci=0;ci<C;ci++) a+=(acc_t)(dw[ci][p]*Wpw2[ci][co]); b2[co][p]=(data_t)(a>0?a:(acc_t)0);
        }

        // ---- DWSep stage 3: 96->96, [12]->[6] ----
        DW3: for (int c=0;c<C;c++) for (int oy=0;oy<S3;oy++) for (int ox=0;ox<S3;ox++) {
#pragma HLS PIPELINE II=2
            acc_t a=0; for(int ky=0;ky<3;ky++) for(int kx=0;kx<3;kx++){int iy=oy*2-1+ky,ix=ox*2-1+kx;
                data_t v=(iy>=0&&iy<S2&&ix>=0&&ix<S2)?b2[c][iy*S2+ix]:(data_t)0; a+=(acc_t)(v*Wdw3[c][ky][kx]);}
            dw[c][oy*S3+ox]=(data_t)(a>0?a:(acc_t)0);
        }
        PW3: for (int co=0;co<C;co++) for (int p=0;p<S3*S3;p++) {
#pragma HLS PIPELINE II=2
            acc_t a=0; for(int ci=0;ci<C;ci++) a+=(acc_t)(dw[ci][p]*Wpw3[ci][co]); b3[co][p]=(data_t)(a>0?a:(acc_t)0);
        }

        // ---- mean-pool [96][6x6] -> [96], temporal proj (96->96) + residual ----
        POOL: for (int c=0;c<C;c++) {
#pragma HLS PIPELINE
            acc_t s=0; for (int p=0;p<S3*S3;p++) s+=(acc_t)b3[c][p];
            pooled[c]=(data_t)(s/(acc_t)(S3*S3));
        }
        TPROJ: for (int o=0;o<C;o++) {
#pragma HLS PIPELINE
            acc_t a=(acc_t)pooled[o]; for (int i=0;i<C;i++) a+=(acc_t)(pooled[i]*Wtp[i][o]);
            video_feat[o][f]=(data_t)a;
        }
    }
}

}  // namespace c7
#endif  // C7_VIDEO_HPP
