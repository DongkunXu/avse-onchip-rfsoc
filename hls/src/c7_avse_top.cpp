// ============================================================================
// c7_avse_top — MONOLITHIC C7 AVSE IP: video encoder + C7 audio mask network in
// ONE design. Gives the real whole-system single-config fit number (not the
// audio+video estimate). Inputs: audio[19200] + video[30*96*96]; output[19200].
//   video_encoder -> video_feat[96][30] --proj(96->64)+upsample(30->1200)-->
//   video_embed[64][1200] --> audio_core(audio, video_embed) -> out
// Placeholder weights (fit is structure-driven, D-9/D-10).
// ============================================================================
#include "c7_audio_core.hpp"
#include "c7_video.hpp"

using namespace c7;
using namespace c7::cfg;

extern "C" void c7_avse_top(
    const sample_t *audio_in,    // [T] = 19200
    const data_t   *video_in,    // [30*96*96] grayscale in [0,1]
    sample_t       *audio_out)   // [T]
{
#pragma HLS INTERFACE m_axi port=audio_in  offset=slave bundle=gmem0 depth=19200
#pragma HLS INTERFACE m_axi port=video_in  offset=slave bundle=gmem1 depth=276480
#pragma HLS INTERFACE m_axi port=audio_out offset=slave bundle=gmem2 depth=19200
#pragma HLS INTERFACE s_axilite port=audio_in  bundle=control
#pragma HLS INTERFACE s_axilite port=video_in  bundle=control
#pragma HLS INTERFACE s_axilite port=audio_out bundle=control
#pragma HLS INTERFACE s_axilite port=return    bundle=control

    // ---- video encoder -> [96][30] ----
    static data_t video_feat[vid::C][vid::TF];
    video_encoder(video_in, video_feat);

    // ---- project (96->B) per frame, then upsample (30 -> T_LAT) -> video_embed[B*T_LAT] ----
    static wgt_t Wvproj[vid::C][B];
    VPINIT: for (int c = 0; c < vid::C; c++)
        for (int b = 0; b < B; b++) Wvproj[c][b] = (wgt_t)(((c + b) % 11 - 5) * 0.03);

    static data_t vproj[B][vid::TF];
    VPROJ: for (int b = 0; b < B; b++)
        for (int fr = 0; fr < vid::TF; fr++) {
#pragma HLS PIPELINE
            acc_t a = 0;
            for (int c = 0; c < vid::C; c++) a += (acc_t)(video_feat[c][fr] * Wvproj[c][b]);
            vproj[b][fr] = (data_t)a;
        }

    static data_t video_embed[B * T_LAT];
    VUP: for (int b = 0; b < B; b++)            // nearest upsample TF->T_LAT: frame = floor(t*TF/T_LAT)
        for (int t = 0; t < T_LAT; t++) {
#pragma HLS PIPELINE
            video_embed[b * T_LAT + t] = vproj[b][(t * vid::TF) / T_LAT];
        }

    // ---- C7 audio mask network (shared core) ----
    audio_core(audio_in, video_embed, audio_out);
}
