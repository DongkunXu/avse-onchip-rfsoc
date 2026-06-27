// ============================================================================
// c7_avse_top — MONOLITHIC C7 AVSE IP: video encoder + C7 audio mask network in
// ONE design. Gives the real whole-system single-config fit number (not the
// audio+video estimate). Inputs: audio[19200] + video[30*96*96]; output[19200].
//   video_encoder -> video_feat[96][30] --proj(96->64)+upsample(30->1201)-->
//   video_embed[64][1201] --> audio_core(audio, video_embed) -> out
// REAL trained weights (c7_weights.hpp). VPROJ/VUP are rolled (conservative, like the video encoder,
// D-19): pipelining the trivial 96->64 projection auto-complete-partitioned video_feat into 96 banks
// (~13% BRAM) for no throughput benefit. Throughput optimization is a separate later phase.
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

    // ---- project (96->B) per frame (proj Conv1d + bias), then upsample (30 -> T_LAT) ----
    static data_t vproj[B][vid::TF];
    VPROJ: for (int b = 0; b < B; b++)
        for (int fr = 0; fr < vid::TF; fr++) {
            acc_t a = (acc_t)wts::vproj_b[b];
            for (int c = 0; c < vid::C; c++) a += (acc_t)(video_feat[c][fr] * wts::vproj_w[b][c]);
            vproj[b][fr] = (data_t)a;
        }

    static data_t video_embed[B * T_LAT];
    VUP: for (int b = 0; b < B; b++)            // nearest upsample TF->T_LAT: frame = floor(t*TF/T_LAT)
        for (int t = 0; t < T_LAT; t++) {
            video_embed[b * T_LAT + t] = vproj[b][(t * vid::TF) / T_LAT];
        }

    // ---- C7 audio mask network (shared core) ----
    audio_core(audio_in, video_embed, audio_out);
}
