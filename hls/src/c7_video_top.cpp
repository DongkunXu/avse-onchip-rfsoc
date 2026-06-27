// c7_video_top.cpp — standalone video-encoder IP (AXI wrapper) for ISOLATED resource/timing synthesis.
// Used to characterize the faithful video encoder on its own (the monolithic AVSE synthesis is dominated
// by it). Not part of the deployed single-config design (that is c7_avse_top); this is a diagnostic top.
#include "c7_video.hpp"

using namespace c7;

extern "C" void c7_video_top(const data_t *video_in, data_t *video_feat_out)
{
#pragma HLS INTERFACE m_axi port=video_in       offset=slave bundle=gmem0 depth=276480
#pragma HLS INTERFACE m_axi port=video_feat_out offset=slave bundle=gmem1 depth=2880
#pragma HLS INTERFACE s_axilite port=video_in       bundle=control
#pragma HLS INTERFACE s_axilite port=video_feat_out bundle=control
#pragma HLS INTERFACE s_axilite port=return         bundle=control

    static data_t video_feat[vid::C][vid::TF];
    video_encoder(video_in, video_feat);
    OUT: for (int c = 0; c < vid::C; c++)
        for (int f = 0; f < vid::TF; f++)
            video_feat_out[c * vid::TF + f] = video_feat[c][f];
}
