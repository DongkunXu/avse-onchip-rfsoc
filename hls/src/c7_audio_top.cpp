// ============================================================================
// c7_audio_top — standalone C7 audio mask network IP (AXI wrapper around the
// shared audio core). video_embed is an INPUT port (the video encoder is a
// separate IP that feeds this one on-chip in the single static configuration).
// See c7_audio_core.hpp for the computation and hls/PHASE3_PLAN.md for context.
// ============================================================================
#include "c7_audio_core.hpp"

using namespace c7;
using namespace c7::cfg;

extern "C" void c7_audio_top(
    const sample_t *audio_in,     // [T]
    const data_t   *video_embed,  // [B*T_LAT] row-major [B][T_LAT]
    sample_t       *audio_out)    // [T]
{
#pragma HLS INTERFACE m_axi port=audio_in    offset=slave bundle=gmem0 depth=19200
#pragma HLS INTERFACE m_axi port=video_embed offset=slave bundle=gmem1 depth=76800
#pragma HLS INTERFACE m_axi port=audio_out   offset=slave bundle=gmem2 depth=19200
#pragma HLS INTERFACE s_axilite port=audio_in    bundle=control
#pragma HLS INTERFACE s_axilite port=video_embed bundle=control
#pragma HLS INTERFACE s_axilite port=audio_out   bundle=control
#pragma HLS INTERFACE s_axilite port=return      bundle=control

    audio_core(audio_in, video_embed, audio_out);
}
