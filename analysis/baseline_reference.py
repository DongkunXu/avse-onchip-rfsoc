"""baseline_reference.py — the reference 4-IP deployment, encoded as data.

Every buffer below is transcribed from the actual HLS source of the reference project
(`../UNet-AVSE-Vitis/src/ip_*/...top.cpp`), NOT guessed — partition factors and storage come from
the real `#pragma HLS ARRAY_PARTITION` / `BIND_STORAGE` lines. This is what validate_baseline.py
checks the model against.

Reference: Tier E.1 — audio_channels=192, audio_layers=5, kernels [19,15,11,9,7], strides all 2;
video_channels=96, video_layers=3; fusion_dim=64. data_t = ap_fixed<16,7> (16-bit).
"""
from __future__ import annotations

from working_set import Buffer, Module, Design, LiveTensor

# ── audio_enc — holds the WHOLE skip set + bottleneck (it produces them) ───────────────────────
#   src/ip_audio_enc/audio_encoder_top.cpp lines 139-150
audio_enc = Module(
    name="audio_enc",
    buffers=[
        Buffer("buf_in0",   1,   19200, partition=1),   # dim=1 complete, C=1 -> 1 bank
        Buffer("skip0_buf", 32,  9600,  partition=2),
        Buffer("skip1_buf", 64,  4800,  partition=2),
        Buffer("skip2_buf", 96,  2400,  partition=2),
        Buffer("skip3_buf", 128, 1200,  partition=2),
        Buffer("bot_buf",   192, 600,   partition=2),
    ],
    note="5 stride-2 dwsep layers; outputs skip0..3 + bottleneck (all resident).",
)

# ── audio_dec — the BRAM-binding IP (95%). DATAFLOW dropped => NO ping-pong. ────────────────────
#   src/ip_audio_dec/audio_decoder_top.cpp lines 89-135
audio_dec = Module(
    name="audio_dec",
    buffers=[
        # input staging (DMA'd from DDR in the deployed split)
        Buffer("bottleneck_buf", 192, 600,  storage="uram"),          # BIND_STORAGE uram (naive 16b)
        Buffer("skip0_buf",      32,  9600, partition=2),
        Buffer("skip1_buf",      64,  4800, partition=2),
        Buffer("skip2_buf",      96,  2400, partition=2),
        Buffer("skip3_buf",      128, 1200, partition=1),             # no partition pragma
        # per-stage decoder buffers (in-place fuse)
        Buffer("s0_main",        128, 1200, partition=2),
        Buffer("s1_main",        96,  2400, partition=2),
        Buffer("s2_main",        64,  4800, partition=2),
        Buffer("s3_main",        32,  9600, partition=2),
        Buffer("s4_out",         1,   19200, partition=1),
    ],
    note="5 upsample+dwsep stages, 4 skip-fuses; bottleneck offloaded to URAM.",
)

# ── fusion — small tensors, but partition-2 banking inflates BRAM ──────────────────────────────
#   src/ip_fusion/fusion_top.cpp (all static data_t arrays)
fusion = Module(
    name="fusion",
    buffers=[
        Buffer("audio_in_buf",  192, 600, partition=2),
        Buffer("v_ct",          96,  30,  partition=2),
        Buffer("a_proj",        64,  600, partition=2),
        Buffer("v_proj",        64,  30,  partition=2),
        Buffer("a_attended",    64,  600, partition=2),
        Buffer("v_attended",    64,  30,  partition=2),
        Buffer("v_up",          64,  600, partition=2),
        Buffer("v_aligned_dw",  64,  600, partition=2),
        Buffer("v_aligned",     64,  600, partition=2),
        Buffer("a_norm",        64,  600, partition=2),
        Buffer("v_norm",        64,  600, partition=2),
        Buffer("concat_fused",  64,  600, partition=2),
        Buffer("enhanced_audio",64,  600, partition=2),
        Buffer("fused_out_buf", 192, 600, partition=2),
    ],
    note="concat-projection fusion at fusion_dim=64; audio bottleneck 192x600 in/out.",
)

# ── video — per-frame spatial pipeline, DATAFLOW staged (some buffers ping-pong) ────────────────
#   src/ip_video/video_encoder_top.cpp
video = Module(
    name="video",
    buffers=[
        Buffer("in_1ch",     1,   9216, partition=1),       # 96x96
        Buffer("buf_v0",     64,  2304, partition=2),        # 48x48
        Buffer("v1_dw_out",  64,  576,  partition=2),        # 24x24
        Buffer("v1_pw_out",  96,  576,  partition=2),
        Buffer("v1_sc_out",  96,  576,  partition=2),
        Buffer("v2_dw_out",  96,  144,  partition=2),        # 12x12
        Buffer("v2_pw_out",  96,  144,  partition=2),
        Buffer("v2_sc_out",  96,  144,  partition=2),
        Buffer("buf_v2",     96,  144,  partition=2),
        Buffer("v3_dw_out",  96,  36,   partition=2),        # 6x6
        Buffer("v3_pw_out",  96,  36,   partition=2),
        Buffer("v3_sc_out",  96,  36,   partition=2),
        Buffer("buf_v5",     96,  4,    partition=2),        # 2x2
        Buffer("flat_v5",    384, 1,    partition=32),       # cyclic f=32 -> 32 banks
        # DATAFLOW inter-stage buffers (per_frame_combined: 3-stage pipeline -> ping-pong)
        Buffer("buf_v1_inter", 96, 576, partition=2, pingpong=True),
        Buffer("buf_v3_inter", 96, 36,  partition=2, pingpong=True),
        # 30-frame feature accumulation (T_video=30, C=96)
        Buffer("feat_buf",   30,  96,   partition=1),
    ],
    note="per-frame 96x96->1x1; cheap activations, but DATAFLOW ping-pong + flat_v5 banking add BRAM.",
)


def reference_design() -> Design:
    """The monolithic single-config design: all 4 IPs co-resident (the 215% case)."""
    return Design("reference_monolithic", [audio_enc, audio_dec, fusion, video])


# ── Liveness schedule of the monolithic AUDIO U-Net (for the static-vs-peak gap) ────────────────
# Schedule steps: enc0..enc4 (0-4), fusion (5), dec0..dec4 (6-10).
# A skip is live from the encoder step that produces it until the decoder step that consumes it.
def audio_unet_liveness() -> list:
    return [
        LiveTensor("in_audio",   1,   19200, produced=0, last_used=0),
        LiveTensor("skip0",      32,  9600,  produced=0, last_used=9),   # consumed at dec3
        LiveTensor("skip1",      64,  4800,  produced=1, last_used=8),   # dec2
        LiveTensor("skip2",      96,  2400,  produced=2, last_used=7),   # dec1
        LiveTensor("skip3",      128, 1200,  produced=3, last_used=6),   # dec0
        LiveTensor("bottleneck", 192, 600,   produced=4, last_used=6),   # fusion->dec0
        LiveTensor("dec_s0",     128, 1200,  produced=6, last_used=7),
        LiveTensor("dec_s1",     96,  2400,  produced=7, last_used=8),
        LiveTensor("dec_s2",     64,  4800,  produced=8, last_used=9),
        LiveTensor("dec_s3",     32,  9600,  produced=9, last_used=10),
        LiveTensor("dec_s4",     1,   19200, produced=10, last_used=10),
    ]
