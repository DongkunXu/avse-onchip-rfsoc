# Phase 2 — quality vs deployable working-set Pareto

SI-SDR / PESQ / STOI on LRS3 dev windows vs the **deployable** on-chip activation working set (MB, audio path; Phase-1-consistent). Anchors are not retrained here.

| candidate | SI-SDR (dB) | PESQ-WB | STOI | working set (MB) | params | note |
|---|---:|---:|---:|---:|---:|---|
| Reference U-Net (4-bitstream) | 5.46 | 1.743 | 0.738 | 4.10 | — | deployed baseline; does NOT fit single-config (215% BRAM) |
| C4 Tiled U-Net (= ref quality) | 5.46 | 1.743 | 0.738 | 0.30 | — | tiling is math-identical to the reference; fits single-config |
| C2 StreamingTCNAVSE (mapping) (p2-c2-r1) | 1.12 | 1.478 | 0.672 | 0.033 | 343,616 | best of 10ep / 10000 win |
| C7 ConvTasNetAVSE (mask) (p2-c7-hq) | 4.89 | 1.683 | 0.718 | 0.017 | 308,544 | best of 20ep / 40000 win |
| C7 ConvTasNetAVSE (mask) (p2-c7-r1) | 3.79 | 1.565 | 0.690 | 0.017 | 308,544 | best of 10ep / 10000 win |