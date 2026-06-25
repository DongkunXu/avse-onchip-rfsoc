# hw/ — Phase 3: Vivado block design, bitstream, board bring-up

**Placeholder.** Populated only for Phase-3 finalists. See [`../docs/ROADMAP.md`](../docs/ROADMAP.md).

What will live here: Vivado 2024.2 block-design + bitstream `.tcl`, and the board harness for the
Real Digital RFSoC 4x2 (ZU48DR). The headline goal — *the empirical core of the circuits paper* — is a
**single static bitstream** whose post-place-and-route reports show the whole AVSE pipeline co-resident
(vs the reference 4-way-reconfig baseline at 215 % BRAM).

Board access (from [`../ENVIRONMENT.md`](../ENVIRONMENT.md)): `ssh xilinx@172.26.206.133` (pw `xilinx`,
DHCP — verify IP); PYNQ env via `/etc/profile.d/{pynq_venv,xrt_setup}.sh`; PL ops need `sudo`.
Device budget: [`../docs/reference/hardware-budget.md`](../docs/reference/hardware-budget.md).
