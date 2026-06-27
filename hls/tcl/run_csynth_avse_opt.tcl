# run_csynth_avse_opt.tcl — Vitis HLS 2022.2 C-synth of the MONOLITHIC C7 AVSE during the
# throughput-optimization phase (Phase 4). Uses a SEPARATE project (c7_avse_opt) so the baseline
# build (c7_avse) stays intact for A/B latency/resource comparison. Run from repo root:
#   "D:/Xilinx/Vitis_HLS/2022.2/bin/vitis_hls.bat" -f hls/tcl/run_csynth_avse_opt.tcl
file mkdir hls/build
cd hls/build
open_project -reset c7_avse_opt
set_top c7_avse_top
add_files ../src/c7_avse_top.cpp -cflags "-I../src"
open_solution -reset sol1
set_part {xczu48dr-ffvg1517-2-e}
create_clock -period 5 -name default
csynth_design
exit
