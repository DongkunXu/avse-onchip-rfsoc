# run_csynth_c7_opt.tcl — fast standalone C-synth of the C7 AUDIO core during the throughput-
# optimization phase (Phase 4, O-3). Separate project (c7_audio_opt) keeps the baseline intact.
# The audio core synthesizes in ~3 min, so O-3 is tuned here, then integrated into the monolith.
# Run from repo root:
#   "D:/Xilinx/Vitis_HLS/2022.2/bin/vitis_hls.bat" -f hls/tcl/run_csynth_c7_opt.tcl
file mkdir hls/build
cd hls/build
open_project -reset c7_audio_opt
set_top c7_audio_top
add_files ../src/c7_audio_top.cpp -cflags "-I../src"
open_solution -reset sol1
set_part {xczu48dr-ffvg1517-2-e}
create_clock -period 5 -name default
csynth_design
exit
