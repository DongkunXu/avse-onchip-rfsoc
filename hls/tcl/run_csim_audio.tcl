# run_csim_audio.tcl — Vitis HLS 2022.2 C-simulation cross-check of the C7 audio core against the
# fixed-point emulator (golden vectors from tools/dump_hls_vectors.py). Run from repo root:
#   "D:/Xilinx/Vitis_HLS/2022.2/bin/vitis_hls.bat" -f hls/tcl/run_csim_audio.tcl
file mkdir hls/build
cd hls/build
open_project -reset c7_audio_csim
set_top c7_audio_top
add_files ../src/c7_audio_top.cpp -cflags "-I../src"
add_files -tb ../tb/tb_audio_core.cpp -cflags "-I../src"
open_solution -reset sol1
set_part {xczu48dr-ffvg1517-2-e}
create_clock -period 5 -name default
csim_design
exit
