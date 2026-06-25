# run_csynth_c7.tcl — Vitis HLS 2022.2 C-synth of the C7 audio mask network.
# Target: Real Digital RFSoC 4x2 / ZU48DR. Run from repo root:
#   "D:/Xilinx/Vitis_HLS/2022.2/bin/vitis_hls.bat" -f hls/tcl/run_csynth_c7.tcl
open_project -reset hls/build/c7_fit
set_top c7_audio_top
add_files hls/src/c7_audio_top.cpp -cflags "-Ihls/src"
open_solution -reset sol1
set_part {xczu48dr-ffvg1517-2-e}
create_clock -period 5 -name default
csynth_design
exit
