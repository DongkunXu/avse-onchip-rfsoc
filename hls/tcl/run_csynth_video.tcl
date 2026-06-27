# run_csynth_video.tcl — Vitis HLS 2022.2 C-synth of the standalone faithful video encoder (diagnostic).
#   "D:/Xilinx/Vitis_HLS/2022.2/bin/vitis_hls.bat" -f hls/tcl/run_csynth_video.tcl
file mkdir hls/build
cd hls/build
open_project -reset c7_video
set_top c7_video_top
add_files ../src/c7_video_top.cpp -cflags "-I../src"
open_solution -reset sol1
set_part {xczu48dr-ffvg1517-2-e}
create_clock -period 5 -name default
csynth_design
exit
