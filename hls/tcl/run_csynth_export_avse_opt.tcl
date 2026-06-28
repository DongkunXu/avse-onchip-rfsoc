# run_csynth_export_avse_opt.tcl — csynth + package the OPTIMIZED monolithic AVSE IP in ONE run
# (project c7_avse_opt). The optimized source (O-1..O-3c) lives in the shared headers. ~2.5 h csynth
# (front-end analysis of the wide unrolls — slowness, not a bug, D-19/O-2b) + ~10 min export.
# Launch DETACHED (Start-Process) so a session boundary cannot kill it; poll the report/flag.
#   "D:/Xilinx/Vitis_HLS/2022.2/bin/vitis_hls.bat" -f hls/tcl/run_csynth_export_avse_opt.tcl
file mkdir hls/build
cd hls/build
open_project -reset c7_avse_opt
set_top c7_avse_top
add_files ../src/c7_avse_top.cpp -cflags "-I../src"
open_solution -reset sol1
set_part {xczu48dr-ffvg1517-2-e}
create_clock -period 5 -name default
csynth_design
export_design -rtl verilog -format ip_catalog
exit
