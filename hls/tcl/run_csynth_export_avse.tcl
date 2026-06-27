# run_csynth_export_avse.tcl — fresh csynth of the monolithic AVSE + package the IP (for the Vivado bitstream).
# Use after editing the HLS sources (e.g. the decoder hazard fix). ~10-13 min (no P&R; the Vivado flow P&Rs).
#   "D:/Xilinx/Vitis_HLS/2022.2/bin/vitis_hls.bat" -f hls/tcl/run_csynth_export_avse.tcl
file mkdir hls/build
cd hls/build
open_project -reset c7_avse
set_top c7_avse_top
add_files ../src/c7_avse_top.cpp -cflags "-I../src"
open_solution -reset sol1
set_part {xczu48dr-ffvg1517-2-e}
create_clock -period 5 -name default
csynth_design
export_design -rtl verilog -format ip_catalog
exit
