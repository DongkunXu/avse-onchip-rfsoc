# run_full_avse.tcl — fresh csynth + Vivado P&R of the monolithic C7 AVSE in ONE clean run.
# Use this (not the split csynth/impl tcls) when a prior impl was interrupted/deleted, so there is
# no stale solution state. Run ALONE (Vivado P&R is memory-heavy on this 32 GB host). From repo root:
#   "D:/Xilinx/Vitis_HLS/2022.2/bin/vitis_hls.bat" -f hls/tcl/run_full_avse.tcl
file mkdir hls/build
cd hls/build
open_project -reset c7_avse
set_top c7_avse_top
add_files ../src/c7_avse_top.cpp -cflags "-I../src"
open_solution -reset sol1
set_part {xczu48dr-ffvg1517-2-e}
create_clock -period 5 -name default
csynth_design
export_design -flow impl -rtl verilog
exit
