# run_impl_avse.tcl — Vivado place-and-route of the MONOLITHIC C7 AVSE IP via the HLS impl flow.
# Real post-route whole-system single-config numbers. Run from repo root AFTER run_csynth_avse.tcl:
#   "D:/Xilinx/Vitis_HLS/2022.2/bin/vitis_hls.bat" -f hls/tcl/run_impl_avse.tcl
cd hls/build
open_project c7_avse
open_solution sol1
export_design -flow impl -rtl verilog
exit
