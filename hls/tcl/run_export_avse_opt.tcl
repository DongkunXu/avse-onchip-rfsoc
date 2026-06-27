# run_export_avse_opt.tcl — package the OPTIMIZED monolithic AVSE IP for the Vivado bitstream,
# REUSING the already-completed csynth in c7_avse_opt (open WITHOUT -reset -> no 2.5 h re-synth).
# Run AFTER run_csynth_avse_opt.tcl has finished:
#   "D:/Xilinx/Vitis_HLS/2022.2/bin/vitis_hls.bat" -f hls/tcl/run_export_avse_opt.tcl
cd hls/build
open_project c7_avse_opt
open_solution sol1
export_design -rtl verilog -format ip_catalog
exit
