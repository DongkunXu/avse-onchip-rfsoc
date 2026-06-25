# run_impl_c7.tcl — Vivado place-and-route of the C7 IP via the HLS impl flow.
# Reuses the synthesized solution and runs real Vivado synth + implementation to get post-route
# (place-and-route) utilization + timing. Run from repo root AFTER run_csynth_c7.tcl:
#   "D:/Xilinx/Vitis_HLS/2022.2/bin/vitis_hls.bat" -f hls/tcl/run_impl_c7.tcl
cd hls/build
open_project c7_fit
open_solution sol1
export_design -flow impl -rtl verilog
exit
