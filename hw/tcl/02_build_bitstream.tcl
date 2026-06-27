# 02_build_bitstream.tcl — synth + impl + bitstream for the static AVSE system, export PYNQ overlay.
# Run AFTER 01_build_bd.tcl:  vivado -mode batch -source hw/tcl/02_build_bitstream.tcl
open_project [pwd]/hw/vivado_proj/avse_sys.xpr

generate_target all [get_files *avse_sys.bd]
export_ip_user_files -of_objects [get_files *avse_sys.bd] -no_script -sync -force -quiet

# ---- synthesis ----
reset_run synth_1
launch_runs synth_1 -jobs 8
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] != "100%"} { error "synth_1 failed: [get_property STATUS [get_runs synth_1]]" }
puts "OK: synth_1 complete"

# ---- implementation + bitstream ----
launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] != "100%"} { error "impl_1 failed: [get_property STATUS [get_runs impl_1]]" }
puts "OK: impl_1 + bitstream complete"

# ---- reports ----
open_run impl_1
report_utilization  -file [pwd]/hw/vivado_proj/util_impl.txt
report_timing_summary -file [pwd]/hw/vivado_proj/timing_impl.txt

# ---- export PYNQ overlay (.bit + .hwh) ----
file mkdir [pwd]/hw/overlay
set bit [glob -nocomplain [pwd]/hw/vivado_proj/avse_sys.runs/impl_1/*_wrapper.bit]
set hwh [glob -nocomplain [pwd]/hw/vivado_proj/avse_sys.gen/sources_1/bd/avse_sys/hw_handoff/avse_sys.hwh]
if {$bit ne ""} { file copy -force [lindex $bit 0] [pwd]/hw/overlay/avse_sys.bit; puts "bit -> hw/overlay/avse_sys.bit" }
if {$hwh ne ""} { file copy -force [lindex $hwh 0] [pwd]/hw/overlay/avse_sys.hwh; puts "hwh -> hw/overlay/avse_sys.hwh" }

# timing one-liner
set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
puts "WNS = $wns ns"
puts "DONE: bitstream + overlay exported"
