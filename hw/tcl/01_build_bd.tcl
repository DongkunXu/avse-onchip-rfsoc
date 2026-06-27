# 01_build_bd.tcl — static single-config block design for the monolithic C7 AVSE IP on RFSoC 4x2.
# PS (zynq_ultra_ps_e) + the ONE HLS IP c7_avse_top. The IP masters DDR directly (3 m_axi: audio_in,
# video_in, audio_out) via one HP port; PS drives its s_axilite control via HPM0. PYNQ allocates DDR
# buffers, writes their physical addresses into the IP control regs, starts it, polls done, reads output.
# This is the SINGLE static bitstream that the reference (4 PCAP bitstreams) could not achieve.
# Run:  vivado -mode batch -source hw/tcl/01_build_bd.tcl   (from repo root, XILINXD_LICENSE_FILE set)

set part        "xczu48dr-ffvg1517-2-e"
set board       "realdigital.org:rfsoc4x2:part0:1.0"
set board_repo  "C:/Users/dongk/Downloads/rfsoc4x2_extracted"
set ip_repo     "[pwd]/hls/build/c7_avse/sol1/impl/ip"
set proj_dir    "[pwd]/hw/vivado_proj"

file delete -force $proj_dir
file mkdir $proj_dir
create_project avse_sys $proj_dir -part $part -force

# board part (RFSoC 4x2) — gives the correct PS DDR/clock preset; fall back to part-only if unavailable
if {[catch {set_property board_part_repo_paths $board_repo [current_project]} e]} { puts "WARN board_repo: $e" }
if {[catch {set_property board_part $board [current_project]} e]} { puts "WARN board_part: $e — part-only flow" }

# HLS IP catalog
set_property ip_repo_paths [list $ip_repo] [current_project]
update_ip_catalog -rebuild
update_ip_catalog

create_bd_design "avse_sys"

# ---- Zynq UltraScale+ PS ----
create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e zynq_ps
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e \
    -config { apply_board_preset "1" } [get_bd_cells zynq_ps]
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0 {1} \
    CONFIG.PSU__USE__M_AXI_GP1 {0} \
    CONFIG.PSU__USE__M_AXI_GP2 {0} \
    CONFIG.PSU__USE__S_AXI_GP2 {1} \
    CONFIG.PSU__SAXIGP2__DATA_WIDTH {128} \
    CONFIG.PSU__MAXIGP0__DATA_WIDTH {32} \
    CONFIG.PSU__FPGA_PL0_ENABLE {1} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {200} \
] [get_bd_cells zynq_ps]

# ---- clock + reset (PL_CLK0 = 200 MHz) ----
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset rst_200
connect_bd_net [get_bd_pins zynq_ps/pl_clk0]    [get_bd_pins rst_200/slowest_sync_clk]
connect_bd_net [get_bd_pins zynq_ps/pl_resetn0] [get_bd_pins rst_200/ext_reset_in]
connect_bd_net [get_bd_pins zynq_ps/pl_clk0]    [get_bd_pins zynq_ps/saxihp0_fpd_aclk]
connect_bd_net [get_bd_pins zynq_ps/pl_clk0]    [get_bd_pins zynq_ps/maxihpm0_fpd_aclk]

# ---- control path: PS HPM0 -> IP s_axi_control ----
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect axi_smc_ctrl
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] [get_bd_cells axi_smc_ctrl]
connect_bd_intf_net [get_bd_intf_pins zynq_ps/M_AXI_HPM0_FPD] [get_bd_intf_pins axi_smc_ctrl/S00_AXI]
connect_bd_net [get_bd_pins zynq_ps/pl_clk0]            [get_bd_pins axi_smc_ctrl/aclk]
connect_bd_net [get_bd_pins rst_200/peripheral_aresetn] [get_bd_pins axi_smc_ctrl/aresetn]

# ---- data path: IP 3x m_axi -> HP0 -> DDR ----
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect axi_smc_data
set_property -dict [list CONFIG.NUM_SI {3} CONFIG.NUM_MI {1}] [get_bd_cells axi_smc_data]
connect_bd_intf_net [get_bd_intf_pins axi_smc_data/M00_AXI] [get_bd_intf_pins zynq_ps/S_AXI_HP0_FPD]
connect_bd_net [get_bd_pins zynq_ps/pl_clk0]            [get_bd_pins axi_smc_data/aclk]
connect_bd_net [get_bd_pins rst_200/peripheral_aresetn] [get_bd_pins axi_smc_data/aresetn]

# ---- the AVSE IP ----
create_bd_cell -type ip -vlnv xilinx.com:hls:c7_avse_top:1.0 avse_0
connect_bd_net [get_bd_pins zynq_ps/pl_clk0]            [get_bd_pins avse_0/ap_clk]
connect_bd_net [get_bd_pins rst_200/peripheral_aresetn] [get_bd_pins avse_0/ap_rst_n]
connect_bd_intf_net [get_bd_intf_pins axi_smc_ctrl/M00_AXI] [get_bd_intf_pins avse_0/s_axi_control]
connect_bd_intf_net [get_bd_intf_pins avse_0/m_axi_gmem0] [get_bd_intf_pins axi_smc_data/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins avse_0/m_axi_gmem1] [get_bd_intf_pins axi_smc_data/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins avse_0/m_axi_gmem2] [get_bd_intf_pins axi_smc_data/S02_AXI]

# ---- address map (auto: s_axi_control into HPM0 space; gmem* into DDR) ----
assign_bd_address
validate_bd_design
save_bd_design

# ---- wrapper ----
set bd_file [get_files *avse_sys.bd]
make_wrapper -files $bd_file -top
add_files -norecurse [pwd]/hw/vivado_proj/avse_sys.gen/sources_1/bd/avse_sys/hdl/avse_sys_wrapper.v
set_property top avse_sys_wrapper [current_fileset]
update_compile_order -fileset sources_1
puts "OK: BD built + validated + wrapped"
