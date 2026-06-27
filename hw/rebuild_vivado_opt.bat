@echo off
REM rebuild_vivado_opt.bat — build the OPTIMIZED (Phase 4) AVSE bitstream from the c7_avse_opt IP.
REM Same as rebuild_vivado.bat but uses 01_build_bd_opt.tcl (ip_repo -> c7_avse_opt). Run alone (D-11).
set XILINXD_LICENSE_FILE=G:/phD_Projects/LICENSE_FOR_ISE_VIVADO.lic
cd /d G:\phD_Projects\AVSE-OnChip-RFSoC
call "D:/Xilinx/Vivado/2022.2/bin/vivado.bat" -mode batch -source hw/tcl/01_build_bd_opt.tcl -nojournal -notrace -log hw/rb_bd_opt.log
if errorlevel 1 ( echo BD_FAILED > hw/rebuild_done.flag & exit /b 1 )
call "D:/Xilinx/Vivado/2022.2/bin/vivado.bat" -mode batch -source hw/tcl/02_build_bitstream.tcl -nojournal -notrace -log hw/rb_bit_opt.log
echo REBUILD_DONE > hw/rebuild_done.flag
