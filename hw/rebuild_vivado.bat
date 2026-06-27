@echo off
set XILINXD_LICENSE_FILE=G:/phD_Projects/LICENSE_FOR_ISE_VIVADO.lic
cd /d G:\phD_Projects\AVSE-OnChip-RFSoC
call "D:/Xilinx/Vivado/2022.2/bin/vivado.bat" -mode batch -source hw/tcl/01_build_bd.tcl -nojournal -notrace -log hw/rb_bd.log
if errorlevel 1 ( echo BD_FAILED > hw/rebuild_done.flag & exit /b 1 )
call "D:/Xilinx/Vivado/2022.2/bin/vivado.bat" -mode batch -source hw/tcl/02_build_bitstream.tcl -nojournal -notrace -log hw/rb_bit.log
echo REBUILD_DONE > hw/rebuild_done.flag
