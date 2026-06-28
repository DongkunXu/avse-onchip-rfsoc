#!/usr/bin/env bash
# run_board_chunks.sh — drive the SNR-bin eval chunks through the optimized FPGA bitstream.
# For each chunk_*_windows.npz in LOCAL_DIR: scp to the board, run run_fpga.py, scp the outputs back.
# Resumable: skips a chunk whose *_outputs.npz already exists locally.
#   bash tools/run_board_chunks.sh <LOCAL_DIR> [REMOTE_SUBDIR]
set -u
LOCAL_DIR="${1:?usage: run_board_chunks.sh <local_dir> [remote_subdir]}"
REMOTE_SUB="${2:-snr}"
IP=172.26.206.133
HK="SHA256:vrFwqkqWIfyDZ1S66aJb1gLy59wx3LakOqhnyQJEOhg"
RBASE="/home/xilinx/avse_onchip"
RDIR="$RBASE/$REMOTE_SUB"
PL="plink -batch -hostkey $HK -pw xilinx xilinx@$IP"
PS="pscp -batch -hostkey $HK -pw xilinx"

$PL "mkdir -p $RDIR" || { echo "ssh failed"; exit 1; }
chunks=$(ls "$LOCAL_DIR"/chunk_*_windows.npz 2>/dev/null | sort)
[ -z "$chunks" ] && { echo "no chunks in $LOCAL_DIR"; exit 1; }
n=$(echo "$chunks" | wc -l); i=0
t0=$(date +%s)
for w in $chunks; do
  i=$((i+1)); base=$(basename "$w" _windows.npz); out="$LOCAL_DIR/${base}_outputs.npz"
  if [ -f "$out" ]; then echo "[$i/$n] $base outputs exist, skip"; continue; fi
  echo "[$i/$n] $base -> board ($(du -h "$w" | cut -f1))"
  $PS "$w" "xilinx@$IP:$RDIR/${base}_windows.npz" >/dev/null || { echo "  scp up FAILED"; exit 1; }
  $PL "echo xilinx | sudo -S bash -c 'source /etc/profile.d/pynq_venv.sh; source /etc/profile.d/xrt_setup.sh 2>/dev/null; cd $RBASE; python3 run_fpga.py --overlay avse_sys.bit --windows $RDIR/${base}_windows.npz --out $RDIR/${base}_outputs.npz'" 2>&1 | grep -aE "mean compute|loaded|Error|Traceback" | tail -3
  $PS "xilinx@$IP:$RDIR/${base}_outputs.npz" "$out" >/dev/null || { echo "  scp down FAILED"; exit 1; }
  $PL "rm -f $RDIR/${base}_windows.npz" >/dev/null 2>&1
  echo "  done ($(( $(date +%s) - t0 ))s elapsed)"
done
echo "ALL $n chunks done in $(( $(date +%s) - t0 ))s"
