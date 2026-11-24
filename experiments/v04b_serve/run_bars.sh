#!/bin/bash
# v0.4-b bars on one GPU (repo root): gpurun -g 1 bash experiments/v04b_serve/run_bars.sh PY MODELS
PY=$1; M=$2; O=experiments/v04b_serve; P=experiments/m5_real/prompts.json
export VAHANA_THREADS=24 OMP_NUM_THREADS=1 PYTHONPATH=src
for cfg in "cuda -" "cuda 0.5" "cpu -"; do set -- $cfg; n=tiny_h100_$1_c$2
  $PY $O/v4b_tiny.py $O/$n.json $1 $2 > $O/$n.log 2>&1; echo "tiny $1 cache=$2 $(tail -1 $O/$n.log)"; done
$PY $O/v4b_real.py $M/OLMoE-1B-7B-0924 $P $O/exact_olmoe.json 16 exact > $O/exact_olmoe.log 2>&1; grep EXACT $O/exact_olmoe.log | cut -c1-300
$PY $O/v4b_real.py $M/Qwen3-30B-A3B $P $O/exact_qwen3.json 18 exact > $O/exact_qwen3.log 2>&1; grep EXACT $O/exact_qwen3.log | cut -c1-300
$PY $O/v4b_http.py $M/OLMoE-1B-7B-0924 $P $O/http_olmoe.json 16 8 > $O/http_olmoe.log 2>&1; grep -E "HTTP|Error" $O/http_olmoe.log | tail -2
$PY $O/v4b_real.py $M/OLMoE-1B-7B-0924 $P $O/speed_olmoe.json 16 speed > $O/speed_olmoe.log 2>&1; grep -E "round|SPEED" $O/speed_olmoe.log
$PY $O/v4b_real.py $M/Qwen3-30B-A3B $P $O/speed_qwen3.json 18 speed > $O/speed_qwen3.log 2>&1; grep -E "round|SPEED" $O/speed_qwen3.log
echo BARS-DONE
