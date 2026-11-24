#!/bin/bash
# v0.3-e bars on one GPU (repo root): gpurun -g 1 bash experiments/v03e_batch/run_bars.sh PY MODELS
PY=$1; M=$2; O=experiments/v03e_batch; P=experiments/m5_real/prompts.json
export VAHANA_THREADS=24 OMP_NUM_THREADS=1 PYTHONPATH=src
for B in 4 8; do
  $PY $O/v3e_tiny.py $O/tiny_cuda_b$B.json cuda - $B > $O/tiny_cuda_b$B.log 2>&1; echo "B=$B $(tail -1 $O/tiny_cuda_b$B.log)"
  $PY $O/v3e_tiny.py $O/tiny_cuda_cache05_b$B.json cuda 0.5 $B > $O/tiny_cuda_cache05_b$B.log 2>&1; echo "B=$B cache0.5 $(tail -1 $O/tiny_cuda_cache05_b$B.log)"
done
$PY $O/v3e_real.py $M/OLMoE-1B-7B-0924 $P $O/real_olmoe.json 16 > $O/real_olmoe.log 2>&1; grep -E "EXACT|SCALE" $O/real_olmoe.log
$PY $O/v3e_real.py $M/Qwen3-30B-A3B $P $O/real_qwen3.json 18 > $O/real_qwen3.log 2>&1; grep -E "EXACT|SCALE" $O/real_qwen3.log
echo BARS-DONE
