#!/bin/bash
# After graph-mode host jobs were chunked by 16 rows (B > 16 had hit an assert): B=24 tiny exactness, then the real
# harness again (exactness + the full B = 1..64 curve). (repo root): gpurun -g 1 bash experiments/v03e_batch/run_rerun.sh PY MODELS
PY=$1; M=$2; O=experiments/v03e_batch; P=experiments/m5_real/prompts.json
export VAHANA_THREADS=24 OMP_NUM_THREADS=1 PYTHONPATH=src
$PY $O/v3e_tiny.py $O/tiny_cuda_cache05_b24.json cuda 0.5 24 > $O/tiny_cuda_cache05_b24.log 2>&1; echo "B=24 cache0.5 $(tail -1 $O/tiny_cuda_cache05_b24.log)"
$PY $O/v3e_real.py $M/OLMoE-1B-7B-0924 $P $O/real_olmoe.json 16 > $O/real_olmoe.log 2>&1; grep -E "EXACT|round 2|SCALE" $O/real_olmoe.log
$PY $O/v3e_real.py $M/Qwen3-30B-A3B $P $O/real_qwen3.json 18 > $O/real_qwen3.log 2>&1; grep -E "EXACT|round 2|SCALE" $O/real_qwen3.log
echo RERUN-DONE
