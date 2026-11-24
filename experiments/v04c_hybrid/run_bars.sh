#!/bin/bash
# v0.4-c bars on one GPU (repo root): gpurun -g 1 bash experiments/v04c_hybrid/run_bars.sh PY
PY=$1; O=experiments/v04c_hybrid
export VAHANA_THREADS=24 OMP_NUM_THREADS=1 PYTHONPATH=src
for cfg in "cuda -" "cuda 0.5" "cpu -"; do set -- $cfg; n=hybrid_h100_$1_c$2
  $PY $O/v4c_tiny.py $O/$n.json $1 $2 > $O/$n.log 2>&1; echo "hybrid $1 cache=$2 $(tail -1 $O/$n.log)"; done
for B in 4 8; do $PY experiments/v03e_batch/v3e_tiny.py $O/batch_b$B.json cuda - $B > $O/batch_b$B.log 2>&1; echo "generate_batch B=$B $(tail -1 $O/batch_b$B.log)"
  grep -c "'batched': False" $O/batch_b$B.log | sed 's/^/  rows not batched (families): /'; done
$PY experiments/v04b_serve/v4b_tiny.py $O/regress_engine.json cuda - > $O/regress_engine.log 2>&1; echo "engine regression $(tail -1 $O/regress_engine.log)"
echo "selftest $($PY -m vahana.selftest --device cuda 2>&1 | grep -v Warning | tail -1)"
echo BARS-DONE
