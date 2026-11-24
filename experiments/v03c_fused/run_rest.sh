#!/bin/bash
# Qwen3 EXACT rerun at the reference length (16 new tokens), then the SPEED bars.
PY=$1; M=$2; BASE=$3; O=experiments/v03c_fused
export VAHANA_THREADS=24 OMP_NUM_THREADS=1
PYTHONPATH=src $PY experiments/v03_speed/v3_run.py $M/Qwen3-30B-A3B experiments/m5_real/prompts.json $O/real_qwen3.json 8 2 16 cuda - 16 > $O/real_qwen3.log 2>&1
$PY - <<'P'
import json
a, r = json.load(open("experiments/v03c_fused/real_qwen3.json")), json.load(open("experiments/p4_doctor_stream/ref_qwen3_h100.json"))
print("REAL qwen3: prefill", sum(x == y for x, y in zip(a["prefill"], r["prefill"])), "/", len(r["prefill"]), "gens",
      sum(x["ids"] == y["ids"] and x["logits"] == y["logits"] for x, y in zip(a["gens"], r["gens"])), "/", len(r["gens"]))
P
bash $O/run_speed.sh $PY $M $BASE
