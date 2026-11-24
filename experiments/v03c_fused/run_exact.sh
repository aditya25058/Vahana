#!/bin/bash
# v0.3-c EXACT + UNIT bars on one GPU (run from the repo root: gpurun -g 1 bash experiments/v03c_fused/run_exact.sh PY MODELS)
PY=$1; M=$2; O=experiments/v03c_fused
export VAHANA_THREADS=24 OMP_NUM_THREADS=1 PYTHONPATH=src
$PY $O/v3c_unit.py $O/unit_h100.json 2>&1 | grep -v Warning | tail -3
for f in - 0.5 1.0; do
  A=""; [ "$f" != "-" ] && A="--gpu-cache $f"
  echo "selftest cache=$f: $($PY -m vahana.selftest --device cuda $A 2>&1 | grep -v Warning | tail -1)" | tee -a $O/selftest_h100.txt
done
$PY experiments/v03b_spec/v3b_tiny.py $O/tiny_spec_cuda.json cuda - > $O/tiny_spec_cuda.log 2>&1; tail -1 $O/tiny_spec_cuda.log | cut -c1-300
$PY experiments/v03b_spec/v3b_tiny.py $O/tiny_spec_cuda_cache05.json cuda 0.5 > $O/tiny_spec_cuda_cache05.log 2>&1; tail -1 $O/tiny_spec_cuda_cache05.log | cut -c1-300
$PY experiments/v03_speed/v3_run.py $M/OLMoE-1B-7B-0924 experiments/m5_real/prompts.json $O/real_olmoe.json 50 4 24 cuda - 16 > $O/real_olmoe.log 2>&1
$PY experiments/v03_speed/v3_run.py $M/Qwen3-30B-A3B experiments/m5_real/prompts.json $O/real_qwen3.json 8 2 16 cuda - 16 > $O/real_qwen3.log 2>&1   # 16: the P4 reference length
$PY - <<'P'
import json
O = "experiments/v03c_fused"
for m in ("olmoe", "qwen3"):
    a, r = json.load(open(f"{O}/real_{m}.json")), json.load(open(f"experiments/p4_doctor_stream/ref_{m}_h100.json"))
    pre = sum(x == y for x, y in zip(a["prefill"], r["prefill"]))
    gen = sum(x["ids"] == y["ids"] and x["logits"] == y["logits"] for x, y in zip(a["gens"], r["gens"]))
    print(f"REAL {m}: prefill {pre}/{len(r['prefill'])}, gens {gen}/{len(r['gens'])}, tok_s",
          [round(g["tok_s"], 1) for g in a["gens"]])
P
echo EXACT-DONE
