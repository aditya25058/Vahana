#!/bin/bash
# v0.4-a bars (repo root): gpurun -g 1 bash experiments/v04a_cpurows/run_bars.sh PY MODELS BASE
PY=$1; M=$2; BASE=$3; O=$PWD/experiments/v04a_cpurows; P=$PWD/experiments/m5_real/prompts.json; E=$PWD/experiments
export VAHANA_THREADS=24 OMP_NUM_THREADS=1 PYTHONPATH=src
for isa in avx512 avx2 scalar; do echo "cpu $isa: $(VAHANA_ISA=$isa $PY -m vahana.selftest --device cpu 2>&1 | grep -v Warning | tail -1)" | tee -a $O/selftest_h100.txt; done
for f in - 0.5 1.0; do A=""; [ "$f" != "-" ] && A="--gpu-cache $f"; echo "cuda cache=$f: $($PY -m vahana.selftest --device cuda $A 2>&1 | grep -v Warning | tail -1)" | tee -a $O/selftest_h100.txt; done
for cfg in "0.5 24" "0.5 48" "- 48"; do set -- $cfg; n=tiny_c${1}_b$2
  $PY $E/v03e_batch/v3e_tiny.py $O/$n.json cuda $1 $2 > $O/$n.log 2>&1; echo "tiny cache=$1 B=$2 $(tail -1 $O/$n.log)"; done
$PY $E/v03_speed/v3_run.py $M/OLMoE-1B-7B-0924 $P $O/hash_olmoe.json 50 4 24 cuda - 16 > $O/hash_olmoe.log 2>&1
$PY $E/v03_speed/v3_run.py $M/Qwen3-30B-A3B $P $O/hash_qwen3.json 8 2 16 cuda - 16 > $O/hash_qwen3.log 2>&1
$PY - <<'Q'
import json
O = "experiments/v04a_cpurows"
for m in ("olmoe", "qwen3"):
    a, r = json.load(open(f"{O}/hash_{m}.json")), json.load(open(f"experiments/p4_doctor_stream/ref_{m}_h100.json"))
    print(f"HASH {m}: prefill", sum(x == y for x, y in zip(a["prefill"], r["prefill"])), "/", len(r["prefill"]), "gens",
          sum(x["ids"] == y["ids"] and x["logits"] == y["logits"] for x, y in zip(a["gens"], r["gens"])), "/", len(r["gens"]))
Q
$PY $O/v4a_real.py $M/Qwen3-30B-A3B $P $O/exact_qwen3_b32.json 18 exact > $O/exact_qwen3_b32.log 2>&1; grep EXACT $O/exact_qwen3_b32.log
for r in 1 2; do for tag in new base; do D=$PWD; [ $tag = base ] && D=$BASE
  (cd $D && PYTHONPATH=src $PY $O/v4a_real.py $M/Qwen3-30B-A3B $P $O/speed_qwen3_${tag}_r$r.json 18 speed > $O/speed_qwen3_${tag}_r$r.log 2>&1); echo "$tag r$r: $(grep -o 'B=[0-9]*: [0-9.]* tok/s' $O/speed_qwen3_${tag}_r$r.log | tr '\n' ' ')"
done; done
PYTHONPATH=src $PY $O/v4a_real.py $M/OLMoE-1B-7B-0924 $P $O/speed_olmoe_new.json 16 speed > $O/speed_olmoe_new.log 2>&1; echo "olmoe new: $(grep -o 'B=[0-9]*: [0-9.]* tok/s' $O/speed_olmoe_new.log | tr '\n' ' ')"
echo BARS-DONE
