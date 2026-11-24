#!/bin/bash
# v0.4-a: the runs after the preemption (Qwen3 B=32 exactness, speed) (repo root): gpurun -g 1 bash experiments/v04a_cpurows/run_rest.sh PY MODELS BASE
PY=$1; M=$2; BASE=$3; O=$PWD/experiments/v04a_cpurows; P=$PWD/experiments/m5_real/prompts.json; E=$PWD/experiments
export VAHANA_THREADS=24 OMP_NUM_THREADS=1 PYTHONPATH=src
$PY $O/v4a_real.py $M/Qwen3-30B-A3B $P $O/exact_qwen3_b32.json 18 exact > $O/exact_qwen3_b32.log 2>&1; grep EXACT $O/exact_qwen3_b32.log
for r in 1 2; do for tag in new base; do D=$PWD; [ $tag = base ] && D=$BASE
  (cd $D && PYTHONPATH=src $PY $O/v4a_real.py $M/Qwen3-30B-A3B $P $O/speed_qwen3_${tag}_r$r.json 18 speed > $O/speed_qwen3_${tag}_r$r.log 2>&1); echo "$tag r$r: $(grep -o 'B=[0-9]*: [0-9.]* tok/s' $O/speed_qwen3_${tag}_r$r.log | tr '\n' ' ')"
done; done
PYTHONPATH=src $PY $O/v4a_real.py $M/OLMoE-1B-7B-0924 $P $O/speed_olmoe_new.json 16 speed > $O/speed_olmoe_new.log 2>&1; echo "olmoe new: $(grep -o 'B=[0-9]*: [0-9.]* tok/s' $O/speed_olmoe_new.log | tr '\n' ' ')"
echo BARS-DONE
