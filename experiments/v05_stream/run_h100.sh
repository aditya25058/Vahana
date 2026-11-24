#!/bin/bash
# v0.5 H100 bars (repo root): gpurun -g 1 bash experiments/v05_stream/run_h100.sh PY MODELS
PY=$1; M=$2; O=experiments/v05_stream; P=experiments/m5_real/prompts.json
export VAHANA_THREADS=24 OMP_NUM_THREADS=1 PYTHONPATH=src
for isa in avx512 avx2 scalar; do echo "cpu $isa prefetch=2: $(VAHANA_PREFETCH=2 VAHANA_ISA=$isa $PY -m vahana.selftest --device cpu 2>&1 | grep -v Warning | tail -1)"; done
for f in - 0.5; do A=""; [ "$f" != "-" ] && A="--gpu-cache $f"; echo "cuda cache=$f prefetch=2: $(VAHANA_PREFETCH=2 $PY -m vahana.selftest --device cuda $A 2>&1 | grep -v Warning | tail -1)"; done
# RAM-resident: prefetch must cost nothing (OLMoE, all experts on the CPU, no GPU cache), alternated 0 2 / 2 0
for r in 1 2; do for mode in $( [ $r = 1 ] && echo "0 2" || echo "2 0" ); do
  VAHANA_PREFETCH=$mode $PY $O/v5_stream.py $M/OLMoE-1B-7B-0924 $P $O/h100_olmoe_m${mode}_r$r.json - 32 4 2>&1 | grep SUMMARY | sed "s/^/olmoe mode $mode r$r: /" | cut -c1-240
done; done
$PY $O/v5_doctor.py $M/Qwen3-30B-A3B $P $O/doctor_h100_qwen3.json > $O/doctor_h100_qwen3.log 2>&1; grep DOCTOR $O/doctor_h100_qwen3.log
echo BARS-DONE
