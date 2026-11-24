#!/bin/bash
# v0.3-d UNIT, EXACT and SPEED bars on one GPU (repo root): gpurun -g 1 bash experiments/v03d_multilin/run_bars.sh PY MODELS BASE
PY=$1; M=$2; BASE=$3; O=$PWD/experiments/v03d_multilin; P=$PWD/experiments/m5_real/prompts.json
export VAHANA_THREADS=24 OMP_NUM_THREADS=1
PYTHONPATH=src $PY $O/v3d_unit.py $O/unit_h100.json 2>&1 | grep -v Warning | tail -1
for f in - 0.5 1.0; do A=""; [ "$f" != "-" ] && A="--gpu-cache $f"
  echo "selftest cache=$f: $(PYTHONPATH=src $PY -m vahana.selftest --device cuda $A 2>&1 | grep -v Warning | tail -1)" | tee -a $O/selftest_h100.txt; done
PYTHONPATH=src $PY experiments/v03b_spec/v3b_tiny.py $O/tiny_spec_cuda.json cuda - > $O/tiny_spec_cuda.log 2>&1; tail -1 $O/tiny_spec_cuda.log | cut -c1-200
PYTHONPATH=src $PY experiments/v03b_spec/v3b_tiny.py $O/tiny_spec_cuda_cache05.json cuda 0.5 > $O/tiny_spec_cuda_cache05.log 2>&1; tail -1 $O/tiny_spec_cuda_cache05.log | cut -c1-200
# OLMoE: exact hashes + prefill/decode speed, alternated with BASE (2 rounds)
for r in 1 2; do for tag in new base; do D=$PWD; [ $tag = base ] && D=$BASE
  (cd $D && PYTHONPATH=src $PY $O/../v03_speed/v3_run.py $M/OLMoE-1B-7B-0924 $P $O/olmoe_${tag}_r$r.json 50 4 24 cuda - 16 > $O/olmoe_${tag}_r$r.log 2>&1)
done; done
for tag in new base; do D=$PWD; [ $tag = base ] && D=$BASE
  (cd $D && PYTHONPATH=src $PY $O/../v03_speed/v3_run.py $M/Qwen3-30B-A3B $P $O/qwen3_${tag}.json 8 2 16 cuda - 16 > $O/qwen3_${tag}.log 2>&1)
done
for tag in new base; do D=$PWD; [ $tag = base ] && D=$BASE
  (cd $D && PYTHONPATH=src $PY $O/../v03b_spec/v3b_real.py $M/OLMoE-1B-7B-0924 $P $O/spec_olmoe_${tag}.json 16 > $O/spec_olmoe_${tag}.log 2>&1)
done
PYTHONPATH=src $PY experiments/v03b_spec/v3b_real.py $M/Qwen3-30B-A3B $P $O/spec_qwen3_new.json 18 > $O/spec_qwen3_new.log 2>&1
$PY - <<'Q'
import json, statistics as st
O = "experiments/v03d_multilin"
def eq(a, r):
    return (sum(x == y for x, y in zip(a["prefill"], r["prefill"])), len(r["prefill"]),
            sum(x["ids"] == y["ids"] and x["logits"] == y["logits"] for x, y in zip(a["gens"], r["gens"])), len(r["gens"]))
ro, rq = (json.load(open(f"experiments/p4_doctor_stream/ref_{m}_h100.json")) for m in ("olmoe", "qwen3"))
for r in (1, 2):
    print(f"REAL olmoe new r{r}: prefill %d/%d gens %d/%d" % eq(json.load(open(f"{O}/olmoe_new_r{r}.json")), ro))
print("REAL qwen3 new: prefill %d/%d gens %d/%d" % eq(json.load(open(f"{O}/qwen3_new.json")), rq))
L = lambda f: json.load(open(f"{O}/{f}.json"))
pre = {t: [L(f"olmoe_{t}_r{r}")["prefill_s"] for r in (1, 2)] for t in ("new", "base")}
dec = {t: [g["tok_s"] for r in (1, 2) for g in L(f"olmoe_{t}_r{r}")["gens"][1:]] for t in ("new", "base")}
print(f"SPEED olmoe prefill_s base {pre['base']} new {pre['new']} -> {st.mean(pre['base'])/st.mean(pre['new']):.3f}x")
print(f"SPEED olmoe decode base {st.mean(dec['base']):.2f} new {st.mean(dec['new']):.2f} -> {st.mean(dec['new'])/st.mean(dec['base']):.3f}x")
print(f"SPEED qwen3 prefill_s base {L('qwen3_base')['prefill_s']:.2f} new {L('qwen3_new')['prefill_s']:.2f} -> {L('qwen3_base')['prefill_s']/L('qwen3_new')['prefill_s']:.3f}x")
for f in ("spec_olmoe_new", "spec_olmoe_base", "spec_qwen3_new"):
    d = L(f); print(f.upper(), {k: d[k] for k in ("exact", "copy_speedup", "copy_plain_tok_s", "copy_spec_tok_s", "m5_speedup", "m5_plain_tok_s", "m5_spec_tok_s")})
Q
echo BARS-DONE
