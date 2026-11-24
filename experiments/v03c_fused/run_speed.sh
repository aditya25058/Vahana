#!/bin/bash
# v0.3-c SPEED bars, alternated against the v0.3-b code (85216b8 worktree at BASE), same session, same GPU.
# usage (repo root): gpurun -g 1 bash experiments/v03c_fused/run_speed.sh PY MODELS BASE
PY=$1; M=$2; BASE=$3; O=$PWD/experiments/v03c_fused; P=$PWD/experiments/m5_real/prompts.json
export VAHANA_THREADS=24 OMP_NUM_THREADS=1
for r in 1 2; do
  for tag in base new; do
    D=$PWD; [ $tag = base ] && D=$BASE
    (cd $D && PYTHONPATH=src $PY $O/../v03_speed/v3_run.py $M/OLMoE-1B-7B-0924 $P $O/speed_olmoe_${tag}_r$r.json 1 4 64 cuda - 16 > $O/speed_olmoe_${tag}_r$r.log 2>&1)
    (cd $D && PYTHONPATH=src $PY $O/../v03_speed/v3_run.py $M/Qwen3-30B-A3B $P $O/speed_qwen3_${tag}_r$r.json 1 4 64 cuda - 18 > $O/speed_qwen3_${tag}_r$r.log 2>&1)
  done
done
PYTHONPATH=src $PY experiments/v03b_spec/v3b_real.py $M/OLMoE-1B-7B-0924 $P $O/spec_olmoe.json 16 > $O/spec_olmoe.log 2>&1
(cd $BASE && PYTHONPATH=src $PY $O/../v03b_spec/v3b_real.py $M/OLMoE-1B-7B-0924 $P $O/spec_olmoe_base.json 16 > $O/spec_olmoe_base.log 2>&1)
$PY - <<'Q'
import json, statistics as st
O = "experiments/v03c_fused"
for m in ("olmoe", "qwen3"):
    t = {tag: [g["tok_s"] for r in (1, 2) for g in json.load(open(f"{O}/speed_{m}_{tag}_r{r}.json"))["gens"]] for tag in ("base", "new")}
    print(f"SPEED {m}: base {st.mean(t['base']):.2f} new {st.mean(t['new']):.2f} tok/s -> {st.mean(t['new'])/st.mean(t['base']):.3f}x")
for f in ("spec_olmoe", "spec_olmoe_base"):
    d = json.load(open(f"{O}/{f}.json"))
    print(f.upper(), {k: d[k] for k in ("exact", "copy_speedup", "copy_plain_tok_s", "copy_spec_tok_s", "m5_speedup", "m5_plain_tok_s", "m5_spec_tok_s")})
Q
echo SPEED-DONE
