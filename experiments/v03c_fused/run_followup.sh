#!/bin/bash
# Follow-up (after the counted runs): 2-D grouped tiles with group size 2 below 32 tokens. Exactness again, then
# OLMoE plain decode alternated with 85216b8 and speculative decoding in the same session.
PY=$1; M=$2; BASE=$3; O=$PWD/experiments/v03c_fused; P=$PWD/experiments/m5_real/prompts.json
export VAHANA_THREADS=24 OMP_NUM_THREADS=1
PYTHONPATH=src $PY $O/v3c_unit.py $O/fu_unit_h100.json 2>&1 | grep -v Warning | tail -1
for f in - 1.0; do A=""; [ "$f" != "-" ] && A="--gpu-cache $f"; echo "selftest cache=$f: $(PYTHONPATH=src $PY -m vahana.selftest --device cuda $A 2>&1 | grep -v Warning | tail -1)"; done
PYTHONPATH=src $PY experiments/v03b_spec/v3b_tiny.py $O/fu_tiny_spec_cuda.json cuda - > $O/fu_tiny_spec_cuda.log 2>&1; tail -1 $O/fu_tiny_spec_cuda.log | cut -c1-200
for r in 1 2; do for tag in base new; do D=$PWD; [ $tag = base ] && D=$BASE
  (cd $D && PYTHONPATH=src $PY $O/../v03_speed/v3_run.py $M/OLMoE-1B-7B-0924 $P $O/fu_speed_olmoe_${tag}_r$r.json 1 4 64 cuda - 16 > $O/fu_speed_olmoe_${tag}_r$r.log 2>&1)
done; done
PYTHONPATH=src $PY experiments/v03b_spec/v3b_real.py $M/OLMoE-1B-7B-0924 $P $O/fu_spec_olmoe.json 16 > $O/fu_spec_olmoe.log 2>&1
$PY - <<'Q'
import json, statistics as st
O = "experiments/v03c_fused"
t = {tag: [g["tok_s"] for r in (1, 2) for g in json.load(open(f"{O}/fu_speed_olmoe_{tag}_r{r}.json"))["gens"]] for tag in ("base", "new")}
print(f"FU SPEED olmoe: base {st.mean(t['base']):.2f} new {st.mean(t['new']):.2f} -> {st.mean(t['new'])/st.mean(t['base']):.3f}x")
d = json.load(open(f"{O}/fu_spec_olmoe.json"))
print("FU SPEC", {k: d[k] for k in ("exact", "copy_speedup", "copy_plain_tok_s", "copy_spec_tok_s", "m5_speedup", "m5_plain_tok_s", "m5_spec_tok_s")})
Q
echo FU-DONE
