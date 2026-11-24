#!/bin/bash
# v0.5 laptop SPEED + EXACT: Qwen3-30B streamed, prefetch modes alternated (0 2 1 / 1 2 0), 3 prompts x 24 tokens.
# usage: run_laptop.sh PY PKG MODEL PROMPTS OUTDIR
PY=$1; PKG=$2; M=$3; P=$4; O=$5
for r in 1 2; do
  for mode in $( [ $r = 1 ] && echo "0 2 1" || echo "1 2 0" ); do
    VAHANA_PREFETCH=$mode PYTHONPATH=$PKG $PY "$O/v5_stream.py" "$M" "$P" "$O/laptop_qwen3_m${mode}_r$r.json" auto 24 3 2>&1 | grep SUMMARY | sed "s/^/mode $mode r$r: /" | cut -c1-260
  done
done
