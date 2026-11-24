#!/bin/bash
# Run a command on physical GPU 1 only, inside a gpurun -g 2 reservation (GPU 0 is busy outside the scheduler).
# Aborts unless GPU 1 is empty (< 1 GB used) at start.
export CUDA_VISIBLE_DEVICES=1
used=$(<work>/venv/bin/python -c "import torch; f,t=torch.cuda.mem_get_info(0); print(round((t-f)/2**30,2))")
echo "GPU1 used at start: $used GB"
awk -v u="$used" "BEGIN{exit !(u < 1.0)}" || { echo "GPU1 not empty: abort"; exit 3; }
exec "$@"
