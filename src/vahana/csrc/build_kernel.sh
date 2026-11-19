#!/bin/bash
# Build the Vahana CPU expert kernel (AVX-512 + VBMI; Zen 4 / Sapphire Rapids or newer).
# Usage: build_kernel.sh [python] [march]   -> src/vahana/_build/cpu_moe*.so
# -ffp-contract=off and -fno-fast-math are required: the twin arithmetic must match the GPU bit for bit.
set -euo pipefail
PY=${1:-python}
MARCH=${2:-native}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$HERE/../_build
mkdir -p "$OUT"
TP=$($PY -c "import torch,os;print(os.path.dirname(torch.__file__))")
PI=$($PY -c "import sysconfig;print(sysconfig.get_paths()['include'])")
ABI=$($PY -c "import torch;print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")
SUF=$($PY -c "import sysconfig;print(sysconfig.get_config_var('EXT_SUFFIX'))")
g++ -O3 -march=$MARCH -ffp-contract=off -fno-fast-math -fopenmp -shared -fPIC -std=c++17 \
  -DTORCH_EXTENSION_NAME=cpu_moe -D_GLIBCXX_USE_CXX11_ABI=$ABI \
  -I"$TP/include" -I"$TP/include/torch/csrc/api/include" -I"$PI" \
  "$HERE/cpu_moe.cpp" -o "$OUT/cpu_moe$SUF" \
  -L"$TP/lib" -lc10 -ltorch -ltorch_cpu -ltorch_python -lnuma -lpthread -Wl,-rpath,"$TP/lib"
echo "built $OUT/cpu_moe$SUF"
