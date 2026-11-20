#!/bin/bash
# Build the Vahana CPU kernel for one instruction set (all builds compute identical bits).
# Usage: build_kernel.sh [python] [isa]    isa: avx512 | avx2 | scalar | neon | all (default: all that apply here)
#   avx512  AVX-512 F/BW/VL/DQ + VBMI/VBMI2 (Ice Lake / Zen 4 and newer): fastest, NX12 SIMD decode
#   avx2    AVX2 + FMA (Haswell / Zen 1 and newer): most PCs since 2013
#   scalar  plain C++ with std::fma: any 64-bit CPU
#   neon    ARM64 (Apple Silicon, Snapdragon, Graviton)
# Output: src/vahana/_build/<isa>/cpu_moe*.so
# -ffp-contract=off and -fno-fast-math are required: the twin arithmetic must not be re-associated or fused.
set -euo pipefail
PY=${1:-python}
WANT=${2:-all}
HERE=$(cd "$(dirname "$0")" && pwd)
TP=$($PY -c "import torch,os;print(os.path.dirname(torch.__file__))")
PI=$($PY -c "import sysconfig;print(sysconfig.get_paths()['include'])")
ABI=$($PY -c "import torch;print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")
SUF=$($PY -c "import sysconfig;print(sysconfig.get_config_var('EXT_SUFFIX'))")
ARCH=$(uname -m)
# NUMA placement is optional: link libnuma only when its header exists (the kernel stubs it otherwise)
NUMA=""
# C++20 when the compiler has it (torch >= 2.10 headers require it), else C++17; the standard never changes FP codegen
STD=c++17
if g++ -std=c++20 -E -x c++ /dev/null > /dev/null 2>&1; then STD=c++20; fi
if echo '#include <numaif.h>' | g++ -E -x c++ - > /dev/null 2>&1; then NUMA="-lnuma"; fi
flags() {
  case $1 in
    avx512) echo "-march=x86-64-v3 -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vbmi -mavx512vbmi2 -mtune=znver4" ;;
    avx2)   echo "-march=x86-64-v3 -mtune=generic" ;;
    scalar) if [ "$ARCH" = "x86_64" ]; then echo "-march=x86-64 -mtune=generic"; else echo "-mgeneral-regs-only"; fi ;;
    neon)   echo "-march=armv8-a" ;;
  esac
}
if [ "$WANT" = all ]; then
  if [ "$ARCH" = "x86_64" ]; then ISAS="avx512 avx2 scalar"; else ISAS="neon scalar"; fi
else
  ISAS=$WANT
fi
for isa in $ISAS; do
  OUT=$HERE/../_build/$isa
  mkdir -p "$OUT"
  g++ -O3 $(flags $isa) -ffp-contract=off -fno-fast-math -fopenmp -shared -fPIC -std=$STD \
    -DTORCH_EXTENSION_NAME=cpu_moe -D_GLIBCXX_USE_CXX11_ABI=$ABI \
    -I"$TP/include" -I"$TP/include/torch/csrc/api/include" -I"$PI" \
    "$HERE/cpu_moe.cpp" -o "$OUT/cpu_moe$SUF" \
    -L"$TP/lib" -lc10 -ltorch -ltorch_cpu -ltorch_python $NUMA -lpthread -Wl,-rpath,"$TP/lib"
  echo "built $isa -> $OUT/cpu_moe$SUF"
done
