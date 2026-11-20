#!/bin/bash
# Build the Vahana CPU kernel (one plain shared library per instruction set; see build.py).
# Usage: build_kernel.sh [python] [isa ...]    isa: avx512 | avx2 | scalar | neon | all (default: all for this CPU)
# The python argument is kept for compatibility: the build needs no torch.
set -euo pipefail
PY=${1:-python}
shift || true
HERE=$(cd "$(dirname "$0")" && pwd)
if [ "${1:-all}" = all ]; then exec "$PY" "$HERE/build.py"; fi
exec "$PY" "$HERE/build.py" "$@"
