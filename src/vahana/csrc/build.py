"""Build the Vahana CPU kernel: one plain shared library per instruction set (P3). No torch, no Python headers.

  python src/vahana/csrc/build.py [isa ...]        isa: avx512 | avx2 | scalar | neon (default: all for this machine)

Output: src/vahana/_build/<isa>/vahana_kernel.{so,dll,dylib}. Compiler: $CXX, else g++ (Linux), clang++ (macOS),
clang++ or cl (Windows). Every build computes identical bits; floating-point contraction is always off.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "_build")
SRC = os.path.join(HERE, "cpu_moe.cpp")
ARM = platform.machine().lower() in ("arm64", "aarch64")
WIN, MAC = sys.platform == "win32", sys.platform == "darwin"
LIB = "vahana_kernel." + ("dll" if WIN else "dylib" if MAC else "so")

GNU_ISA = {
    "avx512": ["-march=x86-64-v3", "-mavx512f", "-mavx512bw", "-mavx512vl", "-mavx512dq", "-mavx512vbmi",
               "-mavx512vbmi2", "-mtune=znver4"],
    "avx2": ["-march=x86-64-v3", "-mtune=generic"],
    "scalar": ["-DVH_FORCE_SCALAR"] + ([] if ARM else ["-march=x86-64", "-mtune=generic"]),
    "neon": [] if MAC else ["-march=armv8-a"],
}
MSVC_ISA = {"avx512": ["/arch:AVX512"], "avx2": ["/arch:AVX2"], "scalar": ["/DVH_FORCE_SCALAR"], "neon": []}


def compiler() -> tuple[str, str]:
    cxx = os.environ.get("CXX")
    if cxx:
        return cxx, ("msvc" if os.path.basename(cxx).lower().startswith("cl") else "gnu")
    for c in (["clang++", "cl"] if WIN else ["clang++", "g++"] if MAC else ["g++", "clang++"]):
        if shutil.which(c):
            return c, ("msvc" if c == "cl" else "gnu")
    raise SystemExit("no C++ compiler found (set CXX)")


def build(isa: str) -> str:
    cxx, kind = compiler()
    d = os.path.join(OUT, isa)
    os.makedirs(d, exist_ok=True)
    out = os.path.join(d, LIB)
    if kind == "msvc":
        cmd = [cxx, "/nologo", "/O2", "/std:c++20", "/EHsc", "/LD", "/fp:precise", "/DNDEBUG", *MSVC_ISA[isa],
               SRC, f"/Fe{out}", f"/Fo{os.path.join(d, 'cpu_moe.obj')}"]
    else:
        cmd = [cxx, "-O3", *GNU_ISA[isa], "-ffp-contract=off", "-fno-fast-math", "-std=c++20", "-shared",
               "-fvisibility=hidden", SRC, "-o", out]
        if not WIN:
            cmd[1:1] = ["-fPIC", "-pthread"]
        if MAC:
            cmd.insert(1, "-dynamiclib")
        if WIN:
            cmd += ["-D_CRT_SECURE_NO_WARNINGS"]
        if sys.platform.startswith("linux") and os.environ.get("VAHANA_STATIC_LIBSTDCXX") == "1":
            cmd += ["-static-libstdc++", "-static-libgcc"]     # wheels: no libstdc++ version requirement
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return out


def default_isas() -> list[str]:
    return ["neon", "scalar"] if ARM else ["avx512", "avx2", "scalar"]


if __name__ == "__main__":
    for i in sys.argv[1:] or default_isas():
        print("built", i, "->", build(i), flush=True)
