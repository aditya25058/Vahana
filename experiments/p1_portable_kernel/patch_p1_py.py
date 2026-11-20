"""P1 Python side: pick the kernel build for this CPU; use NX12 only where the build decodes it fast."""
import sys

root = sys.argv[1]
F = f"{root}/src/vahana/hybrid.py"
s = open(F).read()
a = '''def ext():
    global _EXT
    if _EXT is None:
        # VAHANA_HYBRID_EXT overrides; default is the in-package build (csrc/build_kernel.sh)
        sys.path.insert(0, os.environ.get("VAHANA_HYBRID_EXT") or
                        os.path.join(os.path.dirname(os.path.abspath(__file__)), "_build"))
        import cpu_moe  # noqa: E402
        _EXT = cpu_moe
    return _EXT'''
b = '''def cpu_isa() -> str:
    """Best kernel build this CPU can run: avx512 (with VBMI), avx2 (with FMA), neon, else scalar.
    VAHANA_ISA forces one (every build gives the same bits; only speed differs)."""
    want = os.environ.get("VAHANA_ISA")
    if want:
        return want
    import platform
    if platform.machine().lower() in ("arm64", "aarch64"):
        return "neon"
    flags: set = set()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("flags"):
                    flags = set(line.split(":", 1)[1].split())
                    break
    except OSError:
        cap = torch.backends.cpu.get_cpu_capability().lower()   # non-Linux: torch's own detection
        flags = {"avx2", "fma"} if "avx2" in cap or "avx512" in cap else set()
    if {"avx512f", "avx512bw", "avx512vl", "avx512dq", "avx512vbmi"} <= flags:
        return "avx512"
    if {"avx2", "fma"} <= flags:
        return "avx2"
    return "scalar"


def ext():
    global _EXT
    if _EXT is None:
        # VAHANA_HYBRID_EXT overrides; default is the in-package build for this CPU (csrc/build_kernel.sh)
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_build")
        path = os.environ.get("VAHANA_HYBRID_EXT")
        if not path:
            order = {"avx512": ["avx512", "avx2", "scalar"], "avx2": ["avx2", "scalar"],
                     "neon": ["neon", "scalar"]}.get(cpu_isa(), [cpu_isa(), "scalar"])
            path = next((os.path.join(base, i) for i in order if os.path.isdir(os.path.join(base, i))), base)
        sys.path.insert(0, path)
        import cpu_moe  # noqa: E402
        _EXT = cpu_moe
    return _EXT'''
assert s.count(a) == 1
s = s.replace(a, b)
open(F, "w").write(s)

F = f"{root}/src/vahana/experts.py"
s = open(F).read()
a = "        self.nx = nx and spec.H % 64 == 0 and spec.I % 64 == 0 and \\"
b = ("        nx_fast = getattr(k, \"nx_fast\", lambda: True)()      # portable builds: raw BF16 is faster than NX12\n"
     "        self.nx = nx and nx_fast and spec.H % 64 == 0 and spec.I % 64 == 0 and \\")
assert s.count(a) == 1
s = s.replace(a, b)
open(F, "w").write(s)
print("python side patched")
