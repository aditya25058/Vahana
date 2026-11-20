"""The Vahana CPU kernel through its C ABI (P3): a plain shared library per instruction set, loaded with ctypes.

The library has no libtorch or Python dependency, so one build works with any torch version on Linux, Windows and
macOS. Tensors cross as descriptors of memory torch owns: the caller keeps registered weights and job buffers alive
(as VahanaExperts and runtime._BUF do). Results the kernel allocates are copied into torch tensors and freed.
ctypes releases the GIL during every call.

`Kernel` exposes the same functions (names and arguments) as the old torch extension.
"""
from __future__ import annotations

import ctypes
import os
import sys

import torch

_DT = {torch.float32: 0, torch.bfloat16: 1, torch.int64: 2, torch.int32: 3, torch.uint8: 4, torch.int8: 5,
       torch.float16: 6, torch.float64: 7}
_TD = {v: k for k, v in _DT.items()}


class VhT(ctypes.Structure):
    _fields_ = [("p", ctypes.c_void_p), ("dt", ctypes.c_int64), ("nd", ctypes.c_int64),
                ("sz", ctypes.c_int64 * 8), ("st", ctypes.c_int64 * 8)]


_P = ctypes.POINTER(VhT)
_I64, _D = ctypes.c_int64, ctypes.c_double


def lib_name() -> str:
    if sys.platform == "win32":
        return "vahana_kernel.dll"
    return "vahana_kernel.dylib" if sys.platform == "darwin" else "vahana_kernel.so"


def _desc(t: torch.Tensor, out: VhT | None = None) -> VhT:
    if t.device.type != "cpu":
        raise ValueError(f"vahana kernel: tensor on {t.device}, expected CPU memory")
    d = out if out is not None else VhT()
    d.p = t.data_ptr() or None
    d.dt = _DT[t.dtype]
    d.nd = t.dim()
    for i, (s, st) in enumerate(zip(t.shape, t.stride())):
        d.sz[i], d.st[i] = s, st
    return d


def _arr(ts):
    a = (VhT * max(1, len(ts)))()
    for i, t in enumerate(ts):
        _desc(t, a[i])
    return a


def _opt(t):
    return ctypes.byref(_desc(t)) if t is not None and t.numel() > 0 else None


class Kernel:
    def __init__(self, path: str):
        self.path = path
        self.lib = L = ctypes.CDLL(path)
        sig = {
            "vh_last_error": (ctypes.c_char_p, []), "vh_isa": (ctypes.c_char_p, []), "vh_nx_fast": (ctypes.c_int, []),
            "vh_have_numa": (ctypes.c_int, []), "vh_threads": (ctypes.c_int, []), "vh_abi": (ctypes.c_int, []),
            "vh_list_len": (_I64, [ctypes.c_void_p]), "vh_list_get": (ctypes.c_int, [ctypes.c_void_p, _I64, _P]),
            "vh_list_free": (None, [ctypes.c_void_p]),
            "vh_expert_mlps": (ctypes.c_void_p, [_P, _P, _P, _P, _I64, _I64]),
            "vh_expert_mlps_nx": (ctypes.c_void_p, [_P, _P, _I64, _I64, _I64]),
            "vh_linear_twin": (ctypes.c_void_p, [_P, _P]), "vh_encode_nx": (ctypes.c_void_p, [_P]),
            "vh_register_layer": (ctypes.c_int, [_I64, _P, _P, _P, _I64]),
            "vh_register_layer_act": (ctypes.c_int, [_I64, _I64, _D, _D, _P, _I64, _P, _I64, _P, _I64]),
            "vh_register_shared": (ctypes.c_int, [_I64, _P, _P, _P]),
            "vh_register_layer_nx": (ctypes.c_int, [_I64, _P, _I64]),
            "vh_register_shared_nx": (ctypes.c_int, [_I64, _P, _I64]),
            "vh_register_linear_nx": (ctypes.c_int, [_P, _P, _I64]),
            "vh_submit": (ctypes.c_int, [_P, _P, _P, _P, _I64, _I64]),
            "vh_submit_linear": (ctypes.c_int, [_P, _P, _P]),
            "vh_submit_moe": (ctypes.c_int, [_I64, _P, _P, _P, _P, _I64]),
            "vh_wait_done": (ctypes.c_void_p, []), "vh_kernel_ms": (_D, []),
            "vh_phase_ms": (None, [ctypes.POINTER(_D)]),
            "vh_place_split": (ctypes.c_void_p, [_P, _I64]), "vh_place_rows": (ctypes.c_void_p, [_P]),
            "vh_rows_nodes": (ctypes.c_void_p, [_P]), "vh_read_bw": (_D, [_P, _I64, _I64]),
            "vh_set_launch_host": (None, [ctypes.c_void_p]),
            "vh_make_host_job": (_I64, [_I64, _P, _P, _P, _P, _I64, _I64]),
            "vh_make_host_linear": (_I64, [_P, _P, _P]), "vh_make_host_embed": (_I64, [_P, _P, _P]),
            "vh_enqueue_host_job": (ctypes.c_int, [_I64, _I64]), "vh_host_calls": (_I64, []),
            "vh_attention_twin": (ctypes.c_int, [_P, _P, _P, _P, _P, _P, _P, _D, _D, _I64, _P]),
            "vh_dexp": (ctypes.c_int, [_P, _P]),
        }
        for name, (res, args) in sig.items():
            f = getattr(L, name)
            f.restype, f.argtypes = res, args
        if L.vh_abi() != 1:
            raise RuntimeError(f"{path}: kernel ABI {L.vh_abi()}, this vahana expects 1")
        self._launch_set = False

    # ---- plumbing
    def _err(self, what):
        return RuntimeError(f"vahana kernel {what}: {self.lib.vh_last_error().decode(errors='replace')}")

    def _ok(self, rc, what):
        if rc != 0:
            raise self._err(what)

    def _list(self, h, what):
        if not h:
            raise self._err(what)
        try:
            out = []
            d = VhT()
            for i in range(self.lib.vh_list_len(h)):
                self._ok(self.lib.vh_list_get(h, i, ctypes.byref(d)), what)
                shape = [d.sz[k] for k in range(d.nd)]
                t = torch.empty(shape, dtype=_TD[d.dt])
                if t.numel():
                    ctypes.memmove(t.data_ptr(), d.p, t.numel() * t.element_size())
                out.append(t)
            return out
        finally:
            self.lib.vh_list_free(h)

    # ---- info
    def isa(self) -> str:
        return self.lib.vh_isa().decode()

    def nx_fast(self) -> bool:
        return bool(self.lib.vh_nx_fast())

    def have_numa(self) -> bool:
        return bool(self.lib.vh_have_numa())

    def threads(self) -> int:
        return self.lib.vh_threads()

    # ---- compute
    def expert_mlps(self, x, g, u, d, numa):
        return self._list(self.lib.vh_expert_mlps(_desc(x), _arr(g), _arr(u), _arr(d), len(g), numa), "expert_mlps")

    def expert_mlps_nx(self, x, packs, I, numa):
        return self._list(self.lib.vh_expert_mlps_nx(_desc(x), _arr(packs), len(packs), I, numa), "expert_mlps_nx")[0]

    def linear_twin(self, x, W):
        return self._list(self.lib.vh_linear_twin(_desc(x), _desc(W)), "linear_twin")[0]

    def encode_nx(self, W):
        return self._list(self.lib.vh_encode_nx(_desc(W)), "encode_nx")

    def attention_twin(self, q, k, v, first, last, mask, sinks, scale, cap, has_cap):
        B, H, Q = q.shape[0], q.shape[1], q.shape[2]
        out = torch.empty(B, Q, H, v.shape[3], dtype=q.dtype)
        self._ok(self.lib.vh_attention_twin(_desc(q), _desc(k), _desc(v), _desc(first), _desc(last), _opt(mask),
                                            _opt(sinks), scale, cap, int(has_cap), _desc(out)), "attention_twin")
        return out

    def dexp(self, x):
        x = x.float().contiguous()
        y = torch.empty_like(x)
        self._ok(self.lib.vh_dexp(_desc(x), _desc(y)), "dexp")
        return y

    # ---- registration (the caller keeps every tensor alive)
    def register_layer(self, layer, g, u, d):
        self._ok(self.lib.vh_register_layer(layer, _arr(g), _arr(u), _arr(d), len(g)), "register_layer")

    def register_layer_act(self, layer, act, alpha, limit, bg, bu, bd):
        self._ok(self.lib.vh_register_layer_act(layer, act, alpha, limit, _arr(bg), len(bg), _arr(bu), len(bu),
                                                _arr(bd), len(bd)), "register_layer_act")

    def register_shared(self, layer, g, u, d):
        self._ok(self.lib.vh_register_shared(layer, _desc(g), _desc(u), _desc(d)), "register_shared")

    def register_layer_nx(self, layer, packs):
        self._ok(self.lib.vh_register_layer_nx(layer, _arr(packs), len(packs)), "register_layer_nx")

    def register_shared_nx(self, layer, packs):
        self._ok(self.lib.vh_register_shared_nx(layer, _arr(packs), len(packs)), "register_shared_nx")

    def register_linear_nx(self, W, packs):
        self._ok(self.lib.vh_register_linear_nx(_desc(W), _arr(packs), len(packs)), "register_linear_nx")

    # ---- the worker thread (the caller keeps inputs and outputs alive until wait_done)
    def submit(self, x, g, u, d, numa):
        self._keep = (x, g, u, d)
        self._ok(self.lib.vh_submit(_desc(x), _arr(g), _arr(u), _arr(d), len(g), numa), "submit")

    def submit_linear(self, x, W, out):
        self._keep = (x, W, out)
        self._ok(self.lib.vh_submit_linear(_desc(x), _desc(W), _desc(out)), "submit_linear")

    def submit_moe(self, layer, x, ti, tw, out, numa):
        self._keep = (x, ti, tw, out)
        self._ok(self.lib.vh_submit_moe(layer, _desc(x), _desc(ti), _desc(tw), _desc(out), numa), "submit_moe")

    def wait_done(self):
        r = self._list(self.lib.vh_wait_done(), "worker")
        self._keep = None
        return r

    def kernel_ms(self) -> float:
        return self.lib.vh_kernel_ms()

    def phase_ms(self):
        a = (_D * 5)()
        self.lib.vh_phase_ms(a)
        return list(a)

    # ---- NUMA placement (Linux; elsewhere no-ops that report nothing placed)
    def place_split(self, t, split):
        return self._list(self.lib.vh_place_split(_desc(t), split), "place_split")[0].tolist()

    def place_rows(self, W):
        return self._list(self.lib.vh_place_rows(_desc(W)), "place_rows")[0].tolist()

    def rows_nodes(self, W):
        return self._list(self.lib.vh_rows_nodes(_desc(W)), "rows_nodes")[0].tolist()

    def read_bw(self, packs, numa):
        r = self.lib.vh_read_bw(_arr(packs), len(packs), numa)
        if r < 0:
            raise self._err("read_bw")
        return r

    # ---- CUDA host-node jobs (buffers must be pinned and kept alive by the caller)
    def _launch(self):
        if not self._launch_set:
            self.lib.vh_set_launch_host(cuda_launch_host_func())
            self._launch_set = True

    def make_host_job(self, layer, x, ti, tw, out, numa, noop):
        r = self.lib.vh_make_host_job(layer, _desc(x), _desc(ti), _desc(tw), _desc(out), numa, int(noop))
        if r < 0:
            raise self._err("make_host_job")
        return r

    def make_host_linear(self, W, x, out):
        r = self.lib.vh_make_host_linear(_desc(W), _desc(x), _desc(out))
        if r < 0:
            raise self._err("make_host_linear")
        return r

    def make_host_embed(self, W, ids, out):
        r = self.lib.vh_make_host_embed(_desc(W), _desc(ids), _desc(out))
        if r < 0:
            raise self._err("make_host_embed")
        return r

    def enqueue_host_job(self, stream, job):
        self._launch()
        self._ok(self.lib.vh_enqueue_host_job(stream, job), "enqueue_host_job")

    def host_calls(self) -> int:
        return self.lib.vh_host_calls()


def cuda_launch_host_func() -> int:
    """Address of cudaLaunchHostFunc in the CUDA runtime torch has loaded (Linux or Windows)."""
    torch.cuda.init()
    names = []
    if sys.platform == "win32":
        import glob
        tl = os.path.join(os.path.dirname(torch.__file__), "lib")
        names = [os.path.basename(p) for p in sorted(glob.glob(os.path.join(tl, "cudart64_*.dll")))] + \
            sorted(glob.glob(os.path.join(tl, "cudart64_*.dll")))
    else:
        names = ["libcudart.so.12", "libcudart.so.13", "libcudart.so.11.0", "libcudart.so"]
    for n in names:
        try:
            lib = ctypes.CDLL(n, mode=getattr(os, "RTLD_NOLOAD", 0) | getattr(os, "RTLD_NOW", 0)) \
                if sys.platform != "win32" else ctypes.CDLL(n)
            return ctypes.cast(lib.cudaLaunchHostFunc, ctypes.c_void_p).value
        except (OSError, AttributeError):
            continue
    raise RuntimeError("cudaLaunchHostFunc not found in the CUDA runtime loaded by torch")
