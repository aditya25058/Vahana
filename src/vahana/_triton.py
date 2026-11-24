"""Triton when it is installed; otherwise a stand-in so CPU-only machines can import Vahana.

Without Triton, `@triton.jit` leaves the function as plain Python (never called: every GPU path checks the device
first) and `tl.<anything>` is a placeholder. HAVE_TRITON says which one this is.
"""
from __future__ import annotations

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:            # CPU-only install (no GPU, or torch without Triton)
    HAVE_TRITON = False

    class _Any:
        def __getattr__(self, name):
            return _Any()

        def __call__(self, *a, **k):
            return _Any()

    class _Triton:
        @staticmethod
        def jit(fn=None, **kw):
            return fn if fn is not None else (lambda f: f)

        @staticmethod
        def next_power_of_2(n: int) -> int:
            return 1 << (int(n) - 1).bit_length()

        @staticmethod
        def cdiv(a: int, b: int) -> int:
            return -(-a // b)

        def __getattr__(self, name):
            return _Any()

    triton = _Triton()
    tl = _Any()



@triton.jit
def div_rn(a, b):
    """fp32 a / b, correctly rounded, subnormals kept: PTX div.rn.f32 itself. (tl.math.div_rn goes through libdevice,
    which Triton 3.0 compiles in flush-to-zero mode: a subnormal quotient or operand became 0 there, but not on newer
    Triton or on the CPU.)"""
    return tl.inline_asm_elementwise("div.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32, is_pure=True,
                                     pack=1)


@triton.jit
def sqrt_rn(a):
    """fp32 sqrt, correctly rounded, subnormals kept (PTX sqrt.rn.f32; see div_rn)."""
    return tl.inline_asm_elementwise("sqrt.rn.f32 $0, $1;", "=f,f", [a], dtype=tl.float32, is_pure=True, pack=1)


__all__ = ["triton", "tl", "HAVE_TRITON", "div_rn", "sqrt_rn"]
