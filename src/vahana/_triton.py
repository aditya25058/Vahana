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

__all__ = ["triton", "tl", "HAVE_TRITON"]
