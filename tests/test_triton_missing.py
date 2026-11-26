"""v1.3.1: without Triton, a CUDA run fails at once with the reason (not later with a stand-in kernel's TypeError)."""
import pytest

from vahana import _triton


def test_require_gpu_message(monkeypatch):
    monkeypatch.setattr(_triton, "HAVE_TRITON", False)
    monkeypatch.setattr(_triton, "IMPORT_ERROR", "ModuleNotFoundError: No module named 'setuptools'")
    with pytest.raises(RuntimeError, match="setuptools"):
        _triton.require_gpu()


def test_require_gpu_ok_when_available(monkeypatch):
    monkeypatch.setattr(_triton, "HAVE_TRITON", True)
    _triton.require_gpu()
