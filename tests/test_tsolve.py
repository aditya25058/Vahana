"""v1.1: the twin triangular solve (transformers 5.17+ Qwen3.5 gated delta rule calls torch.linalg.solve_triangular)."""
import torch

from vahana import tmath


def _sys(upper, unit, g):
    A = torch.randn(2, 3, 32, 32, generator=g) * 0.1
    A = A.triu() if upper else A.tril()
    if not unit:
        A = A + torch.eye(32) * 2
    return A, torch.randn(2, 3, 32, 8, generator=g)


def test_accuracy_all_forms():
    g = torch.Generator().manual_seed(0)
    for upper in (False, True):
        for unit in (False, True):
            A, B = _sys(upper, unit, g)
            X = tmath.tsolve_triangular(A, B, upper, unit)
            R = torch.linalg.solve_triangular(A, B, upper=upper, unitriangular=unit)
            assert float((X - R).abs().max() / R.abs().max()) < 1e-5


def test_twinmode_handles_the_call_and_audit_stays_empty():
    g = torch.Generator().manual_seed(1)
    A, B = _sys(False, True, g)
    audit = {}
    with tmath.TwinMode(audit):
        X = torch.linalg.solve_triangular(A, B, upper=False, unitriangular=True)
    assert audit == {}
    assert torch.equal(X, tmath.tsolve_triangular(A, B, False, True))


def test_deterministic_across_batch_layout():
    """Rows are solved independently of the batch: solving one batch element alone gives the same bits."""
    g = torch.Generator().manual_seed(2)
    A, B = _sys(True, False, g)
    X = tmath.tsolve_triangular(A, B, True, False)
    assert torch.equal(X[1, 2], tmath.tsolve_triangular(A[1, 2], B[1, 2], True, False))
