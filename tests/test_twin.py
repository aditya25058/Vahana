"""P2 twin arithmetic: the glue definitions are accurate, and (with a GPU) CPU and GPU give the same bits.
CPU-only machines run the accuracy and CPU-attention tests; the cross-device tests need CUDA."""
import torch
import torch.nn.functional as F

from vahana import tmath as T


def _cuda():
    return torch.cuda.is_available()


def _bits_equal(a, b):
    a, b = a.cpu(), b.cpu()
    na, nb = torch.isnan(a), torch.isnan(b)
    v = {torch.bfloat16: torch.int16, torch.float32: torch.int32, torch.float64: torch.int64}[a.dtype]
    return a.dtype == b.dtype and torch.equal(na, nb) and torch.equal(a.view(v)[~na], b.view(v)[~nb])


def test_transcendentals_accurate():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(5000, generator=g) * 8
    p = torch.rand(5000, generator=g) * 100 + 1e-3
    for f, r, xx in [(T.texp, torch.exp, x), (T.tlog, torch.log, p), (T.tsin, torch.sin, x * 100),
                     (T.tcos, torch.cos, x * 100), (T.ttanh, torch.tanh, x), (T.tsigmoid, torch.sigmoid, x),
                     (T.tsqrt, torch.sqrt, p)]:
        a, ref = f(xx).double(), r(xx.double())
        ulp = (ref.float().abs() * 2**-23).double().clamp_min(2**-149)
        assert float(((a - ref).abs() / ulp).max()) <= 1.0, f.__name__


def test_glue_close_to_stock():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(4, 37, 300, generator=g)
    assert torch.allclose(T.tsum(x, -1), x.sum(-1), atol=1e-4)
    assert torch.allclose(T.tmean(x, (-1, -2), keepdim=True), x.mean((-1, -2), keepdim=True), atol=1e-6)
    assert torch.allclose(T.tsoftmax(x, -1), torch.softmax(x, -1), atol=1e-6)
    w = torch.randn(64, 300, generator=g)
    assert torch.allclose(T.tmatmul(x, w.t()), x @ w.t(), atol=1e-3)
    assert torch.allclose(T.tcumsum(x, 1), x.cumsum(1), atol=1e-3)
    assert torch.allclose(T.tlayer_norm(x, (300,)), F.layer_norm(x, (300,)), atol=1e-5)
    c = torch.randn(2, 64, 50, generator=g)
    k = torch.randn(64, 1, 4, generator=g)
    assert torch.allclose(T.tconv1d(c, k, padding=3, groups=64), F.conv1d(c, k, padding=3, groups=64), atol=1e-5)
    v = torch.tensor([[3.0, 1.0, 3.0, 2.0, 3.0]])
    assert T.ttopk(v, 2).indices.tolist() == [[0, 2]]           # ties: lower index first


def test_cpu_attention_close_to_sdpa():
    from vahana.attention import vahana_attention
    g = torch.Generator().manual_seed(2)
    q, k, v = (torch.randn(2, 4, 9, 64, generator=g).bfloat16() for _ in range(3))
    k, v = k[:, :2], v[:, :2]
    out, _ = vahana_attention(None, q, k, v, None)
    ref = F.scaled_dot_product_attention(q.float(), k.float().repeat_interleave(2, 1),
                                         v.float().repeat_interleave(2, 1), is_causal=True)
    assert (out.float() - ref.transpose(1, 2)).abs().max() < 0.02


def test_twinmode_routes_model_calls():
    audit = {}
    x = torch.randn(3, 8)
    with T.TwinMode(audit):
        y = torch.softmax(x, -1) / 3.0 + torch.rsqrt(x.abs() + 1).sum(-1, keepdim=True)
    assert audit == {}
    ref = T.tdiv(T.tsoftmax(x, -1), 3.0) + T.tsum(T.trsqrt(x.abs() + 1), -1, keepdim=True)
    assert _bits_equal(y, ref)


def test_cpu_equals_gpu():
    if not _cuda():
        return
    from vahana.attention import vahana_attention
    d = torch.device("cuda")
    g = torch.Generator().manual_seed(3)
    x = torch.cat([torch.randn(3000, generator=g) * s for s in (0.1, 3.0, 50.0)])
    for f in (T.texp, T.tsin, T.tcos, T.tsigmoid, T.tsilu, T.tsqrt, T.ttanh):
        for dt in (torch.float32, torch.bfloat16):
            assert _bits_equal(f(x.to(dt)), f(x.to(dt).to(d))), f.__name__
    m = torch.randn(5, 300, generator=g)
    assert _bits_equal(T.tsoftmax(m, -1), T.tsoftmax(m.to(d), -1))
    assert _bits_equal(T.tmean(m, -1), T.tmean(m.to(d), -1))
    q, k, v = (torch.randn(2, 4, 70, 64, generator=g).bfloat16() for _ in range(3))
    mask = torch.ones(2, 1, 70, 70, dtype=torch.bool).tril()
    mask[0, :, :, :5] = False
    a, _ = vahana_attention(None, q, k, v, mask, softcap=30.0)
    b, _ = vahana_attention(None, q.to(d), k.to(d), v.to(d), mask.to(d), softcap=30.0)
    assert _bits_equal(a, b)
