"""vahana attention vs an fp32 reference, and row-alone == in-batch (GPU only)."""
import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _ref(q, k, v, allowed, scale, sinks=None):
    H, Hkv = q.shape[1], k.shape[1]
    kk, vv = (t.float().repeat_interleave(H // Hkv, 1) for t in (k, v))
    s = (q.float() @ kk.transpose(-1, -2) * scale).masked_fill(~allowed, float("-inf"))
    if sinks is not None:
        s = torch.cat([s, sinks.view(1, H, 1, 1).expand(*s.shape[:3], 1)], -1)
    p = torch.nan_to_num(torch.softmax(s, -1))
    p = p[..., :-1] if sinks is not None else p
    return (p @ vv).transpose(1, 2)


@cuda
@pytest.mark.parametrize("Q,window,sink", [(40, None, False), (1, None, False), (40, 16, True)])
def test_attention_matches_reference_and_is_row_invariant(Q, window, sink):
    from vahana.attention import vahana_attention
    dev, g = "cuda", torch.Generator(device="cuda").manual_seed(0)
    B, H, Hkv, T, D = 3, 4, 2, 40, 64
    q = torch.randn(B, H, Q, D, generator=g, device=dev).bfloat16()
    k = torch.randn(B, Hkv, T, D, generator=g, device=dev).bfloat16()
    v = torch.randn(B, Hkv, T, D, generator=g, device=dev).bfloat16()
    qpos = torch.arange(T - Q, T, device=dev)[:, None]
    kpos = torch.arange(T, device=dev)[None]
    m = kpos <= qpos
    if window:
        m &= kpos > qpos - window
    allowed = m[None, None].expand(B, 1, Q, T).clone()
    for b, pad in enumerate((0, 9, 21)):
        allowed[b, :, :, :pad] = False
    sinks = torch.randn(H, generator=g, device=dev) if sink else None
    out, _ = vahana_attention(None, q, k, v, allowed, scaling=D ** -0.5, s_aux=sinks)
    valid = allowed.any(-1)[:, 0]
    assert (out.float() - _ref(q, k, v, allowed, D ** -0.5, sinks)).abs()[valid].max() < 0.02
    for b in range(B):
        one, _ = vahana_attention(None, q[b:b + 1], k[b:b + 1], v[b:b + 1], allowed[b:b + 1], scaling=D ** -0.5,
                                  s_aux=sinks)
        assert torch.equal(one[0], out[b])
