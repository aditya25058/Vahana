"""M3: invariant.install replaces a norm only if the invariant RMSNorm reproduces it exactly.
Structural matching on `variance_epsilon` also caught LayerNorms (cohere2) and gated norms (qwen3.5); now each norm
class is checked once: one positional input, and inv_rmsnorm(x) == module(x) bit for bit on a random input.
Otherwise the class keeps its stock forward and is listed in the result."""
import ast
import sys

F = f"{sys.argv[1]}/invariant.py"
s = open(F).read()
a = """    for m in model.modules():
        if hasattr(m, "variance_epsilon") and isinstance(getattr(m, "weight", None), torch.Tensor):
            m.forward = types.MethodType(lambda self, x: inv_rmsnorm(x, self.weight, self.variance_epsilon), m)
            n_norm += 1
        mods.add(type(m).__module__)"""
b = """    verdict: dict[type, bool] = {}
    kept = set()
    for m in model.modules():
        if hasattr(m, "variance_epsilon") and isinstance(getattr(m, "weight", None), torch.Tensor):
            cls = type(m)
            if cls not in verdict:
                verdict[cls] = _rmsnorm_equivalent(m)
            if verdict[cls]:
                m.forward = types.MethodType(lambda self, x: inv_rmsnorm(x, self.weight, self.variance_epsilon), m)
                n_norm += 1
            else:
                kept.add(cls.__name__)
        mods.add(type(m).__module__)"""
assert s.count(a) == 1
s = s.replace(a, b)
a = """    return {"rmsnorm": n_norm, "f_shimmed_modules": shimmed, "linear": "class-level"}"""
b = """    return {"rmsnorm": n_norm, "norms_kept_stock": sorted(kept), "f_shimmed_modules": shimmed,
            "linear": "class-level"}"""
assert s.count(a) == 1
s = s.replace(a, b)
a = "def install(model: torch.nn.Module) -> dict:"
b = '''def _rmsnorm_equivalent(m: torch.nn.Module) -> bool:
    """True if `m` takes one input and inv_rmsnorm reproduces its output bit for bit (random input, m's device)."""
    import inspect as _insp
    try:
        params = [p for p in _insp.signature(type(m).forward).parameters.values()][1:]
    except (TypeError, ValueError):
        return False
    if len(params) != 1:                           # gated / extra-argument norms are not plain RMSNorm
        return False
    w = m.weight
    if w.device.type == "meta" or w.dim() != 1:
        return False
    g = torch.Generator(device=w.device).manual_seed(0)
    x = torch.randn(5, w.shape[0], generator=g, device=w.device).to(w.dtype)
    with torch.no_grad():
        try:
            ref = m(x)
        except Exception:  # noqa: BLE001
            return False
        return ref.shape == x.shape and torch.equal(ref, inv_rmsnorm(x, w, m.variance_epsilon))


def install(model: torch.nn.Module) -> dict:'''
assert s.count(a) == 1
s = s.replace(a, b)
ast.parse(s)
open(F, "w").write(s)
print("invariant.install: verified norm replacement")
