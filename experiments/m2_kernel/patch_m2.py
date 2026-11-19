"""M2 kernel generalisation (csrc/cpu_moe.cpp):
- ActCtx: per-expert activation (0 = SiLU-GLU, 1 = gpt-oss clamped SwiGLU) and optional gate/up/down biases,
  routed through every expert path (raw n == 1, raw n > 1, NX12, fused v16). ctx == nullptr is the original
  arithmetic character for character.
- Biases are added after the dot product's bf16 rounding, then rounded again (stock: x @ W -> bf16, + b -> bf16).
- register_layer_act(layer, act, alpha, limit, bg, bu, bd).
- Routing weights may be fp32 (used as given) or bf16.
"""
import re
import sys

p = sys.argv[1]
s = open(p).read()


def rep(a, b, n=1):
    global s
    c = s.count(a)
    assert c == n, (c, n, a[:90])
    s = s.replace(a, b)


def rep_re(pat, b, n=1):
    global s
    found = re.findall(pat, s)
    assert len(found) == n, (len(found), n, pat[:90])
    s = re.sub(pat, b, s)


# ---- ActCtx + helpers, before expert_mlps
rep("std::vector<torch::Tensor> expert_mlps(torch::Tensor x, std::vector<torch::Tensor> gate,",
    r'''// M2: per-expert activation / bias context (index j = the expert's position in the call).
// nullptr or empty = SiLU-GLU without biases: exactly the original arithmetic.
struct ActCtx { std::vector<int> act; std::vector<const float*> bg, bu, bd; float alpha = 1.702f, limit = 7.0f; };
static inline float act_h(const ActCtx* c, int j, int r, float gdot, float udot) {
  if (!c || c->act.empty()) {
    float gb = bf2f(f2bf(gdot)), ub = bf2f(f2bf(udot));
    float s = bf2f(f2bf(gb / (1.0f + dexp(-gb))));
    return bf2f(f2bf(s * ub));
  }
  float gb = bf2f(f2bf(gdot)), ub = bf2f(f2bf(udot));
  if (c->bg[j]) gb = bf2f(f2bf(gb + c->bg[j][r]));
  if (c->bu[j]) ub = bf2f(f2bf(ub + c->bu[j][r]));
  if (c->act[j] == 1) {                       // gpt-oss: clamp, g * sigmoid(alpha g), (u + 1) * glu
    gb = std::fmin(gb, c->limit);
    ub = std::fmin(std::fmax(ub, -c->limit), c->limit);
    const float z = bf2f(f2bf(gb * c->alpha));
    const float sg = bf2f(f2bf(1.0f / (1.0f + dexp(-z))));
    const float glu = bf2f(f2bf(gb * sg));
    return bf2f(f2bf(bf2f(f2bf(ub + 1.0f)) * glu));
  }
  float s = bf2f(f2bf(gb / (1.0f + dexp(-gb))));
  return bf2f(f2bf(s * ub));
}
static inline uint16_t down_o(const ActCtx* c, int j, int r, float dot) {
  if (!c || c->act.empty() || !c->bd[j]) return f2bf(dot);
  return f2bf(bf2f(f2bf(dot)) + c->bd[j][r]);
}
std::vector<torch::Tensor> expert_mlps_ctx(torch::Tensor x, std::vector<torch::Tensor> gate,''')
rep("""                                       std::vector<torch::Tensor> up, std::vector<torch::Tensor> down,
                                       int64_t numa) {
  const double T0 = now_ms();""", """                                       std::vector<torch::Tensor> up, std::vector<torch::Tensor> down,
                                       int64_t numa, const ActCtx* ctx) {
  const double T0 = now_ms();""")
# raw n == 1
rep_re(r"float gb = bf2f\(f2bf\(o\[q\]\)\), ub = bf2f\(f2bf\(o\[2 \+ q\]\)\);\n\s*float s = bf2f\(f2bf\(gb / \(1\.0f \+ dexp\(-gb\)\)\)\);\n\s*hf\[\(size_t\)e \* I \+ r \+ q\] = bf2f\(f2bf\(s \* ub\)\);",
       "hf[(size_t)e * I + r + q] = act_h(ctx, e, r + q, o[q], o[2 + q]);")
# raw n > 1
rep_re(r"float gb = bf2f\(f2bf\(g\[t\]\)\), ub = bf2f\(f2bf\(u\[t\]\)\);\n\s*float s = bf2f\(f2bf\(gb / \(1\.0f \+ dexp\(-gb\)\)\)\);\n\s*hf\[\(\(size_t\)e \* n \+ t\) \* I \+ r\] = bf2f\(f2bf\(s \* ub\)\);",
       "hf[((size_t)e * n + t) * I + r] = act_h(ctx, e, r, g[t], u[t]);")
rep("yp[((size_t)e * n + t) * H + r] = f2bf(d[t]);", "yp[((size_t)e * n + t) * H + r] = down_o(ctx, e, r, d[t]);")
# raw n == 1 down and NX down share this text
rep("for (int q = 0; q < 4; q++) yp[(size_t)e * H + r + q] = f2bf(o[q]);",
    "for (int q = 0; q < 4; q++) yp[(size_t)e * H + r + q] = down_o(ctx, e, r + q, o[q]);", n=2)
# wrapper for the public expert_mlps
rep("std::vector<double> phase_ms() {", """std::vector<torch::Tensor> expert_mlps(torch::Tensor x, std::vector<torch::Tensor> gate,
                                       std::vector<torch::Tensor> up, std::vector<torch::Tensor> down, int64_t numa) {
  return expert_mlps_ctx(x, gate, up, down, numa, nullptr);
}
std::vector<double> phase_ms() {""")
# NX path
rep_re(r"float gb = bf2f\(f2bf\(o\[q\]\)\), ub = bf2f\(f2bf\(o\[2 \+ q\]\)\);\n\s*float sv = bf2f\(f2bf\(gb / \(1\.0f \+ dexp\(-gb\)\)\)\);\n\s*hf\[\(size_t\)e \* I \+ r \+ q\] = bf2f\(f2bf\(sv \* ub\)\);",
       "hf[(size_t)e * I + r + q] = act_h(ctx, e, r + q, o[q], o[2 + q]);")
rep("torch::Tensor expert_mlps_nx(torch::Tensor x, std::vector<torch::Tensor> packs, int64_t I_, int64_t numa) {",
    """torch::Tensor expert_mlps_nx_ctx(torch::Tensor x, std::vector<torch::Tensor> packs, int64_t I_, int64_t numa,
                                 const ActCtx* ctx);
torch::Tensor expert_mlps_nx(torch::Tensor x, std::vector<torch::Tensor> packs, int64_t I_, int64_t numa) {
  return expert_mlps_nx_ctx(x, packs, I_, numa, nullptr);
}
torch::Tensor expert_mlps_nx_ctx(torch::Tensor x, std::vector<torch::Tensor> packs, int64_t I_, int64_t numa,
                                 const ActCtx* ctx) {""")
# fused path
rep_re(r"float gb = bf2f\(f2bf\(o\[q\]\)\), ub = bf2f\(f2bf\(o\[2 \+ q\]\)\);\n\s*float sv = bf2f\(f2bf\(gb / \(1\.0f \+ dexp\(-gb\)\)\)\);\n\s*hj\[r \+ q\] = bf2f\(f2bf\(sv \* ub\)\);",
       "hj[r + q] = act_h(ctx, j, r + q, o[q], o[2 + q]);")
rep("for (int q = 0; q < 4; q++) y[j][r + q] = f2bf(o[q]);",
    "for (int q = 0; q < 4; q++) y[j][r + q] = down_o(ctx, j, r + q, o[q]);")
rep("""                             int H, const std::vector<uint16_t*>& y, int numa);""",
    """                             int H, const std::vector<uint16_t*>& y, int numa, const ActCtx* ctx);""")
rep("""                             int H, const std::vector<uint16_t*>& y, int numa) {""",
    """                             int H, const std::vector<uint16_t*>& y, int numa, const ActCtx* ctx) {""")
# forward declaration of the ctx NX path before moe_fused
rep("torch::Tensor expert_mlps_nx(torch::Tensor x, std::vector<torch::Tensor> packs, int64_t I_, int64_t numa);\nstatic void experts_nx_fused",
    "torch::Tensor expert_mlps_nx(torch::Tensor x, std::vector<torch::Tensor> packs, int64_t I_, int64_t numa);\n"
    "torch::Tensor expert_mlps_nx_ctx(torch::Tensor x, std::vector<torch::Tensor> packs, int64_t I_, int64_t numa,\n"
    "                                 const ActCtx* ctx);\nstatic void experts_nx_fused")

# ---- layer state, registration, context building
rep("struct LayerW { std::vector<torch::Tensor> g, u, d; std::vector<torch::Tensor> sg, su, sd; std::vector<torch::Tensor> nx, snx; };",
    "struct LayerW { std::vector<torch::Tensor> g, u, d; std::vector<torch::Tensor> sg, su, sd; std::vector<torch::Tensor> nx, snx;\n"
    "                int act = 0; float alpha = 1.702f, limit = 7.0f; std::vector<torch::Tensor> bg, bu, bd; };")
rep("// Phase 30: shared experts of a layer, computed in the same job and added (bf16 add) to the routed sum.",
    """// M2: activation (0 SiLU-GLU, 1 gpt-oss clamped SwiGLU) and optional per-expert biases (lists may be empty).
void register_layer_act(int64_t layer, int64_t act, double alpha, double limit, std::vector<torch::Tensor> bg,
                        std::vector<torch::Tensor> bu, std::vector<torch::Tensor> bd) {
  LayerW& L = LAYERS->at(layer);
  L.act = (int)act; L.alpha = (float)alpha; L.limit = (float)limit;
  auto f32 = [](std::vector<torch::Tensor> v) { for (auto& t : v) t = t.to(torch::kFloat).contiguous(); return v; };
  L.bg = f32(bg); L.bu = f32(bu); L.bd = f32(bd);
}
static bool layer_has_ctx(const LayerW& L) { return L.act != 0 || !L.bg.empty() || !L.bu.empty() || !L.bd.empty(); }
static void build_ctx(const LayerW& L, const std::vector<int64_t>& ids, bool shared_first, ActCtx& c) {
  c.alpha = L.alpha; c.limit = L.limit;
  auto ptr = [](const std::vector<torch::Tensor>& v, int64_t e) -> const float* {
    return v.empty() ? nullptr : v[e].data_ptr<float>(); };
  if (shared_first) { c.act.push_back(0); c.bg.push_back(nullptr); c.bu.push_back(nullptr); c.bd.push_back(nullptr); }
  for (int64_t e : ids) { c.act.push_back(L.act); c.bg.push_back(ptr(L.bg, e)); c.bu.push_back(ptr(L.bu, e)); c.bd.push_back(ptr(L.bd, e)); }
}
// Phase 30: shared experts of a layer, computed in the same job and added (bf16 add) to the routed sum.""")

# ---- moe_fused: fp32 routing weights, contexts
rep("""  const uint16_t* twp = (const uint16_t*)tw.data_ptr();
  const uint16_t* xp = (const uint16_t*)x.data_ptr();
  static const bool fuse""", """  const uint16_t* twp = (const uint16_t*)tw.data_ptr();
  const bool twf = tw.scalar_type() == torch::kFloat;          // M2: fp32 routing weights used as given
  const float* twfp = twf ? tw.data_ptr<float>() : nullptr;
  const uint16_t* xp = (const uint16_t*)x.data_ptr();
  static const bool fuse""")
rep("const float w = bf2f(twp[ek[q].second]);", "const float w = twf ? twfp[ek[q].second] : bf2f(twp[ek[q].second]);")
rep("const float w = bf2f(twp[b * K + ek[q].second]);",
    "const float w = twf ? twfp[b * K + ek[q].second] : bf2f(twp[b * K + ek[q].second]);")
rep("    experts_nx_fused(P, Is, xf.data(), H, yv, (int)numa);",
    """    ActCtx fc; const ActCtx* fcp = nullptr;
    if (layer_has_ctx(L)) {
      std::vector<int64_t> v; for (int q = 0; q < K; q++) v.push_back(ek[q].first);
      build_ctx(L, v, sh, fc); fcp = &fc;
    }
    experts_nx_fused(P, Is, xf.data(), H, yv, (int)numa, fcp);""")
rep("""  torch::Tensor Y;
  if (nmax == 1 && !L.nx.empty()) {""", """  torch::Tensor Y;
  ActCtx gc; const ActCtx* gcp = nullptr;
  if (layer_has_ctx(L)) { build_ctx(L, ids, false, gc); gcp = &gc; }
  if (nmax == 1 && !L.nx.empty()) {""")
rep("    Y = expert_mlps_nx(X, pk, L.g[0].size(0), numa);", "    Y = expert_mlps_nx_ctx(X, pk, L.g[0].size(0), numa, gcp);")
rep("    Y = expert_mlps(X, g, u, d, numa)[0];", "    Y = expert_mlps_ctx(X, g, u, d, numa, gcp)[0];")

rep("PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n",
    "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n  m.def(\"register_layer_act\", &register_layer_act);\n", 1)
open(p, "w").write(s)
print("M2 kernel patch applied")
