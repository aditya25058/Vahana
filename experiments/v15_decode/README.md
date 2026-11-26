# v1.4 decode: closing the gap to llama.cpp on the 4 GB laptop

## Question
On the laptop (Ryzen 7 4800H, 15 GB RAM, RTX 3050 Ti 4 GB), llama.cpp b12494 decodes Qwen3-30B-A3B Q4_K_M at 13.2–15.5 tok/s, and Vahana 1.3 decodes the w4a16 checkpoint at 9.4–11.4 tok/s (experiments/v14_llamacpp). Where does Vahana's decode step go, and can it match or beat llama.cpp without changing a bit?

## Pre-registered bars (written 2026-11-27, before profiling)
1. **Profile first.** A per-step breakdown is written down before any optimisation: CPU expert time, GPU time, host overhead, and disk reads per token.
2. **Bits:**
   - Any change that keeps Vahana's existing arithmetic must give identical logits, checked by the selftest and by receipt verification against the 1.3 outputs.
   - A change that alters the arithmetic (for example integer dot products) must be a new, opt-in mode with its own references. It must also be exact across CPU and GPU, and its quality must be measured against bf16.
3. **Target:** Vahana decode at least 1.0x llama.cpp's on the same laptop, measured back to back in the same session, median of 3 runs each, 64 tokens after a 512-token prompt. **Stretch:** 1.2x.
4. No regression on the H100 (Qwen3-30B w4a16 decode ≥ 55 tok/s, the 1.3 bench).

## Results
(pending)
