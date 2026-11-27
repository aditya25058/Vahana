# How close is each 4-bit model to the original?

Qwen3-30B-A3B, two 4-bit versions: Vahana's checkpoint, RedHatAI's `quantized.w4a16`, and llama.cpp's Q4_K_M GGUF. Both are judged against the bf16 original on the same tokens, with llama.cpp's own statistics.

## Protocol
- **Text and tokens.** The WikiText-2 test set, in 32 chunks of 512 tokens (16,384 tokens). Each chunk is scored on its second half, 8,160 scored tokens in all, as `llama-perplexity` does.
- **Reference.** The bf16 original, run through Vahana. Vahana gives the same bits as the transformers implementation. Its log-probabilities are saved in the file format of `llama-perplexity --kl-divergence-base` (`kld.py base`).
- **llama.cpp.** It scores itself with `llama-perplexity -m Q4_K_M.gguf --kl-divergence-base BASE --kl-divergence -c 512`. It reads the tokens from the reference file, so both models see identical tokens. Both runs report the same reference perplexity, 7.378.
- **Vahana.** `kld.py score` computes the same statistics with llama.cpp's formulas.
- **Implementation floor.** The bf16 original was converted to a bf16 GGUF with llama.cpp's `convert_hf_to_gguf.py` (b12494), then scored the same way. This is how far llama.cpp's implementation alone is from transformers, before any rounding to 4 bits.

## Results (laptop, `results_laptop.json`)

| against the bf16 original | PPL (original 7.378) | mean KLD | 99th-pct KLD | same top token | RMS Δp |
|---|---|---|---|---|---|
| llama.cpp, bf16 GGUF (implementation floor) | 7.418 | 0.0033 | 0.039 | 97.45% | 1.98% |
| llama.cpp Q4_K_M | 7.829 (+6.1%) | 0.0930 | 1.18 | 88.91% | 10.35% |
| **Vahana 4-bit (w4a16)** | **7.751 (+5.0%)** | **0.0651** | **0.84** | **90.64%** | **8.81%** |

- **Vahana's 4-bit stays closer to the original on every measure.** Its mean KL divergence is 30% lower and it picks the original's top token more often. The 99th percentile, the tokens where the two quantizations stray most, is lower too.
- **The comparison is fair to llama.cpp.** Its implementation alone is 0.0033 KLD from the reference, so nearly all of its 0.093 comes from Q4_K_M's rounding.
- **Vahana has no implementation gap.** In bf16 it gives exactly the reference's bits, so 0.0651 is the checkpoint's rounding alone. That is also why Vahana's bf16 row is not in the table: its KLD is 0 by construction.
- Both 4-bit models are under 10 GB of GPU memory in these runs, and the quality numbers do not depend on the machine.

**Caveats.**
- The reference's logits are bf16, as transformers produces them. Vahana's 4-bit logits are also bf16; llama.cpp's are fp32.
- llama.cpp's file format stores log-probabilities to 16 bits, and only those within 16 nats of the most likely token. That floor makes the reference perplexity read slightly low, the same for every row.
- This is one text (WikiText-2) and one model. KLD on other text can differ.

## Reproduce
```
python kld.py base  Qwen3-30B-A3B wiki.test.raw base.dat 32
python kld.py score Qwen3-30B-A3B-w4a16 base.dat
llama-perplexity -m Qwen3-30B-A3B-Q4_K_M.gguf --kl-divergence-base base.dat --kl-divergence -c 512
```
Or run all of it with `python experiments/reproduce/vs_llamacpp.py --quality`.
