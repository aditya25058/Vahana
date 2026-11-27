# Reproduce Vahana vs llama.cpp

One command runs the whole comparison on your machine (`vs_llamacpp.py`):

```bash
git clone https://github.com/vahana-ai/Vahana
pip install "vahana[gpu] @ git+https://github.com/vahana-ai/Vahana"     # or "vahana @ git+..." without a GPU
python Vahana/experiments/reproduce/vs_llamacpp.py                       # --quality adds the bf16 comparison
```

The script works in four steps:
1. **Download, after asking.** It fetches Vahana's model (RedHatAI's 4-bit Qwen3-30B-A3B, about 16.5 GB) and llama.cpp's model (Qwen's Q4_K_M GGUF, about 18.6 GB). It also fetches the llama.cpp b12494 release built for your OS and GPU.
2. **Tune llama.cpp.** It sweeps `--n-cpu-moe` with `llama-bench` and keeps llama.cpp's fastest setting on this machine. Vahana runs with its defaults.
3. **Run both engines on the same prompts.** They alternate, both warm:
   - decode speed on a benchmark text;
   - decode speed and time to first token on five real prompts (code, chat, math, story, Hindi).

   With `--quality`, both 4-bit models are also scored against the bf16 original on the same WikiText-2 tokens. This downloads the bf16 model, about 61 GB more.
4. **Report.** It prints a table and writes JSON and logs to `~/vahana_vs_llamacpp/results`, along with a description of the machine.

The protocols are described in [experiments/v15_decode](../v15_decode/README.md) and [experiments/v15_quality](../v15_quality/README.md).

## On a free Kaggle GPU (T4, 16 GB)

1. Create a new notebook.
   - Settings → Accelerator: **GPU T4 x2** (the script uses one).
   - Settings → Internet: **on**. Kaggle needs a phone-verified account for this.
2. Run these cells:

```python
!nvidia-smi --query-gpu=name,memory.total --format=csv; free -g; nproc; df -h /tmp /kaggle/working
!pip install -q "vahana[gpu] @ git+https://github.com/vahana-ai/Vahana"
!git clone -q https://github.com/vahana-ai/Vahana /kaggle/working/Vahana
!CUDA_VISIBLE_DEVICES=0 vahana selftest --device cuda
```

```python
!CUDA_VISIBLE_DEVICES=0 python /kaggle/working/Vahana/experiments/reproduce/vs_llamacpp.py --dir /tmp/vs --yes --copy-results /kaggle/working 2>&1 | tail -40
```

The second cell downloads about 35 GB and runs for about an hour. The models go to `/tmp`, because `/kaggle/working` holds only 20 GB. The results are copied to `/kaggle/working/vs_llamacpp_results`.

## Results so far

**Google Colab, free T4 (15 GB), 12.7 GB RAM, 1 CPU core (2 threads)**, run with this script (`results_colab_t4.json`):
- **llama.cpp's tuning.** Its best was `--n-cpu-moe 12 -t 1`. The sweep ran 0.98 tok/s with all 48 layers' experts on the CPU, 5.15 with 12, and 8 did not fit the GPU. Two threads were slower (4.45).
- **Vahana.** Its defaults, plus the adaptive GPU expert cache.

| | llama.cpp | Vahana | |
|---|---|---|---|
| benchmark text, decode (6 runs each, ranges 8.09–9.37 vs 16.82–17.23) | 8.70 tok/s | **17.05 tok/s** | **1.96×** |
| five real prompts, decode (median of 10 each) | 5.37 tok/s | **8.85 tok/s** | **1.65×** |
| time to first token, real prompts (median) | 12.76 s | **3.71 s** | **3.44× sooner** |

- On the real prompts Vahana was faster on 9 of the 10 prompt runs. The code prompt in block 0 was level: 4.72 vs 4.79 tok/s.
- In the same run, the T4 passed `vahana selftest --device cuda`: all 18 families gave the reference machine's bits, although the T4 has no native bf16.

**Laptop, RTX 3050 Ti (4 GB), 16 GB RAM**:
- **Decode:** 1.86× on the benchmark text, and 1.57× on the real prompts.
- **Time to first token:** 2.8× sooner.
- See [experiments/v15_decode](../v15_decode/README.md), section 9.

**Quality against the bf16 original** (the same on any machine): Vahana's 4-bit has mean KL divergence 0.065 against llama.cpp Q4_K_M's 0.093. See [experiments/v15_quality](../v15_quality/README.md).
