# Server acceptance (v0.1)

**Setup:** `vahana serve` (src/vahana/server.py + cli.py) on Qwen3-30B-A3B, 24 threads, checked with server_check.py (standard-library client).

## Run log
- **Attempt 1 (OLMoE).** Completions worked. Chat crashed the handler, because OLMoE is a base model without a chat template. **Fix:** a missing template returns HTTP 400; any other generation error returns HTTP 500 with a JSON body and the server keeps serving. Rerun on Qwen3-30B-A3B, which has a chat template.
- **Attempt 2.** `stop_respected` was false. The check asked for a `"\n"` stop, but the model wrote 40 digits without a newline (`finish_reason: length`): a flawed check. **Fix:** a stop string the free run is known to contain.

## Result (attempt 3, server_check.json): all checks pass
| Check | Result |
|---|---|
| `/health`, `/v1/models` | ok |
| Greedy completion repeated | identical (`" Paris. The capital of the United Kingdom is London..."`) |
| Chat completion (chat template applied) | ok |
| Streaming chat == non-streaming chat (41 chunks) | **identical** |
| Sampling, temperature 0.8, same seed twice | **identical** |
| Sampling, different seed | different |
| `stop: ["The capital"]` | output cut to `" Paris. "`, `finish_reason: stop` |
| Unknown request fields | ignored |
| 128-token completion | 6.5 s end to end, prefill included |
