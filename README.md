# GPT-OSS-120B + Eagle3 benchmark on 8x B200

End-to-end scripts for running NVIDIA's Eagle3 speculative-decoding head on top
of `openai/gpt-oss-120b` with TensorRT-LLM, on a single 8x B200 host.

## Files

| File | Where it runs | Purpose |
|---|---|---|
| `run_docker.sh`     | host      | Launch the TRT-LLM release container with GPUs + ports + volume mounts |
| `setup_inside.sh`   | container | Download base model + Eagle3 draft + drop `eagle.yaml` in place |
| `eagle.yaml`        | container | Speculative-decoding + KV cache + MoE config consumed by `trtllm-serve` |
| `start_server.sh`   | container | Start the OpenAI-compatible `trtllm-serve` endpoint on :8000 |
| `bench_client.py`   | container | Async client to measure throughput / p50 / p95 latency |
| `bench_trtllm.sh`   | container | Run the official `trtllm-bench throughput` harness |
| `bench_mtbench_acceptance.py` | container | Per-category MT-Bench acceptance rate (Eagle3 spec decoding) |

Volume layout once running:

- host `/data/models`              <-> container `/config/models`
- host `/home/leiwu/gpt_oss_eagle3_bench` <-> container `/workspace/bench`

## Quick start

```bash
# host
bash /home/leiwu/gpt_oss_eagle3_bench/run_docker.sh

# inside the container, first time only
bash /workspace/bench/setup_inside.sh

# terminal A inside container
bash /workspace/bench/start_server.sh

# terminal B inside container (after "Application startup complete")
curl -s -w "%{http_code}\n" http://localhost:8000/health
python /workspace/bench/bench_client.py --concurrency 1  --requests 20
python /workspace/bench/bench_client.py --concurrency 10 --requests 200
bash   /workspace/bench/bench_trtllm.sh

# per-category MT-Bench acceptance rate (uses /metrics; needs
# enable_iter_perf_stats: true in eagle.yaml, already set in this repo)
pip install requests
python /workspace/bench/bench_mtbench_acceptance.py --base-url http://localhost:8000
# If your TRT-LLM build exposes /metrics in an unexpected shape:
python /workspace/bench/bench_mtbench_acceptance.py --dump-metrics
```

### About the per-category acceptance script

- Sends the 80 MT-Bench turn-1 prompts **serially** (concurrency must be 1 to
  attribute iter stats to the right prompt — `/metrics` is aggregate, not
  per-request).
- Around each request, drains `/metrics`, sends the request, polls `/metrics`
  again, and sums `accepted_draft_tokens` / `total_draft_tokens` from the
  iter-stats delta. Output goes to stdout and `mtbench_acceptance.json`.
- The PyTorch backend's `/metrics` endpoint is **beta** per NVIDIA docs and
  field names can shift between TRT-LLM releases. The script tries several
  common spellings; if it prints `total=0` for every prompt, run
  `--dump-metrics` to inspect the raw payload and adjust `ACCEPTED_KEYS` /
  `TOTAL_KEYS` in the script.
- NVIDIA's published MT-Bench numbers for this model (writing 2.24, reasoning
  2.47, math 2.83, coding 2.51 tokens/step) come from the long-context Eagle3
  head with `max_draft_len: 3` — same config this repo uses.

## Variants

- **Short context (8k)**: `MAX_SEQ_LEN=8192 MAX_NUM_TOK=8192 bash start_server.sh`
- **Short Eagle3 head**: edit `setup_inside.sh` to use `nvidia/gpt-oss-120b-Eagle3`
- **Baseline (no spec decoding)**: start the server without `--extra_llm_api_options`
- **Tune draft len**: edit `eagle.yaml` `max_draft_len` (2 / 3 / 4) and re-run

## Notes / gotchas

- B200 needs NVIDIA driver >= 560 on the host.
- Eagle3 + GPT-OSS fusion is solid from TRT-LLM `1.1.0rc1`; pin the NGC tag.
- Recommended parallelism on 8x B200 is `tp=8, ep=4`. `ep=8` is usually slower.
- The Eagle3 endpoint forces greedy decoding; `temperature` / `top_p` / `seed`
  are ignored, so don't rely on them in client code.
- `speculative_model_dir` must point to a directory, not a file.
- For 128k seq lengths, drop `free_gpu_memory_fraction` from 0.8 to 0.7 if OOM.
