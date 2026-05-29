#!/usr/bin/env bash
# Official trtllm-bench throughput run (INSIDE container).
set -euo pipefail

MODEL=/config/models/gpt-oss-120b
DATASET=/workspace/bench/synthetic_512_512.jsonl
REPORT=/workspace/bench/trtllm_bench_report.json

python -m tensorrt_llm.bench.benchmark.dataset \
  --tokenizer ${MODEL} \
  --num-requests 1000 \
  --input-mean 512 --input-stdev 0 \
  --output-mean 512 --output-stdev 0 \
  --output ${DATASET}

trtllm-bench --model openai/gpt-oss-120b --model_path ${MODEL} \
  throughput \
  --backend pytorch \
  --dataset ${DATASET} \
  --tp 8 --ep 4 \
  --extra_llm_api_options /config/models/eagle/eagle.yaml \
  --max_batch_size 10 \
  --max_num_tokens 8192 \
  --report_json ${REPORT}

echo "Report written to ${REPORT}"
