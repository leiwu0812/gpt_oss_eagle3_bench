#!/usr/bin/env bash
# Official trtllm-bench throughput run (INSIDE container).
set -euo pipefail

MODEL=/config/models/gpt-oss-120b
DATASET=/workspace/bench/synthetic_512_512.jsonl
REPORT=/workspace/bench/trtllm_bench_report.json

# Generate a 1000-request synthetic dataset (input=512 tok, output=512 tok)
# using the base model's tokenizer. Tokenizer-based generation is more robust
# than relying on the in-tree dataset module, whose path/CLI moves between
# TRT-LLM releases (e.g. tensorrt_llm.serve.scripts.benchmark_dataset in
# 1.1.0rc2 vs tensorrt_llm.bench.benchmark.dataset in older builds).
python3 - <<PY
import json, random
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("${MODEL}", trust_remote_code=True)
vocab = tok.vocab_size
random.seed(0)
with open("${DATASET}", "w") as f:
    for i in range(1000):
        ids = [random.randint(10, vocab - 100) for _ in range(512)]
        f.write(json.dumps({"task_id": i, "input_ids": ids, "output_tokens": 512}) + "\n")
print("wrote", "${DATASET}")
PY

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
