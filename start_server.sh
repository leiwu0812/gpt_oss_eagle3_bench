#!/usr/bin/env bash
# Launch trtllm-serve with Eagle3 speculative decoding (run INSIDE container).
set -euo pipefail

export TRTLLM_ENABLE_PDL=1

# For short-context (8k) benchmarks, drop both max values to 8192.
MAX_SEQ_LEN=${MAX_SEQ_LEN:-131072}
MAX_NUM_TOK=${MAX_NUM_TOK:-131072}

trtllm-serve /config/models/gpt-oss-120b \
  --host 0.0.0.0 --port 8000 \
  --backend pytorch \
  --max_batch_size 10 \
  --tp_size 8 --ep_size 4 \
  --trust_remote_code \
  --extra_llm_api_options /config/models/eagle/eagle.yaml \
  --max_num_tokens ${MAX_NUM_TOK} \
  --max_seq_len ${MAX_SEQ_LEN} \
  2>&1 | tee /workspace/bench/server.log
