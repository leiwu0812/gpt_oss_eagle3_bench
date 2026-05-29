#!/usr/bin/env bash
# Run INSIDE the TRT-LLM container, first time only.
set -euo pipefail

# Pin huggingface_hub < 1.0 because the TRT-LLM container's transformers
# requires huggingface-hub>=0.34,<1.0. A bare `-U` would pull 1.x and break
# `import tensorrt_llm`. Xet replaces the deprecated hf_transfer extra.
pip install -q "huggingface_hub[cli]>=0.34,<1.0" hf_transfer aiohttp requests
export HF_XET_HIGH_PERFORMANCE=1
unset HF_HUB_ENABLE_HF_TRANSFER || true

# If gated: huggingface-cli login --token "$HF_TOKEN"

huggingface-cli download openai/gpt-oss-120b \
  --local-dir /config/models/gpt-oss-120b --repo-type model

# long-context Eagle3 draft head. For the short version use:
#   nvidia/gpt-oss-120b-Eagle3
huggingface-cli download nvidia/gpt-oss-120b-Eagle3-long-context \
  --local-dir /config/models/eagle --repo-type model

# Drop the eagle.yaml shipped with this bench dir into the eagle model dir.
cp /workspace/bench/eagle.yaml /config/models/eagle/eagle.yaml

nvidia-smi
python -c "import tensorrt_llm; print('TRT-LLM:', tensorrt_llm.__version__)"
