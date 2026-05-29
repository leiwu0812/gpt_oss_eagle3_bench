#!/usr/bin/env bash
# Run INSIDE the TRT-LLM container, first time only.
set -euo pipefail

pip install -q -U "huggingface_hub[cli,hf_transfer]" aiohttp
export HF_HUB_ENABLE_HF_TRANSFER=1

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
