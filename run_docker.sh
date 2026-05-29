#!/usr/bin/env bash
# Launch the TensorRT-LLM release container on an 8x B200 host.
set -euo pipefail

# === Edit these paths for your host ===
MODELS_DIR=/data/models                                 # where HF weights are cached
WORK_DIR=/home/leiwu/gpt_oss_eagle3_bench               # scripts + logs + bench results
TRTLLM_TAG=${TRTLLM_TAG:-1.1.0rc2}                      # check NGC: nvcr.io/nvidia/tensorrt-llm/release

mkdir -p "${MODELS_DIR}" "${WORK_DIR}"

docker run --rm -it \
  --name gpt-oss-eagle3 \
  --ipc=host --ulimit stack=67108864 --ulimit memlock=-1 \
  --gpus all --shm-size=32g \
  -p 8000:8000 \
  -e HF_XET_HIGH_PERFORMANCE=1 \
  -v "${MODELS_DIR}":/config/models:rw \
  -v "${WORK_DIR}":/workspace/bench:rw \
  nvcr.io/nvidia/tensorrt-llm/release:${TRTLLM_TAG} \
  /bin/bash
