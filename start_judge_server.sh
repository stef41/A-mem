#!/bin/bash

# Start vLLM server for judge model (gpt-oss-120b) on port 8001
# Uses GPUs 4,5,6,7

export CUDA_VISIBLE_DEVICES=4,5,6,7

vllm serve openai/gpt-oss-120b \
    --host 0.0.0.0 \
    --port 8001 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096 \
    --tensor-parallel-size 4 \
    --dtype auto \
    --trust-remote-code \
    --enable-prefix-caching
