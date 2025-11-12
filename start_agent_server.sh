#!/bin/bash
# Start vLLM server for agent model (Qwen2.5-7B-Instruct) on port 8000
# Uses GPUs 1,2

export CUDA_VISIBLE_DEVICES=1,2

vllm serve Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32768 \
    --tensor-parallel-size 2 \
    --dtype auto \
    --trust-remote-code \
    --enable-prefix-caching \
    --enable-chunked-prefill
