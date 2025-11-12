#!/bin/bash

# Launch A-MEM LoCoMo Baseline Evaluation
# Requires both agent server (port 8000) and judge server (port 8001) to be running

cd /data/users/zacharie/A-mem

python eval_with_judge.py \
    --agent_model "Qwen/Qwen2.5-7B-Instruct" \
    --judge_model "openai/gpt-oss-120b" \
    --embedding_model "Qwen/Qwen3-0.6B" \
    --backend sglang \
    --sglang_host "http://localhost" \
    --sglang_port 8000 \
    --judge_port 8001 \
    --retrieve_k 10 \
    --temperature_c5 0.5 \
    --ratio 1.0 \
    --dataset "data/locomo10.json" \
    2>&1 | tee logs/eval_ours_Qwen/eval_amem_baseline_$(date +%Y%m%d_%H%M%S).log
