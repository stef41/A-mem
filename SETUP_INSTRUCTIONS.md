# A-MEM LoCoMo Baseline Evaluation Setup

## Required Model Servers

You need to run TWO separate vLLM/SGLang servers:

### 1. Agent Model Server (Port 8000) - ALREADY RUNNING ✓
```bash
# Currently running on GPUs 1-3 (or similar configuration)
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.6 \
    --max-model-len 32768 \
    --tensor-parallel-size 3 \
    --dtype auto \
    --trust-remote-code \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
```

### 2. Judge Model Server (Port 8001) - NEEDS TO BE STARTED
```bash
# Run this on GPUs 4-7
CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve openai/gpt-oss-120b \
    --host 0.0.0.0 \
    --port 8001 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 8192 \
    --tensor-parallel-size 4 \
    --dtype auto \
    --trust-remote-code
```

## Running the Evaluation

Once both servers are running, execute:

```bash
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
    --dataset "data/locomo10.json"
```

