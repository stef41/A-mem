#!/bin/bash

# A-MEM Complete Evaluation Script
# This script:
# 1. Clears all GPU processes
# 2. Launches embedding model on GPU 0
# 3. Launches Qwen model on GPU 1-2
# 4. Launches GPT-OSS-120B on GPU 4-7
# 5. Runs evaluation
# 6. Cleans up GPU processes after completion

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
QWEN_PORT=8000
JUDGE_PORT=8001
MAX_RETRIES=30
RETRY_DELAY=10

# Log file
LOGDIR="logs"
mkdir -p "$LOGDIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo -e "${GREEN}=== A-MEM Evaluation Setup ===${NC}"
echo "Timestamp: $TIMESTAMP"

# Function to kill processes on GPU
kill_gpu_processes() {
    echo -e "${YELLOW}Clearing GPU processes...${NC}"
    nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null || true
    sleep 3
    echo -e "${GREEN}GPU processes cleared${NC}"
}

# Function to check if server is ready
check_server() {
    local port=$1
    local name=$2
    local retries=0
    
    echo -e "${YELLOW}Waiting for $name to be ready on port $port...${NC}"
    while [ $retries -lt $MAX_RETRIES ]; do
        if curl -s "http://localhost:$port/health" > /dev/null 2>&1 || \
           curl -s "http://localhost:$port/v1/models" > /dev/null 2>&1; then
            echo -e "${GREEN}$name is ready!${NC}"
            return 0
        fi
        retries=$((retries + 1))
        echo "Attempt $retries/$MAX_RETRIES - waiting ${RETRY_DELAY}s..."
        sleep $RETRY_DELAY
    done
    
    echo -e "${RED}$name failed to start after $MAX_RETRIES attempts${NC}"
    return 1
}

# Function to cleanup on exit
cleanup() {
    echo -e "${YELLOW}Cleaning up...${NC}"
    kill_gpu_processes
    echo -e "${GREEN}Cleanup complete${NC}"
}

# Set trap to cleanup on exit
trap cleanup EXIT

# Step 1: Clear GPU processes before starting
kill_gpu_processes

# Step 2: Launch Qwen Model on GPU 1-2
echo -e "${GREEN}=== Launching Qwen Model (Qwen2.5-7B-Instruct) on GPU 1-2 ===${NC}"
CUDA_VISIBLE_DEVICES=1,2 python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-7B-Instruct \
    --port $QWEN_PORT \
    --host 0.0.0.0 \
    --tp 2 \
    > "$LOGDIR/qwen_server_$TIMESTAMP.log" 2>&1 &
QWEN_PID=$!
echo "Qwen server PID: $QWEN_PID"

# Wait for Qwen server to be ready
if ! check_server $QWEN_PORT "Qwen Model"; then
    echo -e "${RED}Failed to start Qwen server${NC}"
    exit 1
fi

# Step 3: Launch GPT-OSS-120B on GPU 4-7
echo -e "${GREEN}=== Launching GPT-OSS-120B (Judge Model) on GPU 4,5,6,7 ===${NC}"
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m sglang.launch_server \
    --model-path openai/gpt-oss-120b \
    --port $JUDGE_PORT \
    --host 0.0.0.0 \
    --tp 4 \
    > "$LOGDIR/judge_server_$TIMESTAMP.log" 2>&1 &
JUDGE_PID=$!
echo "Judge server PID: $JUDGE_PID"

# Wait for judge server to be ready
if ! check_server $JUDGE_PORT "Judge Model (GPT-OSS-120B)"; then
    echo -e "${RED}Failed to start judge server${NC}"
    exit 1
fi

# Step 4: Display server status
echo -e "${GREEN}=== All Servers Running ===${NC}"
echo "Qwen Model: http://localhost:$QWEN_PORT (GPU 1-2)"
echo "Judge Model (GPT-OSS-120B): http://localhost:$JUDGE_PORT (GPU 4-7)"
echo "Embedding Model (Qwen3-Embedding-0.6B): Loaded locally on GPU 0 by eval script"
echo ""

# Step 5: Run Evaluation
echo -e "${GREEN}=== Starting Evaluation ===${NC}"

# Run evaluation using the existing Python environment
cd /data/users/zacharie/A-mem
python -u eval_with_judge.py \
    --agent_model "Qwen/Qwen2.5-7B-Instruct" \
    --judge_model "openai/gpt-oss-120b" \
    --embedding_model "Qwen/Qwen3-Embedding-0.6B" \
    --dataset "data/locomo10.json" \
    --backend "sglang" \
    --sglang_host "http://localhost" \
    --sglang_port $QWEN_PORT \
    --judge_port $JUDGE_PORT \
    --retrieve_k 10 \
    --temperature_c5 0.5 \
    --ratio 1.0 \
    2>&1 | tee "$LOGDIR/evaluation_$TIMESTAMP.log"

EVAL_EXIT_CODE=${PIPESTATUS[0]}

# Step 6: Report results
echo ""
if [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}=== Evaluation Completed Successfully ===${NC}"
else
    echo -e "${RED}=== Evaluation Failed with exit code $EVAL_EXIT_CODE ===${NC}"
fi

echo "Logs saved to:"
echo "  - Qwen: $LOGDIR/qwen_server_$TIMESTAMP.log"
echo "  - Judge: $LOGDIR/judge_server_$TIMESTAMP.log"
echo "  - Evaluation: $LOGDIR/evaluation_$TIMESTAMP.log"

# Exit with evaluation status
exit $EVAL_EXIT_CODE
