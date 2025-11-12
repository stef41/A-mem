#!/bin/bash
# Launch all required servers for A-MEM evaluation
# GPU 0: Embedding model (handled by eval script - uses CPU/GPU0 as needed)
# GPU 1-2: Agent model (Qwen2.5-7B-Instruct) on port 8000
# GPU 4-7: Judge model (gpt-oss-120b) on port 8001

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting all servers for A-MEM evaluation...${NC}"

# Kill any existing servers
echo "Cleaning up any existing servers..."
pkill -f "vllm serve Qwen/Qwen2.5-7B-Instruct" 2>/dev/null || true
pkill -f "vllm serve openai/gpt-oss-120b" 2>/dev/null || true
sleep 5

# Start agent server on GPUs 1-2
echo -e "${GREEN}Starting agent server (Qwen2.5-7B-Instruct) on GPUs 1-2...${NC}"
nohup ./start_agent_server.sh > logs/agent_server.log 2>&1 &
AGENT_PID=$!
echo "Agent server started with PID: $AGENT_PID"

# Wait a bit before starting the next server
sleep 10

# Start judge server on GPUs 4-7
echo -e "${GREEN}Starting judge server (gpt-oss-120b) on GPUs 4-7...${NC}"
nohup ./start_judge_server.sh > logs/judge_server.log 2>&1 &
JUDGE_PID=$!
echo "Judge server started with PID: $JUDGE_PID"

echo ""
echo -e "${YELLOW}Servers starting up...${NC}"
echo "Agent server log: logs/agent_server.log"
echo "Judge server log: logs/judge_server.log"
echo ""
echo "Monitor startup with:"
echo "  tail -f logs/agent_server.log"
echo "  tail -f logs/judge_server.log"
echo ""
echo "Check if servers are ready:"
echo "  curl http://localhost:8000/v1/models"
echo "  curl http://localhost:8001/v1/models"
echo ""
echo -e "${GREEN}Note: Embedding model (Qwen3-0.6B) will run on CPU during evaluation${NC}"
