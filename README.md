# Agentic Memory 🧠

A novel agentic memory system for LLM agents that can dynamically organize memories in an agentic way.

## Quick Start: Running the Complete Evaluation

**One-Command Launch:** Run the complete setup with a single script that clears GPUs, launches all models, and runs evaluation:
```bash
./launch_and_eval.sh
```

This script will:
1. **Clear all GPU processes** to ensure clean state
2. **Launch Qwen2.5-7B-Instruct** (agent model) on GPU 1-2 at port 8000
3. **Launch GPT-OSS-120B** (judge model) on GPU 4,5,6,7 at port 8001  
4. **Run evaluation** with Qwen3-Embedding-0.6B (embedding model loaded on GPU 0)
5. **Clean up** GPU processes after completion

**Manual Server Launch (if needed):**
- Qwen agent model: `CUDA_VISIBLE_DEVICES=1,2 python -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --port 8000 --tp 2`
- GPT-OSS-120B judge: `CUDA_VISIBLE_DEVICES=4,5,6,7 python -m sglang.launch_server --model-path openai/gpt-oss-120b --port 8001 --tp 4`
- Embedding model is loaded automatically by the eval script on GPU 0

**Evaluation Command:**
```bash
python -u eval_with_judge.py \
    --agent_model "Qwen/Qwen2.5-7B-Instruct" \
    --judge_model "openai/gpt-oss-120b" \
    --embedding_model "Qwen/Qwen3-Embedding-0.6B" \
    --dataset "data/locomo10.json" \
    --backend "sglang" \
    --sglang_host "http://localhost" \
    --sglang_port 8000 \
    --judge_port 8001 \
    --retrieve_k 10 \
    --temperature_c5 0.5 \
    --ratio 1.0
```

---

> **Note:** This repository is specifically designed to reproduce the results presented in our paper. If you want to use the A-MEM system in building your agents, please refer to our official implementation at: [A-mem-sys](https://github.com/WujiangXu/A-mem-sys)

For more details, please refer to our paper: [A-MEM: Agentic Memory for LLM Agents](https://arxiv.org/pdf/2502.12110)

## Introduction 🌟

Large Language Model (LLM) agents have demonstrated remarkable capabilities in handling complex real-world tasks through external tool usage. However, to effectively leverage historical experiences, they require sophisticated memory systems. Traditional memory systems, while providing basic storage and retrieval functionality, often lack advanced memory organization capabilities.

Our project introduces an innovative **Agentic Memory** system that revolutionizes how LLM agents manage and utilize their memories:

<div align="center">
  <img src="Figure/intro-a.jpg" alt="Traditional Memory System" width="600"/>
  <img src="Figure/intro-b.jpg" alt="Our Proposed Agentic Memory" width="600"/>
  <br>
  <em>Comparison between traditional memory system (top) and our proposed agentic memory (bottom). Our system enables dynamic memory operations and flexible agent-memory interactions.</em>
</div>

## Key Features ✨

- 🔄 Dynamic memory organization based on Zettelkasten principles
- 🔍 Intelligent indexing and linking of memories
- 📝 Comprehensive note generation with structured attributes
- 🌐 Interconnected knowledge networks
- 🔄 Continuous memory evolution and refinement
- 🤖 Agent-driven decision making for adaptive memory management

## Framework 🏗️

<div align="center">
  <img src="Figure/framework.jpg" alt="Agentic Memory Framework" width="800"/>
  <br>
  <em>The framework of our Agentic Memory system showing the dynamic interaction between LLM agents and memory components.</em>
</div>

## How It Works 🛠️

When a new memory is added to the system:
1. Generates comprehensive notes with structured attributes
2. Creates contextual descriptions and tags
3. Analyzes historical memories for relevant connections
4. Establishes meaningful links based on similarities
5. Enables dynamic memory evolution and updates

## Results 📊

Empirical experiments conducted on six foundation models demonstrate superior performance compared to existing SOTA baselines.

## Getting Started 🚀

### Option 1: Using the existing Python environment (Recommended for this setup)

If you already have the required dependencies installed:

```bash
# Just run the complete evaluation
./launch_and_eval.sh
```

### Option 2: Using Pipenv

1. Install Pipenv:
```bash
pip install --user pipenv
```

2. Install dependencies:
```bash
pipenv install
```

3. Run the complete evaluation:
```bash
pipenv shell
./launch_and_eval.sh
```

### Option 3: Using venv or Conda (Original method)

1. Clone the repository:
```bash
git clone https://github.com/WujiangXu/AgenticMemory.git
cd AgenticMemory
```

2. Install dependencies:

**Using venv (Python virtual environment)**
```bash
# Create and activate virtual environment
python -m venv a-mem
source a-mem/bin/activate  # Linux/Mac
a-mem\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

**Using Conda**
```bash
# Create and activate conda environment
conda create -n myenv python=3.9
conda activate myenv

# Install dependencies
pip install -r requirements.txt
```

3. Run the experiments in LoCoMo dataset:
```bash
python test_advanced.py 
```

**Note:** To achieve the optimal performance reported in our paper, please adjust the hyperparameter k value accordingly.

**Categories Information:** The LoCoMo dataset contains the following categories:
* Category 1: Multi-hop
* Category 2: Temporal
* Category 3: Open-domain
* Category 4: Single-hop
* Category 5: Adversarial

For more details about the categories, please refer to [this GitHub issue](https://github.com/snap-research/locomo/issues/6). 

## Citation 📚

If you use this code in your research, please cite our work:

```bibtex
@inproceedings{xu2025amem,
  title={A-mem: Agentic memory for llm agents},
  author={Xu, Wujiang and Liang, Zujie and Mei, Kai and Gao, Hang and Tan, Juntao and Zhang, Yongfeng},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```

## License 📄

This project is licensed under the MIT License. See LICENSE for details.


