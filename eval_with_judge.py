from memory_layer import LLMController, AgenticMemorySystem
import os
import json
import argparse
import logging
import sys
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from load_dataset import load_locomo_dataset, QA, Turn, Session, Conversation
import nltk
from sentence_transformers import SentenceTransformer
import statistics
from collections import defaultdict
import pickle
import random
from tqdm import tqdm
from utils import calculate_metrics, aggregate_metrics
from datetime import datetime
import requests
import time

# Force unbuffered output for continuous logging
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')

# Judge prompt
ACCURACY_PROMPT = """
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question (posed by one user to another user),
    (2) a 'gold' (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

Return your judgment in JSON format with the key "label" set to either "CORRECT" or "WRONG".
Do NOT include both CORRECT and WRONG in your response.

Example response: {{"label": "CORRECT"}}
"""

class JudgeLLM:
    """LLM Judge for evaluating answers"""
    def __init__(self, model: str, backend: str = "vllm", vllm_host: str = "http://localhost", vllm_port: int = 8000, judge_port: int = None):
        self.model = model
        self.backend = backend
        self.vllm_host = vllm_host
        # Use separate judge_port if provided, otherwise use same port as agent
        self.vllm_port = judge_port if judge_port is not None else vllm_port
        self.base_url = f"{vllm_host}:{self.vllm_port}"
    
    def judge_answer(self, question: str, gold_answer: str, generated_answer: str) -> Dict:
        """Judge if generated answer is correct - retry on failure"""
        prompt = ACCURACY_PROMPT.format(
            question=question,
            gold_answer=gold_answer,
            generated_answer=generated_answer
        )
        
        max_retries = None  # Infinite retries
        retry_delay = 5  # seconds
        attempt = 0
        
        while True:
            attempt += 1
            try:
                # Use vLLM API
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are a helpful judge evaluating answers."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.0,
                    "max_tokens": 2048,  # Large enough for reasoning models with long reasoning chains
                    "response_format": {"type": "json_object"}
                }
                
                response = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    message = result.get("choices", [{}])[0].get("message", {})
                    
                    # Try to get content from either 'content' or 'reasoning_content' field
                    content = message.get("content")
                    if content is None:
                        # For reasoning models (o1-style), check reasoning_content
                        content = message.get("reasoning_content")
                    
                    if content is None:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] Judge returned None content on attempt {attempt}", flush=True)
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] Response structure: {result}", flush=True)
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] Retrying in {retry_delay} seconds...", flush=True)
                        time.sleep(retry_delay)
                        continue
                    
                    # Parse JSON response
                    try:
                        judgment = json.loads(content)
                        label = judgment.get("label", "WRONG").upper()
                        return {
                            "label": label,
                            "raw_response": content,
                            "correct": 1 if label == "CORRECT" else 0
                        }
                    except json.JSONDecodeError:
                        # Fallback: search for CORRECT/WRONG in response
                        if "CORRECT" in content.upper() and "WRONG" not in content.upper():
                            return {"label": "CORRECT", "raw_response": content, "correct": 1}
                        else:
                            return {"label": "WRONG", "raw_response": content, "correct": 0}
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Judge API returned status {response.status_code}: {response.text}", flush=True)
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Attempt {attempt} failed. Retrying in {retry_delay} seconds...", flush=True)
                    time.sleep(retry_delay)
                    continue
                    
            except Exception as e:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Judge error on attempt {attempt}: {e}", flush=True)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Retrying in {retry_delay} seconds...", flush=True)
                time.sleep(retry_delay)
                continue

class advancedMemAgent:
    def __init__(self, model, backend, retrieve_k, temperature_c5, embedding_model, 
                 sglang_host="http://localhost", sglang_port=8000):
        # Use the specified embedding model directly
        self.memory_system = AgenticMemorySystem(
            model_name=embedding_model,
            llm_backend=backend,
            llm_model=model,
            sglang_host=sglang_host,
            sglang_port=sglang_port,
            evo_threshold=100  # Enable evolution feature as per A-MEM paper (default: 100)
        )
        self.retriever_llm = LLMController(
            backend=backend, 
            model=model, 
            api_key=None, 
            sglang_host=sglang_host, 
            sglang_port=sglang_port
        )
        self.retrieve_k = retrieve_k
        self.temperature_c5 = temperature_c5

    def add_memory(self, content, time=None):
        self.memory_system.add_note(content, time=time)

    def retrieve_memory(self, content, k=10):
        return self.memory_system.find_related_memories_raw(content, k=k)
    
    def generate_query_llm(self, question):
        prompt = f"""Given the following question, generate several keywords, using 'cosmos' as the separator.

                Question: {question}

                Format your response as a JSON object with a "keywords" field containing the selected text. 

                Example response format:
                {{"keywords": "keyword1, keyword2, keyword3"}}"""
            
        response = self.retriever_llm.llm.get_completion(prompt,response_format={"type": "json_schema", "json_schema": {
                            "name": "response",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "keywords": {
                                        "type": "string",
                                    }
                                },
                                "required": ["keywords"],
                                "additionalProperties": False
                            },
                            "strict": True
                        }})
        print("response:{}".format(response))
        try:
            response = json.loads(response)["keywords"]
        except:
            response = response.strip()
        return response

    def answer_question(self, question: str, category: int, answer: str) -> str:
        """Generate answer for a question given the conversation context."""
        keywords = self.generate_query_llm(question)
        raw_context = self.retrieve_memory(keywords, k=self.retrieve_k)
        context = raw_context
        
        # Log retrieved context for debugging
        print(f"\n{'='*80}")
        print(f"QUESTION: {question}")
        print(f"KEYWORDS: {keywords}")
        print(f"RETRIEVED CONTEXT ({len(raw_context)} chars):")
        print(f"{raw_context}")
        print(f"{'='*80}\n")
        
        assert category in [1,2,3,4,5]
        user_prompt = f"""Context:
                {context}

                Question: {question}

                Answer the question based only on the information provided in the context above."""
        temperature = 0.7
        if category == 5: # adversarial question
            answer_tmp = list()
            if random.random() < 0.5:
                answer_tmp.append('Not mentioned in the conversation')
                answer_tmp.append(answer)
            else:
                answer_tmp.append(answer)
                answer_tmp.append('Not mentioned in the conversation')
            user_prompt = f"""
                            Based on the context: {context}, answer the following question. {question} 
                            
                            Select the correct answer: {answer_tmp[0]} or {answer_tmp[1]}  Short answer:
                            """
            temperature = self.temperature_c5
        elif category == 2:
            user_prompt = f"""
                            Based on the context: {context}, answer the following question. Use DATE of CONVERSATION to answer with an approximate date.
                            Please generate the shortest possible answer, using words from the conversation where possible, and avoid using any subjects.   

                            Question: {question} Short answer:
                            """
        elif category == 3:
            user_prompt = f"""
                            Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

                            Question: {question} Short answer:
                            """
        else:
            user_prompt = f"""Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

                            Question: {question} Short answer:
                            """
        response = self.memory_system.llm_controller.llm.get_completion(
            user_prompt,response_format={"type": "json_schema", "json_schema": {
                        "name": "response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "answer": {
                                    "type": "string",
                                }
                            },
                            "required": ["answer"],
                            "additionalProperties": False
                        },
                        "strict": True
                    }},temperature=temperature
        )
        return response, user_prompt, raw_context

def setup_logger(log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger('locomo_eval')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def print_metrics_table(category_results: Dict):
    """Print metrics in the requested table format"""
    print("\n" + "="*120)
    print("LOCOMO EVALUATION RESULTS")
    print("="*120)
    
    # Category mapping
    category_names = {
        1: "Single-Hop",
        2: "Temporal",
        3: "Multi-Hop",
        4: "Open Domain",
        5: "Adversarial"
    }
    
    # Print header
    print(f"\n{'Method':<15}", end="")
    for cat_id in [1, 3, 2, 4]:  # Order: Single-Hop, Multi-Hop, Temporal, Open Domain
        print(f"{category_names[cat_id]:<30}", end="")
    print(f"{'Overall':<30}")
    
    print(f"{'':<15}", end="")
    for _ in range(5):  # 4 categories + overall
        print(f"{'F1':<10}{'B1':<10}{'J':<10}", end="")
    print()
    
    print("-" * 120)
    
    # Print results row
    print(f"{'A-MEM':<15}", end="")
    
    # Print each category in order
    for cat_id in [1, 3, 2, 4]:  # Single-Hop, Multi-Hop, Temporal, Open Domain
        cat_key = f"category_{cat_id}"
        if cat_key in category_results:
            f1 = category_results[cat_key].get('f1', {}).get('mean', 0.0) * 100
            b1 = category_results[cat_key].get('bleu1', {}).get('mean', 0.0) * 100
            judge_acc = category_results[cat_key].get('judge_accuracy', {}).get('mean', 0.0) * 100
            print(f"{f1:<10.2f}{b1:<10.2f}{judge_acc:<10.2f}", end="")
        else:
            print(f"{'N/A':<10}{'N/A':<10}{'N/A':<10}", end="")
    
    # Print overall
    if 'overall' in category_results:
        f1 = category_results['overall'].get('f1', {}).get('mean', 0.0) * 100
        b1 = category_results['overall'].get('bleu1', {}).get('mean', 0.0) * 100
        judge_acc = category_results['overall'].get('judge_accuracy', {}).get('mean', 0.0) * 100
        print(f"{f1:<10.2f}{b1:<10.2f}{judge_acc:<10.2f}")
    else:
        print(f"{'N/A':<10}{'N/A':<10}{'N/A':<10}")
    
    print("="*120)
    print("\nLegend: F1 = Token F1 Score, B1 = BLEU-1, J = Judge Accuracy")
    print("Note: All values are percentages")
    print("="*120 + "\n")

def evaluate_dataset(dataset_path: str, agent_model: str, judge_model: str, 
                     embedding_model: str, output_path: Optional[str] = None, 
                     ratio: float = 1.0, backend: str = "sglang", 
                     temperature_c5: float = 0.5, retrieve_k: int = 10, 
                     sglang_host: str = "http://localhost", sglang_port: int = 8000,
                     judge_port: int = None):
    """Evaluate the agent on the LoComo dataset with judge.
    
    Args:
        judge_port: Optional separate port for judge model. If None, uses sglang_port.
    """
    # Validate that models are available - retry until they are
    retry_delay = 10  # seconds
    while True:
        try:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Checking for agent model on {sglang_host}:{sglang_port}...", flush=True)
            response = requests.get(f"{sglang_host}:{sglang_port}/v1/models", timeout=5)
            if response.status_code == 200:
                available_models = [m["id"] for m in response.json()["data"]]
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Available models on server: {available_models}", flush=True)
                
                if agent_model not in available_models:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ⏳ Waiting for agent model '{agent_model}' to be available on server...", flush=True)
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Currently available: {available_models}", flush=True)
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Retrying in {retry_delay} seconds...", flush=True)
                    time.sleep(retry_delay)
                    continue
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Agent model '{agent_model}' is available", flush=True)
                
                # Check judge model on its port
                judge_check_port = judge_port if judge_port is not None else sglang_port
                if judge_check_port != sglang_port:
                    try:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] Checking for judge model on {sglang_host}:{judge_check_port}...", flush=True)
                        judge_response = requests.get(f"{sglang_host}:{judge_check_port}/v1/models", timeout=5)
                        if judge_response.status_code == 200:
                            judge_available_models = [m["id"] for m in judge_response.json()["data"]]
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] Available judge models on port {judge_check_port}: {judge_available_models}", flush=True)
                            if judge_model not in judge_available_models:
                                print(f"[{datetime.now().strftime('%H:%M:%S')}] ⏳ Waiting for judge model '{judge_model}' on port {judge_check_port}...", flush=True)
                                print(f"[{datetime.now().strftime('%H:%M:%S')}] Currently available: {judge_available_models}", flush=True)
                                print(f"[{datetime.now().strftime('%H:%M:%S')}] Retrying in {retry_delay} seconds...", flush=True)
                                time.sleep(retry_delay)
                                continue
                            else:
                                print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Judge model '{judge_model}' is available on port {judge_check_port}", flush=True)
                        else:
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] Judge server on port {judge_check_port} returned status {judge_response.status_code}", flush=True)
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] Retrying in {retry_delay} seconds...", flush=True)
                            time.sleep(retry_delay)
                            continue
                    except Exception as e:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] Could not connect to judge port {judge_check_port}: {e}", flush=True)
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] Retrying in {retry_delay} seconds...", flush=True)
                        time.sleep(retry_delay)
                        continue
                elif judge_model not in available_models:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ⏳ Waiting for judge model '{judge_model}' on port {sglang_port}...", flush=True)
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Currently available: {available_models}", flush=True)
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Retrying in {retry_delay} seconds...", flush=True)
                    time.sleep(retry_delay)
                    continue
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Judge model '{judge_model}' is available", flush=True)
                
                # All models are available, break out of retry loop
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ All required models are available. Starting evaluation...", flush=True)
                break
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Server returned status {response.status_code}", flush=True)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Retrying in {retry_delay} seconds...", flush=True)
                time.sleep(retry_delay)
                continue
                
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Error connecting to server: {e}", flush=True)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Retrying in {retry_delay} seconds...", flush=True)
            time.sleep(retry_delay)
            continue
    
    # Generate automatic log filename with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    log_filename = f"eval_with_judge_{agent_model.replace('/', '_')}_{backend}_ratio{ratio}_{timestamp}.log"
    log_path = os.path.join(os.path.dirname(__file__), "logs", "eval_ours_Qwen", log_filename)
    
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    logger = setup_logger(log_path)
    logger.info(f"Loading dataset from {dataset_path}")
    logger.info(f"Agent Model: {agent_model}")
    logger.info(f"Judge Model: {judge_model}")
    logger.info(f"Embedding Model: {embedding_model}")
    
    # Load dataset
    samples = load_locomo_dataset(dataset_path)
    logger.info(f"Loaded {len(samples)} samples")
    
    # Select subset of samples based on ratio
    if ratio < 1.0:
        num_samples = max(1, int(len(samples) * ratio))
        samples = samples[:num_samples]
        logger.info(f"Using {num_samples} samples ({ratio*100:.1f}% of dataset)")
    
    # Initialize judge
    judge = JudgeLLM(judge_model, backend="vllm", vllm_host=sglang_host, vllm_port=sglang_port, judge_port=judge_port)
    
    # Store results
    results = []
    all_metrics = []
    all_categories = []
    total_questions = 0
    category_counts = defaultdict(int)
    
    # Evaluate each sample
    error_num = 0
    memories_dir = os.path.join(os.path.dirname(__file__), 
                                f"cached_memories_advanced_{backend}_{agent_model.replace('/', '_')}")
    os.makedirs(memories_dir, exist_ok=True)
    allow_categories = [1, 2, 3, 4, 5]
    
    for sample_idx, sample in enumerate(tqdm(samples, desc="Processing samples")):
        agent = advancedMemAgent(agent_model, backend, retrieve_k, temperature_c5, 
                                embedding_model, sglang_host, sglang_port)
        
        # Create memory cache filename
        memory_cache_file = os.path.join(memories_dir, f"memory_cache_sample_{sample_idx}.pkl")
        retriever_cache_file = os.path.join(memories_dir, f"retriever_cache_sample_{sample_idx}.pkl")
        retriever_cache_embeddings_file = os.path.join(memories_dir, 
                                                       f"retriever_cache_embeddings_sample_{sample_idx}.npy")

        # Load or create memories
        if os.path.exists(memory_cache_file):
            logger.info(f"Loading cached memories for sample {sample_idx}")
            with open(memory_cache_file, 'rb') as f:
                cached_memories = pickle.load(f)
            agent.memory_system.memories = cached_memories
            
            if os.path.exists(retriever_cache_file):
                agent.memory_system.retriever = agent.memory_system.retriever.load(
                    retriever_cache_file, retriever_cache_embeddings_file)
            else:
                agent.memory_system.retriever = agent.memory_system.retriever.load_from_local_memory(
                    cached_memories, embedding_model)
            logger.info(f"Successfully loaded {len(cached_memories)} memories")
        else:
            logger.info(f"Creating new memories for sample {sample_idx}")
            for _, turns in sample.conversation.sessions.items():
                for turn in turns.turns:
                    turn_datetime = turns.date_time
                    conversation_tmp = "Speaker " + turn.speaker + " says: " + turn.text
                    agent.add_memory(conversation_tmp, time=turn_datetime)
            
            memories_to_cache = agent.memory_system.memories
            with open(memory_cache_file, 'wb') as f:
                pickle.dump(memories_to_cache, f)
            agent.memory_system.retriever.save(retriever_cache_file, retriever_cache_embeddings_file)
            logger.info(f"Successfully cached {len(memories_to_cache)} memories")
        
        logger.info(f"\nProcessing sample {sample_idx + 1}/{len(samples)}")
        
        # Process QA pairs
        for qa in sample.qa:
            if int(qa.category) in allow_categories:
                total_questions += 1
                category_counts[qa.category] += 1
                
                # Generate prediction
                prediction, user_prompt, raw_context = agent.answer_question(
                    qa.question, qa.category, qa.final_answer)
                try:
                    prediction = json.loads(prediction)["answer"]
                except:
                    prediction = prediction
                    logger.info(f"Failed to parse prediction as JSON: {prediction}")
                    error_num += 1
                
                # Calculate traditional metrics
                metrics = calculate_metrics(prediction, qa.final_answer) if qa.final_answer else {
                    "exact_match": 0, "f1": 0.0, "rouge1_f": 0.0, "rouge2_f": 0.0, 
                    "rougeL_f": 0.0, "bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, 
                    "bleu4": 0.0, "bert_f1": 0.0, "meteor": 0.0, "sbert_similarity": 0.0
                }
                
                # Get judge evaluation
                judge_result = judge.judge_answer(qa.question, qa.final_answer, prediction)
                metrics['judge_accuracy'] = judge_result['correct']
                metrics['judge_label'] = judge_result['label']
                
                # Log results
                logger.info(f"\nQuestion {total_questions}: {qa.question}")
                logger.info(f"Prediction: {prediction}")
                logger.info(f"Reference: {qa.final_answer}")
                logger.info(f"Judge: {judge_result['label']}")
                logger.info(f"Category: {qa.category}")
                
                all_metrics.append(metrics)
                all_categories.append(qa.category)
                
                # Store individual result
                result = {
                    "sample_id": sample_idx,
                    "question": qa.question,
                    "prediction": prediction,
                    "reference": qa.final_answer,
                    "category": qa.category,
                    "metrics": metrics,
                    "judge_result": judge_result
                }
                results.append(result)
    
    # Calculate aggregate metrics
    aggregate_results = aggregate_metrics(all_metrics, all_categories)
    
    # Print formatted table
    print_metrics_table(aggregate_results)
    
    # Prepare final results
    final_results = {
        "agent_model": agent_model,
        "judge_model": judge_model,
        "embedding_model": embedding_model,
        "dataset": dataset_path,
        "total_questions": total_questions,
        "category_distribution": {str(cat): count for cat, count in category_counts.items()},
        "aggregate_metrics": aggregate_results,
        "individual_results": results
    }
    logger.info(f"Error number: {error_num}")
    
    # Save results
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        logger.info(f"Results saved to {output_path}")
    
    # Log summary
    logger.info("\nEvaluation Summary:")
    logger.info(f"Total questions evaluated: {total_questions}")
    logger.info("\nCategory Distribution:")
    for category, count in sorted(category_counts.items()):
        logger.info(f"Category {category}: {count} questions ({count/total_questions*100:.1f}%)")
    
    return final_results

def main():
    parser = argparse.ArgumentParser(description="Evaluate agent on LoComo with LLM judge")
    parser.add_argument("--dataset", type=str, default="data/locomo10.json",
                      help="Path to the dataset file")
    parser.add_argument("--agent_model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                      help="Agent model to use")
    parser.add_argument("--judge_model", type=str, default="openai/gpt-oss-120b",
                      help="Judge model to use")
    parser.add_argument("--embedding_model", type=str, default="Qwen/Qwen3-0.6B",
                      help="Embedding model for retrieval")
    parser.add_argument("--output", type=str, default=None,
                      help="Path to save evaluation results")
    parser.add_argument("--ratio", type=float, default=1.0,
                      help="Ratio of dataset to evaluate (0.0 to 1.0)")
    parser.add_argument("--backend", type=str, default="sglang",
                      help="Backend to use (openai, ollama, or sglang)")
    parser.add_argument("--temperature_c5", type=float, default=0.5,
                      help="Temperature for adversarial questions")
    parser.add_argument("--retrieve_k", type=int, default=10,
                      help="Number of memories to retrieve")
    parser.add_argument("--sglang_host", type=str, default="http://localhost",
                      help="SGLang/vLLM server host")
    parser.add_argument("--sglang_port", type=int, default=8000,
                      help="SGLang/vLLM server port")
    parser.add_argument("--judge_port", type=int, default=None,
                      help="Optional separate port for judge model (if served separately)")
    args = parser.parse_args()
    
    if args.ratio <= 0.0 or args.ratio > 1.0:
        raise ValueError("Ratio must be between 0.0 and 1.0")
    
    # Use full model names as vLLM expects them with namespace
    agent_model = args.agent_model
    judge_model = args.judge_model
    
    # Convert relative path to absolute path
    dataset_path = os.path.join(os.path.dirname(__file__), args.dataset)
    if args.output:
        output_path = os.path.join(os.path.dirname(__file__), args.output)
    else:
        output_path = None
    
    evaluate_dataset(dataset_path, agent_model, judge_model, 
                    args.embedding_model, output_path, args.ratio, args.backend, 
                    args.temperature_c5, args.retrieve_k, args.sglang_host, args.sglang_port,
                    args.judge_port)

if __name__ == "__main__":
    main()
