"""
Evaluation harness for ConsistencyBench.

Supports multiple models via OpenRouter (LiteLLM) and multiple prompting strategies:
1. Vanilla (direct prompting)
2. Chain-of-Thought (CoT)
3. Self-Consistency (SC) - sample k=5, majority vote
4. Consistency-Guided Decoding (CGD) - our method
"""

import json
import os
import time
import re
import asyncio
from collections import Counter
from typing import Optional
from tqdm import tqdm
from litellm import completion
import litellm

# Disable litellm verbose logging
litellm.suppress_debug_info = True

# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODELS = {
    # === OpenAI ===
    "gpt-5.2": "openrouter/openai/gpt-5.2",
    "gpt-5": "openrouter/openai/gpt-5",
    "gpt-5-mini": "openrouter/openai/gpt-5-mini",
    "gpt-4.1": "openrouter/openai/gpt-4.1",
    "gpt-4o": "openrouter/openai/gpt-4o-2024-11-20",
    "gpt-4o-mini": "openrouter/openai/gpt-4o-mini",
    "o3": "openrouter/openai/o3",
    # === Anthropic ===
    "claude-opus-4.6": "openrouter/anthropic/claude-opus-4.6",
    "claude-sonnet-4.6": "openrouter/anthropic/claude-sonnet-4.6",
    "claude-opus-4.5": "openrouter/anthropic/claude-opus-4.5",
    "claude-sonnet-4.5": "openrouter/anthropic/claude-sonnet-4.5",
    "claude-3.5-sonnet": "openrouter/anthropic/claude-3.5-sonnet",
    "claude-3.5-haiku": "openrouter/anthropic/claude-3.5-haiku",
    # === Google ===
    "gemini-2.5-pro": "openrouter/google/gemini-2.5-pro",
    "gemini-2.5-flash": "openrouter/google/gemini-2.5-flash",
    "gemini-2.0-flash": "openrouter/google/gemini-2.0-flash-001",
    # === DeepSeek ===
    "deepseek-v3.2": "openrouter/deepseek/deepseek-v3.2",
    "deepseek-r1": "openrouter/deepseek/deepseek-r1",
    "deepseek-v3": "openrouter/deepseek/deepseek-chat",
    # === Meta ===
    "llama-3.1-70b": "openrouter/meta-llama/llama-3.1-70b-instruct",
    "llama-3.3-70b": "openrouter/meta-llama/llama-3.3-70b-instruct",
    # === Qwen ===
    "qwen-2.5-72b": "openrouter/qwen/qwen-2.5-72b-instruct",
}

# ============================================================
# PROMPTING STRATEGIES
# ============================================================

SYSTEM_PROMPT = """You are a precise logical reasoning assistant. Answer questions based strictly on the given premises and logical rules. Be concise and clear.

IMPORTANT: Always start your final answer with one of these exact phrases:
- "Yes" (if the answer is affirmative)
- "No" (if the answer is negative)
- "Cannot be determined" (if the information is insufficient)

After your answer word, you may provide a brief explanation."""

COT_SYSTEM_PROMPT = """You are a precise logical reasoning assistant. When answering questions, think step by step through the logical reasoning before giving your final answer.

IMPORTANT: 
1. First, show your reasoning step by step.
2. Then, state your final answer starting with "FINAL ANSWER:" followed by one of:
   - "Yes" (if the answer is affirmative)
   - "No" (if the answer is negative)  
   - "Cannot be determined" (if the information is insufficient)"""

CGD_REPAIR_PROMPT = """You previously answered a related question as follows:

Previous Question: {prev_question}
Previous Answer: {prev_answer}

However, your current answer may be logically inconsistent with your previous answer.

The logical relationship is: {relationship}

Please reconsider the current question in light of your previous answer and provide a logically consistent response.

Current Question: {current_question}

Provide your revised answer, starting with "Yes", "No", or "Cannot be determined"."""


def extract_answer(response_text: str) -> str:
    """Extract the core answer (Yes/No/Cannot be determined) from model response."""
    text = response_text.strip()
    
    # Check for "FINAL ANSWER:" pattern (CoT)
    if "FINAL ANSWER:" in text.upper():
        text = text.upper().split("FINAL ANSWER:")[-1].strip()
    
    # Normalize
    text_lower = text.lower()
    
    # Check for clear yes/no/cannot determine patterns
    if text_lower.startswith("yes"):
        return "Yes"
    elif text_lower.startswith("no"):
        # Distinguish "No" from "No, we cannot conclude" 
        if any(phrase in text_lower[:100] for phrase in [
            "cannot conclude", "cannot be determined", "not necessarily",
            "we cannot", "insufficient", "cannot determine", "not enough"
        ]):
            return "Cannot be determined"
        return "No"
    elif any(phrase in text_lower[:150] for phrase in [
        "cannot be determined", "cannot determine", "insufficient information",
        "not enough information", "cannot conclude", "indeterminate",
        "we cannot", "it is not possible to determine", "not necessarily"
    ]):
        return "Cannot be determined"
    
    # Fallback: look for yes/no anywhere in first 200 chars
    first_200 = text_lower[:200]
    if "yes" in first_200 and "no" not in first_200:
        return "Yes"
    elif "no" in first_200 and "yes" not in first_200:
        return "No"
    
    return "Unclear"


def call_model(model_id: str, system_prompt: str, user_prompt: str, 
               temperature: float = 0.0, max_tokens: int = 512) -> str:
    """Call a model via OpenRouter/LiteLLM."""
    try:
        response = completion(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=60,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"  [ERROR] Model call failed: {e}")
        time.sleep(2)
        # Retry once
        try:
            response = completion(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=90,
            )
            return response.choices[0].message.content
        except Exception as e2:
            print(f"  [ERROR] Retry failed: {e2}")
            return "[ERROR]"


# ============================================================
# EVALUATION STRATEGIES
# ============================================================

def evaluate_vanilla(model_id: str, question_set: dict) -> dict:
    """Evaluate with direct prompting (no CoT)."""
    results = []
    for q in question_set["questions"]:
        raw_response = call_model(model_id, SYSTEM_PROMPT, q["question"])
        extracted = extract_answer(raw_response)
        results.append({
            "question": q["question"],
            "expected": q["expected_answer"],
            "raw_response": raw_response,
            "extracted_answer": extracted,
            "reasoning_ref": q["reasoning"]
        })
    return {"strategy": "vanilla", "results": results}


def evaluate_cot(model_id: str, question_set: dict) -> dict:
    """Evaluate with Chain-of-Thought prompting."""
    results = []
    for q in question_set["questions"]:
        prompt = f"{q['question']}\n\nThink step by step before answering."
        raw_response = call_model(model_id, COT_SYSTEM_PROMPT, prompt)
        extracted = extract_answer(raw_response)
        results.append({
            "question": q["question"],
            "expected": q["expected_answer"],
            "raw_response": raw_response,
            "extracted_answer": extracted,
            "reasoning_ref": q["reasoning"]
        })
    return {"strategy": "cot", "results": results}


def evaluate_self_consistency(model_id: str, question_set: dict, k: int = 3) -> dict:
    """Evaluate with Self-Consistency (sample k times, majority vote)."""
    results = []
    for q in question_set["questions"]:
        prompt = f"{q['question']}\n\nThink step by step before answering."
        samples = []
        for _ in range(k):
            raw_response = call_model(model_id, COT_SYSTEM_PROMPT, prompt, temperature=0.7)
            extracted = extract_answer(raw_response)
            samples.append({"raw": raw_response, "extracted": extracted})
        
        # Majority vote
        answers = [s["extracted"] for s in samples]
        vote_counts = Counter(answers)
        majority_answer = vote_counts.most_common(1)[0][0]
        
        results.append({
            "question": q["question"],
            "expected": q["expected_answer"],
            "samples": samples,
            "majority_answer": majority_answer,
            "extracted_answer": majority_answer,
            "vote_counts": dict(vote_counts),
            "reasoning_ref": q["reasoning"]
        })
    return {"strategy": "self_consistency", "results": results}


def evaluate_cgd(model_id: str, question_set: dict, nli_checker=None) -> dict:
    """Evaluate with Consistency-Guided Decoding (CGD) - our method.
    
    For each question in the set:
    1. Generate an answer
    2. Check logical consistency with all prior answers using NLI
    3. If contradiction detected, prompt model to revise
    """
    results = []
    prior_qa_pairs = []
    
    for q in question_set["questions"]:
        # Step 1: Generate initial answer
        raw_response = call_model(model_id, SYSTEM_PROMPT, q["question"])
        initial_answer = extract_answer(raw_response)
        
        # Step 2: Check consistency with prior answers
        contradiction_found = False
        contradicting_pair = None
        
        if prior_qa_pairs and nli_checker is not None:
            for prev_q, prev_a, prev_raw in prior_qa_pairs:
                is_contradictory = nli_checker(prev_raw, raw_response)
                if is_contradictory:
                    contradiction_found = True
                    contradicting_pair = (prev_q, prev_a, prev_raw)
                    break
        
        # Step 3: If contradiction, prompt for revision
        final_answer = initial_answer
        revision_response = None
        if contradiction_found and contradicting_pair:
            revision_prompt = CGD_REPAIR_PROMPT.format(
                prev_question=contradicting_pair[0],
                prev_answer=contradicting_pair[1],
                relationship=f"These questions are part of the same logical scenario: {question_set.get('premise', '')}",
                current_question=q["question"]
            )
            revision_response = call_model(model_id, SYSTEM_PROMPT, revision_prompt)
            final_answer = extract_answer(revision_response)
        
        prior_qa_pairs.append((q["question"], final_answer, raw_response))
        
        results.append({
            "question": q["question"],
            "expected": q["expected_answer"],
            "initial_response": raw_response,
            "initial_answer": initial_answer,
            "contradiction_detected": contradiction_found,
            "revision_response": revision_response,
            "extracted_answer": final_answer,
            "reasoning_ref": q["reasoning"]
        })
    
    return {"strategy": "cgd", "results": results}


# ============================================================
# NLI-BASED CONTRADICTION CHECKER
# ============================================================

class NLIContradictionChecker:
    """Check for logical contradictions between two statements using NLI."""
    
    def __init__(self):
        """Initialize with a lightweight NLI approach using the evaluation model itself."""
        self.initialized = True
    
    def check_contradiction_via_llm(self, statement1: str, statement2: str, model_id: str) -> bool:
        """Use a fast LLM to check if two statements contradict each other."""
        prompt = f"""Determine if the following two responses are logically contradictory. 
Two responses are contradictory if they cannot both be true at the same time given the same premises.

Response 1: {statement1[:500]}

Response 2: {statement2[:500]}

Answer with ONLY "CONTRADICTORY" or "CONSISTENT" (one word only)."""
        
        response = call_model(
            "openrouter/openai/gpt-4o-mini",  # Fast, cheap model for NLI
            "You are a logical consistency checker. Respond with only one word.",
            prompt,
            temperature=0.0,
            max_tokens=10
        )
        return "CONTRADICTORY" in response.upper() if response else False
    
    def __call__(self, statement1: str, statement2: str) -> bool:
        return self.check_contradiction_via_llm(statement1, statement2, None)


# ============================================================
# METRICS
# ============================================================

def compute_metrics(evaluation_results: list) -> dict:
    """Compute accuracy and consistency metrics for a set of evaluated question sets."""
    
    all_individual_correct = 0
    all_individual_total = 0
    all_pairwise_consistent = 0
    all_pairwise_total = 0
    all_set_consistent = 0
    all_set_total = 0
    
    category_metrics = {}
    
    for eval_result in evaluation_results:
        category = eval_result["category"]
        if category not in category_metrics:
            category_metrics[category] = {
                "individual_correct": 0, "individual_total": 0,
                "pairwise_consistent": 0, "pairwise_total": 0,
                "set_consistent": 0, "set_total": 0
            }
        
        results = eval_result["evaluation"]["results"]
        
        # Individual accuracy
        set_all_correct = True
        for r in results:
            expected = normalize_expected(r["expected"])
            predicted = normalize_expected(r["extracted_answer"])
            is_correct = expected == predicted
            
            if is_correct:
                all_individual_correct += 1
                category_metrics[category]["individual_correct"] += 1
            else:
                set_all_correct = False
            all_individual_total += 1
            category_metrics[category]["individual_total"] += 1
        
        # Pairwise consistency: for each pair of questions, check if answers
        # are logically consistent (both correct or both following the same logic)
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                exp_i = normalize_expected(results[i]["expected"])
                exp_j = normalize_expected(results[j]["expected"])
                pred_i = normalize_expected(results[i]["extracted_answer"])
                pred_j = normalize_expected(results[j]["extracted_answer"])
                
                # Pairwise consistent if the relationship between predictions
                # matches the relationship between expected answers
                # i.e., if both are correct, or if the "direction" of error is consistent
                is_consistent = (pred_i == exp_i) == (pred_j == exp_j) or \
                               (pred_i == exp_i and pred_j == exp_j)
                
                if is_consistent:
                    all_pairwise_consistent += 1
                    category_metrics[category]["pairwise_consistent"] += 1
                all_pairwise_total += 1
                category_metrics[category]["pairwise_total"] += 1
        
        # Set-level consistency: all answers in set are correct
        if set_all_correct:
            all_set_consistent += 1
            category_metrics[category]["set_consistent"] += 1
        all_set_total += 1
        category_metrics[category]["set_total"] += 1
    
    # Compute rates
    metrics = {
        "individual_accuracy": all_individual_correct / max(all_individual_total, 1),
        "pairwise_consistency_rate": all_pairwise_consistent / max(all_pairwise_total, 1),
        "set_consistency_rate": all_set_consistent / max(all_set_total, 1),
        "individual_correct": all_individual_correct,
        "individual_total": all_individual_total,
        "pairwise_consistent": all_pairwise_consistent,
        "pairwise_total": all_pairwise_total,
        "set_consistent": all_set_consistent,
        "set_total": all_set_total,
    }
    
    # Per-category metrics
    for cat, cm in category_metrics.items():
        metrics[f"{cat}_individual_accuracy"] = cm["individual_correct"] / max(cm["individual_total"], 1)
        metrics[f"{cat}_pairwise_consistency"] = cm["pairwise_consistent"] / max(cm["pairwise_total"], 1)
        metrics[f"{cat}_set_consistency"] = cm["set_consistent"] / max(cm["set_total"], 1)
    
    return metrics


def normalize_expected(answer: str) -> str:
    """Normalize expected/predicted answers for comparison."""
    answer = answer.strip().lower()
    
    if answer.startswith("yes"):
        return "yes"
    elif any(phrase in answer for phrase in [
        "cannot be determined", "cannot conclude", "not necessarily",
        "we cannot", "insufficient", "cannot determine"
    ]):
        return "cannot_determine"
    elif answer.startswith("no"):
        return "no"
    elif answer == "unclear":
        return "unclear"
    return answer


# ============================================================
# MAIN EVALUATION RUNNER
# ============================================================

def run_evaluation(
    benchmark_path: str,
    model_name: str,
    strategy: str = "vanilla",
    max_sets: Optional[int] = None,
    output_dir: str = "results",
    categories: Optional[list] = None
):
    """Run evaluation for a specific model and strategy."""
    
    # Load benchmark
    with open(benchmark_path) as f:
        benchmark = json.load(f)
    
    question_sets = benchmark["sets"]
    
    # Filter categories if specified
    if categories:
        question_sets = [s for s in question_sets if s["category"] in categories]
    
    # Limit number of sets if specified
    if max_sets:
        question_sets = question_sets[:max_sets]
    
    model_id = MODELS.get(model_name, model_name)
    
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"Strategy: {strategy}")
    print(f"Sets to evaluate: {len(question_sets)}")
    print(f"Total questions: {sum(s['num_questions'] for s in question_sets)}")
    print(f"{'='*60}\n")
    
    # Initialize NLI checker for CGD
    nli_checker = None
    if strategy == "cgd":
        nli_checker = NLIContradictionChecker()
    
    # Run evaluation
    all_results = []
    
    for i, qset in enumerate(tqdm(question_sets, desc=f"{model_name}/{strategy}")):
        if strategy == "vanilla":
            eval_result = evaluate_vanilla(model_id, qset)
        elif strategy == "cot":
            eval_result = evaluate_cot(model_id, qset)
        elif strategy == "self_consistency":
            eval_result = evaluate_self_consistency(model_id, qset, k=5)
        elif strategy == "cgd":
            eval_result = evaluate_cgd(model_id, qset, nli_checker)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        all_results.append({
            "set_id": qset["id"],
            "category": qset["category"],
            "premise": qset["premise"],
            "evaluation": eval_result
        })
        
        # Rate limiting
        time.sleep(0.2)
    
    # Compute metrics
    metrics = compute_metrics(all_results)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{model_name}_{strategy}.json")
    
    output_data = {
        "model": model_name,
        "model_id": model_id,
        "strategy": strategy,
        "num_sets": len(question_sets),
        "metrics": metrics,
        "results": all_results
    }
    
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results for {model_name} ({strategy}):")
    print(f"  Individual Accuracy: {metrics['individual_accuracy']:.3f}")
    print(f"  Pairwise Consistency: {metrics['pairwise_consistency_rate']:.3f}")
    print(f"  Set-Level Consistency: {metrics['set_consistency_rate']:.3f}")
    print(f"  Saved to: {output_file}")
    print(f"{'='*60}\n")
    
    return output_data


# ============================================================
# BATCH RUNNER
# ============================================================

def run_full_evaluation(
    benchmark_path: str = "data/consistency_bench.json",
    models: Optional[list] = None,
    strategies: Optional[list] = None,
    max_sets: Optional[int] = None,
    output_dir: str = "results"
):
    """Run full evaluation across all models and strategies."""
    
    if models is None:
        models = list(MODELS.keys())
    if strategies is None:
        strategies = ["vanilla", "cot", "self_consistency", "cgd"]
    
    all_outputs = {}
    
    for model_name in models:
        for strategy in strategies:
            key = f"{model_name}_{strategy}"
            output_file = os.path.join(output_dir, f"{key}.json")
            
            # Skip if already evaluated
            if os.path.exists(output_file):
                print(f"[SKIP] {key} already evaluated, loading from disk...")
                with open(output_file) as f:
                    all_outputs[key] = json.load(f)
                continue
            
            try:
                result = run_evaluation(
                    benchmark_path, model_name, strategy, max_sets, output_dir
                )
                all_outputs[key] = result
            except Exception as e:
                print(f"[ERROR] Failed {key}: {e}")
                continue
    
    # Save summary
    summary = {}
    for key, output in all_outputs.items():
        summary[key] = output.get("metrics", {})
    
    summary_file = os.path.join(output_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to {summary_file}")
    return all_outputs


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run ConsistencyBench evaluation")
    parser.add_argument("--benchmark", default="data/consistency_bench.json")
    parser.add_argument("--model", default=None, help="Specific model to evaluate")
    parser.add_argument("--strategy", default=None, help="Specific strategy")
    parser.add_argument("--max-sets", type=int, default=None, help="Max sets to evaluate")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--all", action="store_true", help="Run all models and strategies")
    
    args = parser.parse_args()
    
    if args.all:
        run_full_evaluation(args.benchmark, max_sets=args.max_sets, output_dir=args.output_dir)
    elif args.model and args.strategy:
        run_evaluation(args.benchmark, args.model, args.strategy, args.max_sets, args.output_dir)
    else:
        print("Usage: python evaluate.py --model gpt-4o --strategy vanilla")
        print("       python evaluate.py --all")
        print("       python evaluate.py --all --max-sets 10  # quick test")
