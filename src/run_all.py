"""
Orchestrator: run all evaluations for the paper.

Strategy:
- Run 6 models x 4 strategies on a stratified sample of the benchmark
- 50 sets per category = 300 sets total (~1,160 questions)
- SC uses k=3 to save budget
- Saves results incrementally
"""

import os
import sys
import json
import time
import random
import subprocess

random.seed(42)

# Models to evaluate (ordered by priority for the paper)
MODELS = [
    "gpt-4o",
    "claude-3.5-sonnet",
    "deepseek-r1",
    "llama-3.1-70b",
    "qwen-2.5-72b",
    "gemini-2.0-flash",
]

# Strategies in priority order
STRATEGIES = ["vanilla", "cgd", "cot", "self_consistency"]

SETS_PER_CATEGORY = 50  # 50 x 6 categories = 300 sets

def create_stratified_sample(benchmark_path="data/consistency_bench.json",
                              output_path="data/consistency_bench_sample.json"):
    """Create a stratified sample of the benchmark."""
    with open(benchmark_path) as f:
        benchmark = json.load(f)
    
    categories = {}
    for s in benchmark["sets"]:
        cat = s["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(s)
    
    sampled_sets = []
    for cat, sets in categories.items():
        n = min(SETS_PER_CATEGORY, len(sets))
        sampled = random.sample(sets, n)
        sampled_sets.extend(sampled)
        print(f"  {cat}: sampled {n}/{len(sets)} sets")
    
    random.shuffle(sampled_sets)
    
    total_qs = sum(s["num_questions"] for s in sampled_sets)
    
    sample_benchmark = {
        "name": "ConsistencyBench (Stratified Sample)",
        "version": "1.0",
        "description": benchmark["description"],
        "num_sets": len(sampled_sets),
        "num_questions": total_qs,
        "categories": benchmark["categories"],
        "sets": sampled_sets
    }
    
    with open(output_path, "w") as f:
        json.dump(sample_benchmark, f, indent=2)
    
    print(f"\nStratified sample: {len(sampled_sets)} sets, {total_qs} questions")
    print(f"Saved to {output_path}")
    return output_path

def check_done(model, strategy, results_dir="results"):
    """Check if this evaluation has already been run."""
    path = os.path.join(results_dir, f"{model}_{strategy}.json")
    return os.path.exists(path)

def run_single(model, strategy, benchmark_path):
    """Run a single model/strategy evaluation."""
    cmd = [
        sys.executable, "src/evaluate.py",
        "--model", model,
        "--strategy", strategy,
        "--benchmark", benchmark_path,
        "--output-dir", "results"
    ]
    
    print(f"\n{'#'*70}")
    print(f"# RUNNING: {model} / {strategy}")
    print(f"{'#'*70}")
    
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    
    result = subprocess.run(
        cmd, 
        env=env,
        timeout=7200  # 2 hour timeout per run
    )
    
    return result.returncode == 0

def main():
    os.makedirs("results", exist_ok=True)
    
    # Create stratified sample
    print("Creating stratified sample...")
    sample_path = create_stratified_sample()
    
    total = len(MODELS) * len(STRATEGIES)
    done_count = 0
    failed = []
    
    for model in MODELS:
        for strategy in STRATEGIES:
            done_count += 1
            
            if check_done(model, strategy):
                print(f"[{done_count}/{total}] SKIP {model}/{strategy} (already done)")
                continue
            
            print(f"\n[{done_count}/{total}] Starting {model}/{strategy}...")
            start = time.time()
            
            try:
                success = run_single(model, strategy, sample_path)
                elapsed = time.time() - start
                
                if success:
                    print(f"[{done_count}/{total}] DONE {model}/{strategy} in {elapsed:.0f}s")
                else:
                    print(f"[{done_count}/{total}] FAILED {model}/{strategy}")
                    failed.append(f"{model}/{strategy}")
            except subprocess.TimeoutExpired:
                print(f"[{done_count}/{total}] TIMEOUT {model}/{strategy}")
                failed.append(f"{model}/{strategy} (timeout)")
            except Exception as e:
                print(f"[{done_count}/{total}] ERROR {model}/{strategy}: {e}")
                failed.append(f"{model}/{strategy} ({e})")
    
    print(f"\n{'='*70}")
    print(f"EVALUATION COMPLETE")
    print(f"  Total: {total}")
    print(f"  Failed: {len(failed)}")
    if failed:
        for f in failed:
            print(f"    - {f}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
