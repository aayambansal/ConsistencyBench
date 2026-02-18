"""
Analysis and visualization for ConsistencyBench results.

Generates:
1. Main results table (Table 1)
2. Per-category breakdown (Table 2)
3. Consistency heatmap (Figure 1)
4. Category-wise bar chart (Figure 2)
5. Error analysis and qualitative examples
"""

import json
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Publication-quality settings
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

STRATEGY_DISPLAY = {
    "vanilla": "Direct",
    "cot": "CoT",
    "self_consistency": "SC (k=5)",
    "cgd": "CGD (Ours)"
}

MODEL_DISPLAY = {
    "gpt-4o": "GPT-4o",
    "gpt-4o-mini": "GPT-4o Mini",
    "claude-3.5-sonnet": "Claude 3.5 Sonnet",
    "claude-3.5-haiku": "Claude 3.5 Haiku",
    "gemini-2.0-flash": "Gemini 2.0 Flash",
    "deepseek-r1": "DeepSeek-R1",
    "deepseek-v3": "DeepSeek-V3",
    "llama-3.1-70b": "Llama 3.1 70B",
    "qwen-2.5-72b": "Qwen 2.5 72B",
}

CATEGORY_DISPLAY = {
    "contrapositive": "Contra.",
    "transitivity": "Trans.",
    "syllogistic": "Syll.",
    "negation": "Neg.",
    "modus_tollens": "M.Tollens",
    "commonsense_entailment": "Common."
}

CATEGORIES = ["contrapositive", "transitivity", "syllogistic", 
              "negation", "modus_tollens", "commonsense_entailment"]


def load_all_results(results_dir: str = "results") -> dict:
    """Load all evaluation results."""
    results = {}
    for f in glob.glob(os.path.join(results_dir, "*.json")):
        if "summary" in f or "analysis" in f:
            continue
        with open(f) as fh:
            data = json.load(fh)
            key = f"{data['model']}_{data['strategy']}"
            results[key] = data
    return results


def build_main_table(results: dict) -> pd.DataFrame:
    """Build the main results table (Table 1)."""
    rows = []
    
    for key, data in sorted(results.items()):
        model = data["model"]
        strategy = data["strategy"]
        metrics = data["metrics"]
        
        rows.append({
            "Model": MODEL_DISPLAY.get(model, model),
            "Strategy": STRATEGY_DISPLAY.get(strategy, strategy),
            "IA (%)": metrics["individual_accuracy"] * 100,
            "PCR (%)": metrics["pairwise_consistency_rate"] * 100,
            "SCR (%)": metrics["set_consistency_rate"] * 100,
            "model_key": model,
            "strategy_key": strategy,
        })
    
    df = pd.DataFrame(rows)
    return df


def build_category_table(results: dict) -> pd.DataFrame:
    """Build per-category breakdown (Table 2)."""
    rows = []
    
    for key, data in sorted(results.items()):
        model = data["model"]
        strategy = data["strategy"]
        metrics = data["metrics"]
        
        for cat in CATEGORIES:
            ia_key = f"{cat}_individual_accuracy"
            pcr_key = f"{cat}_pairwise_consistency"
            scr_key = f"{cat}_set_consistency"
            
            if ia_key in metrics:
                rows.append({
                    "Model": MODEL_DISPLAY.get(model, model),
                    "Strategy": STRATEGY_DISPLAY.get(strategy, strategy),
                    "Category": CATEGORY_DISPLAY.get(cat, cat),
                    "IA (%)": metrics[ia_key] * 100,
                    "PCR (%)": metrics[pcr_key] * 100,
                    "SCR (%)": metrics[scr_key] * 100,
                    "category_key": cat,
                    "model_key": model,
                    "strategy_key": strategy,
                })
    
    return pd.DataFrame(rows)


def plot_consistency_heatmap(results: dict, output_path: str = "figures/consistency_heatmap.pdf"):
    """Generate consistency heatmap: models (rows) x categories (cols) for set-level consistency."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Filter for vanilla strategy only for the heatmap
    models = sorted(set(d["model"] for d in results.values()))
    
    strategies_to_show = ["vanilla", "cgd"]
    
    fig, axes = plt.subplots(1, len(strategies_to_show), figsize=(14, 5), sharey=True)
    
    if len(strategies_to_show) == 1:
        axes = [axes]
    
    for ax, strategy in zip(axes, strategies_to_show):
        matrix = []
        model_labels = []
        
        for model in models:
            key = f"{model}_{strategy}"
            if key not in results:
                continue
            
            metrics = results[key]["metrics"]
            row = []
            for cat in CATEGORIES:
                scr_key = f"{cat}_set_consistency"
                row.append(metrics.get(scr_key, 0) * 100)
            
            matrix.append(row)
            model_labels.append(MODEL_DISPLAY.get(model, model))
        
        if not matrix:
            continue
            
        matrix = np.array(matrix)
        
        im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)
        
        # Labels
        ax.set_xticks(range(len(CATEGORIES)))
        ax.set_xticklabels([CATEGORY_DISPLAY[c] for c in CATEGORIES], rotation=45, ha="right")
        ax.set_yticks(range(len(model_labels)))
        ax.set_yticklabels(model_labels)
        ax.set_title(STRATEGY_DISPLAY.get(strategy, strategy))
        
        # Add value annotations
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                color = "white" if val < 40 or val > 80 else "black"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=8, color=color)
    
    fig.colorbar(im, ax=axes, label="Set-Level Consistency Rate (%)", shrink=0.8)
    fig.suptitle("Set-Level Consistency Rate (SCR%) by Model and Category", fontsize=14, y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), bbox_inches='tight')
    print(f"Saved heatmap to {output_path}")
    plt.close()


def plot_strategy_comparison(results: dict, output_path: str = "figures/strategy_comparison.pdf"):
    """Bar chart comparing strategies across models."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    models = sorted(set(d["model"] for d in results.values()))
    strategies = ["vanilla", "cot", "self_consistency", "cgd"]
    
    # Filter to strategies that exist in results
    available_strategies = []
    for s in strategies:
        if any(f"{m}_{s}" in results for m in models):
            available_strategies.append(s)
    
    x = np.arange(len(models))
    width = 0.8 / len(available_strategies)
    
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    metric_keys = [
        ("individual_accuracy", "Individual Accuracy (IA%)"),
        ("pairwise_consistency_rate", "Pairwise Consistency (PCR%)"),
        ("set_consistency_rate", "Set-Level Consistency (SCR%)")
    ]
    
    for ax, (metric_key, metric_label) in zip(axes, metric_keys):
        for i, strategy in enumerate(available_strategies):
            values = []
            for model in models:
                key = f"{model}_{strategy}"
                if key in results:
                    values.append(results[key]["metrics"][metric_key] * 100)
                else:
                    values.append(0)
            
            offset = (i - len(available_strategies)/2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, 
                         label=STRATEGY_DISPLAY.get(strategy, strategy),
                         color=colors[i % len(colors)], alpha=0.85)
        
        ax.set_ylabel(metric_label)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_DISPLAY.get(m, m) for m in models], rotation=45, ha="right")
        ax.legend(loc="lower right", fontsize=8)
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle("Strategy Comparison Across Models on ConsistencyBench", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), bbox_inches='tight')
    print(f"Saved strategy comparison to {output_path}")
    plt.close()


def plot_category_radar(results: dict, output_path: str = "figures/category_radar.pdf"):
    """Radar/spider chart showing per-category set consistency for select models."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Pick representative models for vanilla vs CGD
    select_models = []
    for m in ["gpt-4o", "claude-3.5-sonnet", "deepseek-r1", "llama-3.1-70b", "qwen-2.5-72b"]:
        if any(f"{m}_vanilla" in results for _ in [1]):
            if f"{m}_vanilla" in results:
                select_models.append(m)
    
    if not select_models:
        select_models = list(set(d["model"] for d in results.values()))[:4]
    
    categories = CATEGORIES
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw=dict(polar=True))
    
    colors_cycle = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
    
    for ax, strategy in zip(axes, ["vanilla", "cgd"]):
        for i, model in enumerate(select_models):
            key = f"{model}_{strategy}"
            if key not in results:
                continue
            
            metrics = results[key]["metrics"]
            values = [metrics.get(f"{cat}_set_consistency", 0) * 100 for cat in categories]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=1.5, 
                   label=MODEL_DISPLAY.get(model, model),
                   color=colors_cycle[i % len(colors_cycle)])
            ax.fill(angles, values, alpha=0.1, color=colors_cycle[i % len(colors_cycle)])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([CATEGORY_DISPLAY[c] for c in categories])
        ax.set_ylim(0, 100)
        ax.set_title(STRATEGY_DISPLAY.get(strategy, strategy), pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
    
    plt.suptitle("Per-Category Set Consistency: Direct vs. CGD", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), bbox_inches='tight')
    print(f"Saved radar chart to {output_path}")
    plt.close()


def plot_cgd_improvement(results: dict, output_path: str = "figures/cgd_improvement.pdf"):
    """Bar chart showing CGD improvement over vanilla for each model."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    models = sorted(set(d["model"] for d in results.values()))
    
    improvements_scr = []
    improvements_pcr = []
    model_labels = []
    
    for model in models:
        vanilla_key = f"{model}_vanilla"
        cgd_key = f"{model}_cgd"
        
        if vanilla_key in results and cgd_key in results:
            vanilla_scr = results[vanilla_key]["metrics"]["set_consistency_rate"] * 100
            cgd_scr = results[cgd_key]["metrics"]["set_consistency_rate"] * 100
            vanilla_pcr = results[vanilla_key]["metrics"]["pairwise_consistency_rate"] * 100
            cgd_pcr = results[cgd_key]["metrics"]["pairwise_consistency_rate"] * 100
            
            improvements_scr.append(cgd_scr - vanilla_scr)
            improvements_pcr.append(cgd_pcr - vanilla_pcr)
            model_labels.append(MODEL_DISPLAY.get(model, model))
    
    if not improvements_scr:
        print("No CGD vs vanilla comparison available, skipping improvement plot")
        return
    
    x = np.arange(len(model_labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    bars1 = ax.bar(x - width/2, improvements_pcr, width, label='PCR Improvement', 
                   color='#3498DB', alpha=0.85)
    bars2 = ax.bar(x + width/2, improvements_scr, width, label='SCR Improvement',
                   color='#2ECC71', alpha=0.85)
    
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_ylabel('Improvement (percentage points)')
    ax.set_title('CGD Improvement Over Direct Prompting')
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            va = 'bottom' if height >= 0 else 'top'
            ax.annotate(f'{height:+.1f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3 if height >= 0 else -3),
                       textcoords="offset points",
                       ha='center', va=va, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), bbox_inches='tight')
    print(f"Saved CGD improvement plot to {output_path}")
    plt.close()


def generate_latex_table(df: pd.DataFrame, output_path: str):
    """Generate LaTeX table from DataFrame."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Format numbers
    df_display = df.copy()
    for col in ["IA (%)", "PCR (%)", "SCR (%)"]:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(lambda x: f"{x:.1f}")
    
    # Drop internal keys
    cols_to_drop = [c for c in df_display.columns if c.endswith("_key")]
    df_display = df_display.drop(columns=cols_to_drop, errors='ignore')
    
    latex = df_display.to_latex(index=False, escape=False)
    
    with open(output_path, "w") as f:
        f.write(latex)
    
    print(f"Saved LaTeX table to {output_path}")


def run_error_analysis(results: dict, output_path: str = "results/error_analysis.json"):
    """Analyze common error patterns."""
    
    errors_by_category = {}
    
    for key, data in results.items():
        if data["strategy"] != "vanilla":
            continue
        
        for eval_result in data.get("results", []):
            cat = eval_result["category"]
            if cat not in errors_by_category:
                errors_by_category[cat] = {"total": 0, "errors": 0, "examples": []}
            
            for r in eval_result["evaluation"]["results"]:
                from evaluate import normalize_expected
                expected = normalize_expected(r["expected"])
                predicted = normalize_expected(r["extracted_answer"])
                
                errors_by_category[cat]["total"] += 1
                if expected != predicted:
                    errors_by_category[cat]["errors"] += 1
                    if len(errors_by_category[cat]["examples"]) < 5:
                        errors_by_category[cat]["examples"].append({
                            "model": data["model"],
                            "question": r["question"][:200],
                            "expected": r["expected"],
                            "predicted": r["extracted_answer"],
                            "response_excerpt": r.get("raw_response", "")[:300]
                        })
    
    # Summary
    analysis = {}
    for cat, info in errors_by_category.items():
        analysis[cat] = {
            "total_questions": info["total"],
            "errors": info["errors"],
            "error_rate": info["errors"] / max(info["total"], 1),
            "example_errors": info["examples"]
        }
    
    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2)
    
    print(f"Saved error analysis to {output_path}")
    return analysis


def run_full_analysis(results_dir: str = "results", figures_dir: str = "figures"):
    """Run the complete analysis pipeline."""
    
    print("Loading results...")
    results = load_all_results(results_dir)
    
    if not results:
        print("No results found! Run evaluation first.")
        return
    
    print(f"Loaded {len(results)} result files")
    
    # Build tables
    print("\nBuilding main results table...")
    main_df = build_main_table(results)
    print(main_df.to_string(index=False))
    generate_latex_table(main_df, os.path.join(figures_dir, "table1_main.tex"))
    main_df.to_csv(os.path.join(results_dir, "main_results.csv"), index=False)
    
    print("\nBuilding category table...")
    cat_df = build_category_table(results)
    generate_latex_table(cat_df, os.path.join(figures_dir, "table2_categories.tex"))
    cat_df.to_csv(os.path.join(results_dir, "category_results.csv"), index=False)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_consistency_heatmap(results, os.path.join(figures_dir, "consistency_heatmap.pdf"))
    plot_strategy_comparison(results, os.path.join(figures_dir, "strategy_comparison.pdf"))
    plot_category_radar(results, os.path.join(figures_dir, "category_radar.pdf"))
    plot_cgd_improvement(results, os.path.join(figures_dir, "cgd_improvement.pdf"))
    
    # Error analysis
    print("\nRunning error analysis...")
    try:
        error_analysis = run_error_analysis(results)
        print("\nError rates by category:")
        for cat, info in error_analysis.items():
            print(f"  {cat}: {info['error_rate']:.1%} ({info['errors']}/{info['total_questions']})")
    except Exception as e:
        print(f"Error analysis failed: {e}")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    run_full_analysis()
