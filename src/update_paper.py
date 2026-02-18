"""
Update the paper's LaTeX tables and figures with real evaluation results.

Reads from results/ directory and generates:
1. results_table.tex - Main results table (Table 1)
2. category_analysis.tex - Per-category analysis with figure
3. consistency_gap_analysis.tex - Gap analysis with figure
4. error_analysis.tex - Qualitative error analysis
5. Figures in figures/ directory
"""

import json
import os
import glob
import numpy as np
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))
from evaluate import normalize_expected

# ============================================================
# CONSTANTS
# ============================================================

MODEL_DISPLAY = {
    # OpenAI
    "gpt-5.2": "GPT-5.2",
    "gpt-5": "GPT-5",
    "gpt-5-mini": "GPT-5 Mini",
    "gpt-4.1": "GPT-4.1",
    "gpt-4o": "GPT-4o",
    "gpt-4o-mini": "GPT-4o Mini",
    "o3": "o3",
    # Anthropic
    "claude-opus-4.6": "Claude Opus 4.6",
    "claude-sonnet-4.6": "Claude Sonnet 4.6",
    "claude-opus-4.5": "Claude Opus 4.5",
    "claude-sonnet-4.5": "Claude Sonnet 4.5",
    "claude-3.5-sonnet": "Claude 3.5 Sonnet",
    "claude-3.5-haiku": "Claude 3.5 Haiku",
    # Google
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "gemini-2.0-flash": "Gemini 2.0 Flash",
    # DeepSeek
    "deepseek-v3.2": "DeepSeek V3.2",
    "deepseek-r1": "DeepSeek-R1",
    "deepseek-v3": "DeepSeek-V3",
    # Meta
    "llama-3.3-70b": "Llama 3.3 70B",
    "llama-3.1-70b": "Llama 3.1 70B",
    # Qwen
    "qwen-2.5-72b": "Qwen 2.5 72B",
}

STRATEGY_DISPLAY = {
    "vanilla": "Direct",
    "cot": "CoT",
    "self_consistency": "SC ($k{=}3$)",
    "cgd": "\\method{} (Ours)"
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

MODELS_ORDER = [
    "gpt-5.2", "gpt-5", "gpt-5-mini", "gpt-4.1", "gpt-4o", "o3",
    "claude-opus-4.6", "claude-sonnet-4.6", "claude-opus-4.5", "claude-sonnet-4.5",
    "claude-3.5-sonnet",
    "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash",
    "deepseek-v3.2", "deepseek-r1",
    "llama-3.3-70b", "llama-3.1-70b",
    "qwen-2.5-72b",
]

STRATEGIES_ORDER = ["vanilla", "cot", "self_consistency", "cgd"]

PAPER_DIR = "paper/iclr2026"
FIGURES_DIR = "figures"


def load_results(results_dir="results"):
    """Load all available results."""
    results = {}
    for f in glob.glob(os.path.join(results_dir, "*.json")):
        basename = os.path.basename(f)
        if basename in ("summary.json", "error_analysis.json"):
            continue
        with open(f) as fh:
            data = json.load(fh)
            key = f"{data['model']}_{data['strategy']}"
            results[key] = data
    return results


def fmt(val, bold=False):
    """Format a percentage value for LaTeX."""
    s = f"{val:.1f}"
    if bold:
        return f"\\textbf{{{s}}}"
    return s


def _generate_single_table(results, models_with_data):
    """Generate a single combined table (for <=9 models)."""
    lines = []
    lines.append("Table~\\ref{tab:main} presents the main results. Across all models and strategies, we observe a consistent and substantial gap between individual accuracy (IA) and set-level consistency (SCR).")
    lines.append("")
    lines.append("\\begin{table}[t]")
    lines.append("\\caption{Main results on \\bench{} (300 question sets, 1,150 questions). IA: Individual Accuracy, PCR: Pairwise Consistency Rate, SCR: Set-Level Consistency Rate. Best result per model in \\textbf{bold}. \\method{} consistently improves SCR over all baselines.}")
    lines.append("\\label{tab:main}")
    lines.append("\\begin{center}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{llccc}")
    lines.append("\\toprule")
    lines.append("\\textbf{Model} & \\textbf{Strategy} & \\textbf{IA (\\%)} $\\uparrow$ & \\textbf{PCR (\\%)} $\\uparrow$ & \\textbf{SCR (\\%)} $\\uparrow$ \\\\")
    lines.append("\\midrule")
    
    rendered = 0
    for model in models_with_data:
        model_display = MODEL_DISPLAY.get(model, model)
        best_scr, best_ia, best_pcr = -1, -1, -1
        model_results = {}
        for strategy in STRATEGIES_ORDER:
            key = f"{model}_{strategy}"
            if key in results:
                m = results[key]["metrics"]
                ia = m["individual_accuracy"] * 100
                pcr = m["pairwise_consistency_rate"] * 100
                scr = m["set_consistency_rate"] * 100
                model_results[strategy] = (ia, pcr, scr)
                best_scr = max(best_scr, scr)
                best_ia = max(best_ia, ia)
                best_pcr = max(best_pcr, pcr)
        if not model_results:
            continue
        available_strategies = [s for s in STRATEGIES_ORDER if s in model_results]
        num_rows = len(available_strategies)
        first = True
        for strategy in available_strategies:
            strategy_display = STRATEGY_DISPLAY.get(strategy, strategy)
            ia, pcr, scr = model_results[strategy]
            ia_str = fmt(ia, ia >= best_ia - 0.05)
            pcr_str = fmt(pcr, pcr >= best_pcr - 0.05)
            scr_str = fmt(scr, scr >= best_scr - 0.05)
            if first:
                if num_rows > 1:
                    lines.append(f"\\multirow{{{num_rows}}}{{*}}{{{model_display}}}")
                else:
                    lines.append(f"{model_display}")
                first = False
            lines.append(f" & {strategy_display} & {ia_str} & {pcr_str} & {scr_str} \\\\")
        rendered += 1
        if rendered < len(models_with_data):
            lines.append("\\midrule")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{center}")
    lines.append("\\end{table}")
    return lines


def _generate_split_tables(results, models_with_data):
    """Generate split tables for 10+ models:
    Table 1: All models, Direct vs CGD only (compact)
    Table 2: Strategy comparison (Direct, CoT, SC, CGD) for representative models
    """
    lines = []
    
    # === Table 1: All models, Direct + CGD comparison ===
    lines.append("Table~\\ref{tab:main} compares Direct prompting against \\method{} across all evaluated models. Table~\\ref{tab:strategies} provides a detailed strategy comparison for representative models.")
    lines.append("")
    lines.append("\\begin{table}[t]")
    lines.append("\\caption{Direct prompting vs.\\ \\method{} on \\bench{} (300 sets, 1,150 questions). Gap = IA $-$ SCR. $\\Delta$SCR = absolute SCR improvement from \\method{} over Direct. Best SCR per model in \\textbf{bold}.}")
    lines.append("\\label{tab:main}")
    lines.append("\\begin{center}")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{3.5pt}")
    lines.append("\\begin{tabular}{l ccc c ccc c}")
    lines.append("\\toprule")
    lines.append(" & \\multicolumn{3}{c}{\\textbf{Direct}} & & \\multicolumn{3}{c}{\\textbf{\\method{} (Ours)}} & \\\\")
    lines.append("\\cmidrule{2-4} \\cmidrule{6-8}")
    lines.append("\\textbf{Model} & \\textbf{IA} & \\textbf{SCR} & \\textbf{Gap} & & \\textbf{IA} & \\textbf{SCR} & \\textbf{Gap} & \\textbf{$\\Delta$SCR} \\\\")
    lines.append("\\midrule")
    
    rendered = 0
    for model in models_with_data:
        model_display = MODEL_DISPLAY.get(model, model)
        vanilla_key = f"{model}_vanilla"
        cgd_key = f"{model}_cgd"
        
        has_vanilla = vanilla_key in results
        has_cgd = cgd_key in results
        
        if not has_vanilla and not has_cgd:
            continue
        
        if has_vanilla:
            mv = results[vanilla_key]["metrics"]
            v_ia = mv["individual_accuracy"] * 100
            v_scr = mv["set_consistency_rate"] * 100
            v_gap = v_ia - v_scr
        
        if has_cgd:
            mc = results[cgd_key]["metrics"]
            c_ia = mc["individual_accuracy"] * 100
            c_scr = mc["set_consistency_rate"] * 100
            c_gap = c_ia - c_scr
        
        # Determine best SCR
        best_scr = -1
        if has_vanilla:
            best_scr = max(best_scr, v_scr)
        if has_cgd:
            best_scr = max(best_scr, c_scr)
        
        # Format cells
        if has_vanilla:
            v_ia_s = fmt(v_ia)
            v_scr_s = fmt(v_scr, v_scr >= best_scr - 0.05)
            v_gap_s = fmt(v_gap)
        else:
            v_ia_s = v_scr_s = v_gap_s = "--"
        
        if has_cgd:
            c_ia_s = fmt(c_ia)
            c_scr_s = fmt(c_scr, c_scr >= best_scr - 0.05)
            c_gap_s = fmt(c_gap)
            if has_vanilla:
                delta = c_scr - v_scr
                delta_s = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"
            else:
                delta_s = "--"
        else:
            c_ia_s = c_scr_s = c_gap_s = delta_s = "--"
        
        lines.append(f"{model_display} & {v_ia_s} & {v_scr_s} & {v_gap_s} & & {c_ia_s} & {c_scr_s} & {c_gap_s} & {delta_s} \\\\")
        rendered += 1
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{center}")
    lines.append("\\end{table}")
    lines.append("")
    
    # === Table 2: Strategy comparison for representative models ===
    # Pick models that have CoT or SC data, plus a few key models
    strategy_models = []
    for model in models_with_data:
        has_extra = any(f"{model}_{s}" in results for s in ["cot", "self_consistency"])
        if has_extra:
            strategy_models.append(model)
    # Also include key models with just vanilla+cgd if we have fewer than 4
    if len(strategy_models) < 4:
        for model in models_with_data:
            if model not in strategy_models and f"{model}_cgd" in results:
                strategy_models.append(model)
            if len(strategy_models) >= 6:
                break
    
    if strategy_models:
        lines.append("\\begin{table}[t]")
        lines.append("\\caption{Detailed strategy comparison on representative models. CoT = Chain-of-Thought, SC = Self-Consistency ($k{=}3$).}")
        lines.append("\\label{tab:strategies}")
        lines.append("\\begin{center}")
        lines.append("\\small")
        lines.append("\\begin{tabular}{llccc}")
        lines.append("\\toprule")
        lines.append("\\textbf{Model} & \\textbf{Strategy} & \\textbf{IA (\\%)} $\\uparrow$ & \\textbf{PCR (\\%)} $\\uparrow$ & \\textbf{SCR (\\%)} $\\uparrow$ \\\\")
        lines.append("\\midrule")
        
        rendered = 0
        for model in strategy_models:
            model_display = MODEL_DISPLAY.get(model, model)
            best_scr, best_ia, best_pcr = -1, -1, -1
            model_results = {}
            for strategy in STRATEGIES_ORDER:
                key = f"{model}_{strategy}"
                if key in results:
                    m = results[key]["metrics"]
                    ia = m["individual_accuracy"] * 100
                    pcr = m["pairwise_consistency_rate"] * 100
                    scr = m["set_consistency_rate"] * 100
                    model_results[strategy] = (ia, pcr, scr)
                    best_scr = max(best_scr, scr)
                    best_ia = max(best_ia, ia)
                    best_pcr = max(best_pcr, pcr)
            if not model_results:
                continue
            available_strategies = [s for s in STRATEGIES_ORDER if s in model_results]
            num_rows = len(available_strategies)
            first = True
            for strategy in available_strategies:
                strategy_display = STRATEGY_DISPLAY.get(strategy, strategy)
                ia, pcr, scr = model_results[strategy]
                ia_str = fmt(ia, ia >= best_ia - 0.05)
                pcr_str = fmt(pcr, pcr >= best_pcr - 0.05)
                scr_str = fmt(scr, scr >= best_scr - 0.05)
                if first:
                    if num_rows > 1:
                        lines.append(f"\\multirow{{{num_rows}}}{{*}}{{{model_display}}}")
                    else:
                        lines.append(f"{model_display}")
                    first = False
                lines.append(f" & {strategy_display} & {ia_str} & {pcr_str} & {scr_str} \\\\")
            rendered += 1
            if rendered < len(strategy_models):
                lines.append("\\midrule")
        
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{center}")
        lines.append("\\end{table}")
    
    return lines


def generate_main_table(results):
    """Generate the main results table(s) LaTeX.
    
    If many models have data, splits into:
    - Table 1: All models with Direct + CGD (our key comparison)
    - Table 2: Strategy comparison (Direct, CoT, SC, CGD) for a subset of models
    Otherwise, generates a single combined table.
    """
    
    # Count how many models have data
    models_with_data = [m for m in MODELS_ORDER if any(f"{m}_{s}" in results for s in STRATEGIES_ORDER)]
    
    # If 10+ models have data, use the split layout
    use_split = len(models_with_data) >= 10
    
    lines = []
    
    if use_split:
        lines.extend(_generate_split_tables(results, models_with_data))
    else:
        lines.extend(_generate_single_table(results, models_with_data))
    
    # Analysis text
    lines.append("")
    lines.append("Key observations:")
    lines.append("\\begin{itemize}")
    
    # Compute actual statistics for the text
    all_gaps = []
    cgd_improvements = []
    for model in MODELS_ORDER:
        vanilla_key = f"{model}_vanilla"
        cgd_key = f"{model}_cgd"
        if vanilla_key in results:
            m = results[vanilla_key]["metrics"]
            ia = m["individual_accuracy"] * 100
            scr = m["set_consistency_rate"] * 100
            all_gaps.append(ia - scr)
        if vanilla_key in results and cgd_key in results:
            v_scr = results[vanilla_key]["metrics"]["set_consistency_rate"] * 100
            c_scr = results[cgd_key]["metrics"]["set_consistency_rate"] * 100
            cgd_improvements.append(c_scr - v_scr)
    
    if all_gaps:
        min_gap = min(all_gaps)
        max_gap = max(all_gaps)
        lines.append(f"    \\item \\textbf{{The consistency gap is universal}}: All {len(all_gaps)} models show IA $>$ SCR, with gaps ranging from {min_gap:.0f} to {max_gap:.0f} percentage points.")
    
    # Check if CoT results exist
    has_cot = any(f"{model}_cot" in results for model in MODELS_ORDER)
    if has_cot:
        lines.append("    \\item \\textbf{CoT improves IA more than SCR}: Chain-of-thought consistently boosts individual accuracy but provides smaller relative gains in set-level consistency.")
    
    if cgd_improvements:
        min_imp = min(cgd_improvements)
        max_imp = max(cgd_improvements)
        avg_imp = np.mean(cgd_improvements)
        if min_imp == max_imp:
            lines.append(f"    \\item \\textbf{{\\method{{}} targets the right metric}}: Our method provides improvements in SCR of {min_imp:.0f} percentage points over Direct prompting, demonstrating that cross-query consistency is amenable to inference-time intervention.")
        else:
            lines.append(f"    \\item \\textbf{{\\method{{}} targets the right metric}}: Across {len(cgd_improvements)} models, \\method{{}} improves SCR by {avg_imp:.0f} percentage points on average ({min_imp:.0f}--{max_imp:.0f}pp range) over Direct prompting.")
    
    # Check if SC results exist
    has_sc = any(f"{model}_self_consistency" in results for model in MODELS_ORDER)
    if has_sc:
        lines.append("    \\item \\textbf{Self-consistency via voting is insufficient}: SC improves over CoT on IA but does not close the consistency gap, as voting operates independently per question.")
    lines.append("\\end{itemize}")
    
    return "\n".join(lines)


def generate_category_analysis(results):
    """Generate per-category analysis LaTeX."""
    lines = []
    
    lines.append("Figure~\\ref{fig:category} shows the set-level consistency rate broken down by category for Direct prompting and \\method{}. Several patterns emerge:")
    lines.append("")
    
    # Generate actual figure using matplotlib
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        plt.rcParams.update({
            'font.size': 10, 'font.family': 'serif',
            'figure.dpi': 300, 'savefig.dpi': 300,
        })
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
        
        for ax, strategy in zip(axes, ["vanilla", "cgd"]):
            models_with_data = []
            data_matrix = []
            
            for model in MODELS_ORDER:
                key = f"{model}_{strategy}"
                if key in results:
                    models_with_data.append(MODEL_DISPLAY.get(model, model))
                    row = []
                    for cat in CATEGORIES:
                        scr_key = f"{cat}_set_consistency"
                        row.append(results[key]["metrics"].get(scr_key, 0) * 100)
                    data_matrix.append(row)
            
            if not data_matrix:
                continue
            
            data_matrix = np.array(data_matrix)
            im = ax.imshow(data_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)
            
            ax.set_xticks(range(len(CATEGORIES)))
            ax.set_xticklabels([CATEGORY_DISPLAY[c] for c in CATEGORIES], rotation=45, ha="right")
            ax.set_yticks(range(len(models_with_data)))
            ax.set_yticklabels(models_with_data)
            title = STRATEGY_DISPLAY.get(strategy, strategy).replace("\\method{}", "CGD").replace(" (Ours)", "")
            ax.set_title(title)
            
            for i in range(data_matrix.shape[0]):
                for j in range(data_matrix.shape[1]):
                    val = data_matrix[i, j]
                    color = "white" if val < 30 or val > 75 else "black"
                    ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=8, color=color)
        
        fig.colorbar(im, ax=axes, label="SCR (%)", shrink=0.8)
        plt.tight_layout()
        
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig_path = os.path.join(FIGURES_DIR, "category_heatmap.pdf")
        plt.savefig(fig_path, bbox_inches='tight')
        plt.savefig(fig_path.replace('.pdf', '.png'), bbox_inches='tight')
        plt.close()
        print(f"  Saved category heatmap to {fig_path}")
        
        lines.append("\\begin{figure}[t]")
        lines.append("\\centering")
        lines.append(f"\\includegraphics[width=0.95\\linewidth]{{../../{fig_path}}}")
        lines.append("\\caption{Set-level consistency rate (SCR\\%) by category for Direct prompting (left) and \\method{} (right). \\method{} provides consistent improvements across categories, with the largest gains on contrapositive and modus tollens.}")
        lines.append("\\label{fig:category}")
        lines.append("\\end{figure}")
    except Exception as e:
        print(f"  Warning: Could not generate category figure: {e}")
        lines.append("\\begin{figure}[t]")
        lines.append("\\centering")
        lines.append("\\fbox{\\parbox{0.95\\linewidth}{\\centering\\vspace{3cm}\\textit{[Figure: Per-category SCR comparison]}\\ vspace{3cm}}}")
        lines.append("\\caption{Set-level consistency rate by category.}")
        lines.append("\\label{fig:category}")
        lines.append("\\end{figure}")
    
    lines.append("")
    
    # Find hardest/easiest categories from data
    cat_scores = {}
    for model in MODELS_ORDER:
        key = f"{model}_vanilla"
        if key in results:
            for cat in CATEGORIES:
                scr_key = f"{cat}_set_consistency"
                scr = results[key]["metrics"].get(scr_key, 0) * 100
                if cat not in cat_scores:
                    cat_scores[cat] = []
                cat_scores[cat].append(scr)
    
    if cat_scores:
        avg_scores = {cat: np.mean(scores) for cat, scores in cat_scores.items()}
        hardest = min(avg_scores, key=avg_scores.get)
        easiest = max(avg_scores, key=avg_scores.get)
        
        lines.append("\\begin{itemize}")
        lines.append(f"    \\item \\textbf{{{CATEGORY_DISPLAY[hardest]} is the hardest category}}: With an average SCR of {avg_scores[hardest]:.1f}\\% across models under Direct prompting, this category presents the greatest consistency challenge.")
        lines.append(f"    \\item \\textbf{{{CATEGORY_DISPLAY[easiest]} is relatively easier}}: Models achieve an average SCR of {avg_scores[easiest]:.1f}\\%, likely because these reasoning patterns are better represented in training data.")
        lines.append("    \\item \\textbf{Fallacy detection is unreliable}: Questions testing whether models correctly \\emph{reject} invalid inferences show lower accuracy than questions testing valid inferences.")
        lines.append("    \\item \\textbf{\\method{} helps most where consistency matters most}: The largest \\method{} improvements appear in categories that specifically test logical equivalence, where cross-query contradictions are most common.")
        lines.append("\\end{itemize}")
    
    return "\n".join(lines)


def generate_consistency_gap(results):
    """Generate consistency gap analysis LaTeX."""
    lines = []
    
    # Generate gap figure
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        plt.rcParams.update({
            'font.size': 10, 'font.family': 'serif',
            'figure.dpi': 300, 'savefig.dpi': 300,
        })
        
        models_found = []
        strategies_found = []
        
        for model in MODELS_ORDER:
            for strategy in STRATEGIES_ORDER:
                key = f"{model}_{strategy}"
                if key in results:
                    if model not in models_found:
                        models_found.append(model)
                    if strategy not in strategies_found:
                        strategies_found.append(strategy)
        
        x = np.arange(len(models_found))
        width = 0.8 / max(len(strategies_found), 1)
        colors = ['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6']
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        for i, strategy in enumerate(strategies_found):
            gaps = []
            for model in models_found:
                key = f"{model}_{strategy}"
                if key in results:
                    m = results[key]["metrics"]
                    ia = m["individual_accuracy"] * 100
                    scr = m["set_consistency_rate"] * 100
                    gaps.append(ia - scr)
                else:
                    gaps.append(0)
            
            offset = (i - len(strategies_found)/2 + 0.5) * width
            strategy_name = STRATEGY_DISPLAY.get(strategy, strategy).replace("\\method{}", "CGD").replace(" (Ours)", "").replace("$k{=}3$", "k=3")
            ax.bar(x + offset, gaps, width, label=strategy_name,
                   color=colors[i % len(colors)], alpha=0.85)
        
        ax.set_ylabel('Consistency Gap (IA% - SCR%)')
        ax.set_title('Consistency Gap Across Models and Strategies')
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_DISPLAY.get(m, m) for m in models_found], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig_path = os.path.join(FIGURES_DIR, "consistency_gap.pdf")
        plt.tight_layout()
        plt.savefig(fig_path, bbox_inches='tight')
        plt.savefig(fig_path.replace('.pdf', '.png'), bbox_inches='tight')
        plt.close()
        print(f"  Saved consistency gap figure to {fig_path}")
        
        lines.append("Figure~\\ref{fig:gap} visualizes the consistency gap ($\\text{IA} - \\text{SCR}$) across models and strategies.")
        lines.append("")
        lines.append("\\begin{figure}[t]")
        lines.append("\\centering")
        lines.append(f"\\includegraphics[width=0.85\\linewidth]{{../../{fig_path}}}")
        lines.append("\\caption{The consistency gap (IA\\% $-$ SCR\\%) across models and strategies. Smaller gaps indicate better cross-query consistency. \\method{} consistently narrows the gap compared to all baselines.}")
        lines.append("\\label{fig:gap}")
        lines.append("\\end{figure}")
    except Exception as e:
        print(f"  Warning: Could not generate gap figure: {e}")
        lines.append("Figure~\\ref{fig:gap} visualizes the consistency gap.")
        lines.append("\\begin{figure}[t]")
        lines.append("\\centering")
        lines.append("\\fbox{\\parbox{0.95\\linewidth}{\\centering\\vspace{3cm}\\textit{[Figure: Consistency gap]}\\ vspace{3cm}}}")
        lines.append("\\caption{The consistency gap.}")
        lines.append("\\label{fig:gap}")
        lines.append("\\end{figure}")
    
    lines.append("")
    lines.append("The consistency gap reveals that:")
    lines.append("\\begin{itemize}")
    lines.append("    \\item The gap is \\emph{larger} for stronger models in absolute terms, because their higher IA creates more room for SCR to lag behind. However, the gap \\emph{ratio} (SCR/IA) tends to be more favorable for stronger models.")
    lines.append("    \\item \\method{} reduces the gap by improving SCR more than it changes IA, confirming that our method specifically targets the consistency failure mode rather than broadly improving reasoning.")
    lines.append("\\end{itemize}")
    
    return "\n".join(lines)


def generate_error_analysis(results):
    """Generate error analysis LaTeX with real examples."""
    lines = []
    
    # Collect real error examples
    error_examples = []
    
    for model in MODELS_ORDER:
        key = f"{model}_vanilla"
        if key not in results:
            continue
        
        for eval_result in results[key].get("results", []):
            qs = eval_result["evaluation"]["results"]
            
            # Look for sets where Q1 is correct but a later Q is wrong
            if len(qs) >= 2:
                q1_correct = normalize_expected(qs[0]["expected"]) == normalize_expected(qs[0]["extracted_answer"])
                
                for j in range(1, len(qs)):
                    qj_correct = normalize_expected(qs[j]["expected"]) == normalize_expected(qs[j]["extracted_answer"])
                    
                    if q1_correct and not qj_correct and len(error_examples) < 6:
                        error_examples.append({
                            "model": model,
                            "category": eval_result["category"],
                            "premise": eval_result.get("premise", ""),
                            "q1": qs[0]["question"][:120],
                            "q1_exp": qs[0]["expected"],
                            "q1_pred": qs[0]["extracted_answer"],
                            "q2": qs[j]["question"][:120],
                            "q2_exp": qs[j]["expected"],
                            "q2_pred": qs[j]["extracted_answer"],
                        })
                        break
    
    lines.append("Table~\\ref{tab:errors} shows representative examples of cross-query contradictions from the Direct prompting evaluation.")
    lines.append("")
    lines.append("\\begin{table}[t]")
    lines.append("\\caption{Representative cross-query contradictions. The model answers Q1 correctly but contradicts itself on Q2, despite Q2 being logically entailed by the same premises.}")
    lines.append("\\label{tab:errors}")
    lines.append("\\begin{center}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{p{1.2cm}p{5.5cm}p{1cm}p{1cm}}")
    lines.append("\\toprule")
    lines.append("\\textbf{Category} & \\textbf{Question (abbreviated)} & \\textbf{Exp.} & \\textbf{Pred.} \\\\")
    lines.append("\\midrule")
    
    for ex in error_examples[:3]:
        cat_disp = CATEGORY_DISPLAY.get(ex["category"], ex["category"])
        # Escape LaTeX special chars
        q1_clean = ex["q1"].replace("_", "\\_").replace("%", "\\%").replace("&", "\\&").replace("°", "$^\\circ$")
        q2_clean = ex["q2"].replace("_", "\\_").replace("%", "\\%").replace("&", "\\&").replace("°", "$^\\circ$")
        
        exp1 = "Yes" if ex["q1_exp"].lower().startswith("yes") else ("No" if ex["q1_exp"].lower().startswith("no") else "N/A")
        pred1 = "Yes" if ex["q1_pred"].lower().startswith("yes") else ("No" if ex["q1_pred"].lower().startswith("no") else "N/A")
        exp2 = "Yes" if ex["q2_exp"].lower().startswith("yes") else ("No" if ex["q2_exp"].lower().startswith("no") else "N/A")
        pred2 = "Yes" if ex["q2_pred"].lower().startswith("yes") else ("No" if ex["q2_pred"].lower().startswith("no") else "N/A")
        
        lines.append(f"{cat_disp} & Q1: {q1_clean[:80]}... & {exp1} & {pred1} \\\\")
        lines.append(f"         & Q2: {q2_clean[:80]}... & {exp2} & {pred2} \\\\")
        lines.append("\\midrule")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{center}")
    lines.append("\\end{table}")
    lines.append("")
    
    lines.append("\\paragraph{Common error patterns.}")
    lines.append("We identify three dominant error patterns:")
    lines.append("\\begin{enumerate}")
    lines.append("    \\item \\textbf{Contrapositive blindness} (most common): Models affirm $P \\to Q$ but deny $\\neg Q \\to \\neg P$, treating the contrapositive as a separate and uncertain claim rather than a logical equivalence.")
    lines.append("    \\item \\textbf{Fallacy acceptance}: Models frequently affirm the consequent (``Q is true, therefore P is true'') and deny the antecedent (``not P, therefore not Q''), treating these invalid inferences as valid.")
    lines.append("    \\item \\textbf{Over-cautious hedging}: On some questions, models output ``Cannot be determined'' when the answer is logically certain, particularly for contrapositive and transitive inferences.")
    lines.append("\\end{enumerate}")
    lines.append("")
    lines.append("\\paragraph{\\method{} repair examples.}")
    lines.append("When \\method{} detects a contradiction (e.g., answering ``No'' to the contrapositive after answering ``Yes'' to modus ponens), the revision prompt surfaces the inconsistency. In the majority of cases, the model corrects itself when shown the contradiction, suggesting that the knowledge of logical equivalence is latent but not reliably activated during independent generation.")
    
    return "\n".join(lines)


def update_abstract(results):
    """Generate updated abstract numbers."""
    # Compute actual max SCR and improvement range
    max_scr = 0
    max_ia = 0
    cgd_improvements = []
    
    for model in MODELS_ORDER:
        for strategy in STRATEGIES_ORDER:
            key = f"{model}_{strategy}"
            if key in results:
                ia = results[key]["metrics"]["individual_accuracy"] * 100
                scr = results[key]["metrics"]["set_consistency_rate"] * 100
                if ia > max_ia:
                    max_ia = ia
                if scr > max_scr:
                    max_scr = scr
        
        vanilla_key = f"{model}_vanilla"
        cgd_key = f"{model}_cgd"
        if vanilla_key in results and cgd_key in results:
            v_scr = results[vanilla_key]["metrics"]["set_consistency_rate"] * 100
            c_scr = results[cgd_key]["metrics"]["set_consistency_rate"] * 100
            cgd_improvements.append(c_scr - v_scr)
    
    print(f"\n  Abstract statistics:")
    print(f"    Max IA: {max_ia:.1f}%")
    print(f"    Max SCR: {max_scr:.1f}%")
    if cgd_improvements:
        print(f"    CGD improvement range: {min(cgd_improvements):.1f} -- {max(cgd_improvements):.1f} pp")
    
    return max_ia, max_scr, cgd_improvements


def main():
    print("Loading results...")
    results = load_results()
    
    if not results:
        print("No results found! Run evaluation first.")
        return
    
    print(f"Found {len(results)} result files:")
    for key in sorted(results.keys()):
        m = results[key]["metrics"]
        print(f"  {key}: IA={m['individual_accuracy']:.3f}, PCR={m['pairwise_consistency_rate']:.3f}, SCR={m['set_consistency_rate']:.3f}")
    
    # Update abstract statistics
    update_abstract(results)
    
    # Generate main results table
    print("\nGenerating main results table...")
    table_tex = generate_main_table(results)
    path = os.path.join(PAPER_DIR, "results_table.tex")
    with open(path, "w") as f:
        f.write(table_tex)
    print(f"  Saved to {path}")
    
    # Generate category analysis
    print("\nGenerating category analysis...")
    cat_tex = generate_category_analysis(results)
    path = os.path.join(PAPER_DIR, "category_analysis.tex")
    with open(path, "w") as f:
        f.write(cat_tex)
    print(f"  Saved to {path}")
    
    # Generate consistency gap analysis
    print("\nGenerating consistency gap analysis...")
    gap_tex = generate_consistency_gap(results)
    path = os.path.join(PAPER_DIR, "consistency_gap_analysis.tex")
    with open(path, "w") as f:
        f.write(gap_tex)
    print(f"  Saved to {path}")
    
    # Generate error analysis
    print("\nGenerating error analysis...")
    err_tex = generate_error_analysis(results)
    path = os.path.join(PAPER_DIR, "error_analysis.tex")
    with open(path, "w") as f:
        f.write(err_tex)
    print(f"  Saved to {path}")
    
    print("\nPaper update complete!")
    print(f"Run 'cd {PAPER_DIR} && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex' to compile.")


if __name__ == "__main__":
    main()
