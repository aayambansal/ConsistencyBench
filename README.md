# ConsistencyBench

**Logical Consistency Under Pressure: Probing and Repairing Cross-Query Contradictions in LLMs**

*ICLR 2026 Workshop on Logical Reasoning of Large Language Models*

## Overview

LLMs answer individual logic questions with reasonable accuracy, yet frequently contradict themselves across logically related queries. ConsistencyBench is a benchmark of **493 question sets (1,904 questions)** spanning six categories of logical reasoning, designed to measure **cross-query logical consistency**.

We evaluate **18 frontier LLMs** and find that even the best model (GPT-4.1) achieves only **46.7% set-level consistency** despite 83.0% individual accuracy, revealing consistency gaps of **36-57 percentage points**. We propose **Consistency-Guided Decoding (CGD)**, a training-free inference-time method that improves consistency for **16/17 models** (+6.6pp mean, up to +19.7pp).

## Key Results

| Model | Individual Accuracy | Set Consistency | Gap |
|-------|:------------------:|:---------------:|:---:|
| GPT-4.1 | 83.0% | 46.7% | 36.4pp |
| Qwen 2.5 72B | 75.0% | 32.7% | 42.3pp |
| Claude Opus 4.6 | 74.8% | 23.3% | 51.4pp |
| DeepSeek-R1 | 70.2% | 21.7% | 48.5pp |
| GPT-4o | 70.8% | 20.0% | 50.8pp |
| GPT-5.2 | 69.3% | 12.0% | 57.3pp |
| o3 | 67.2% | 12.0% | 55.2pp |
| GPT-5 | 62.9% | 9.3% | 53.5pp |

*Full results for all 18 models are in the paper.*

**CGD improves consistency broadly:**
- 16/17 models improved (94%)
- Mean improvement: +6.6pp SCR
- Best improvement: +19.7pp (GPT-4o)
- Also improves individual accuracy by +2.8pp on average

## Benchmark Categories

| Category | Pattern | Sets | Fallacy Tested |
|----------|---------|:----:|----------------|
| Contrapositive | P->Q equiv ~Q->~P | 85 | Affirming the consequent |
| Transitivity | A->B, B->C |- A->C | 85 | Chain break |
| Syllogistic | forall x: A(x)->B(x) | 85 | Converse error |
| Negation | P vs ~P | 85 | Negation incoherence |
| Modus Tollens | P->Q, ~Q |- ~P | 85 | Denying the antecedent |
| Commonsense | Everyday entailment | 68 | Reverse entailment |

## Repository Structure

```
ConsistencyBench/
├── data/                          # Benchmark data
│   ├── consistency_bench.json     # Full benchmark (493 sets, 1,904 questions)
│   └── consistency_bench_sample.json  # Evaluation sample (300 sets)
├── src/                           # Source code
│   ├── generate_benchmark.py      # Benchmark generation
│   ├── evaluate.py                # Evaluation pipeline (vanilla, CoT, CGD)
│   ├── run_all.py                 # Run all evaluations
│   ├── analyze_results.py         # Results analysis and visualization
│   └── update_paper.py            # Generate LaTeX tables and figures from results
├── results/                       # All 37 evaluation result files (JSON)
│   ├── {model}_vanilla.json       # 18 direct prompting results
│   ├── {model}_cgd.json           # 17 CGD results
│   └── {model}_cot.json           # 2 chain-of-thought results
├── eval_logs/                     # Evaluation logs for all 37 runs
├── figures/                       # Generated figures (PDF + PNG)
│   ├── figure1_overview.pdf       # Figure 1: consistency problem illustration
│   ├── cgd_pipeline.pdf           # CGD pipeline diagram
│   ├── category_heatmap.pdf       # Per-category SCR heatmap
│   └── consistency_gap.pdf        # Consistency gap visualization
├── scripts/                       # Figure generation scripts
│   ├── generate_figure1.py        # Figure 1 (matplotlib)
│   └── generate_cgd_diagram.py    # CGD pipeline diagram (matplotlib)
├── final-paper/                   # ICLR 2026 workshop paper
│   ├── main.tex                   # Paper source
│   ├── main.pdf                   # Compiled paper
│   ├── references.bib             # Bibliography
│   └── ...                        # Style files and supporting TeX
└── LICENSE
```

## Metrics

- **Individual Accuracy (IA):** Fraction of individual questions answered correctly.
- **Pairwise Consistency Rate (PCR):** Fraction of within-set pairs where both are correct.
- **Set-Level Consistency Rate (SCR):** Fraction of sets where *all* questions are correct.
- **Consistency Gap:** IA - SCR. Measures how much single-query accuracy overestimates logical reasoning.

## Consistency-Guided Decoding (CGD)

CGD is a training-free, model-agnostic inference-time method:

1. **Generate:** The LLM produces an initial answer for each question.
2. **Check:** An NLI checker (GPT-4o-mini) tests the answer against all prior answers for contradictions.
3. **Repair:** If a contradiction is found, a revision prompt with the conflicting answer and shared premise elicits a corrected response.

CGD works with any LLM (including black-box APIs), requires no fine-tuning, and adds minimal overhead.

## Models Evaluated

**18 frontier LLMs across 6 families:**

- **OpenAI:** GPT-5.2, GPT-5, GPT-5 Mini, GPT-4.1, GPT-4o, o3
- **Anthropic:** Claude Opus 4.6, Claude Sonnet 4.6, Claude Sonnet 4.5, Claude 3.5 Sonnet
- **Google:** Gemini 2.5 Pro, Gemini 2.5 Flash, Gemini 2.0 Flash
- **DeepSeek:** DeepSeek V3.2, DeepSeek-R1
- **Meta:** Llama 3.3 70B, Llama 3.1 70B
- **Qwen:** Qwen 2.5 72B

## Citation

```bibtex
@inproceedings{bansal2026consistency,
  title={Logical Consistency Under Pressure: Probing and Repairing Cross-Query Contradictions in {LLMs}},
  author={Bansal, Aayam},
  booktitle={ICLR 2026 Workshop on Logical Reasoning of Large Language Models},
  year={2026}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
