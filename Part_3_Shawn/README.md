# Person 3 — ReAct Detective Agent
### KLab × arXiv Interdisciplinary Research Opportunity Detection System

**MACS 37005 Final Project | University of Chicago**
**Role: Person 3 (Shawn) — ReAct Agent / "Detective Maker"**

---

## Overview

This module implements a **ReAct (Reasoning + Acting) Agent** that autonomously detects interdisciplinary research opportunities between the Knowledge Lab (KLab) at UChicago and arXiv's paper corpus. The agent acts as a "detective": given a KLab research direction, it calls a suite of data tools, synthesizes evidence, and recommends the most promising cross-disciplinary intersections.

The agent integrates upstream outputs from two teammates:
- **Person 1 (Leo)** — 4-metric embedding system scoring 358,943 KLab × arXiv paper pairs (similarity, perplexity, steering score, sweet-spot flag)
- **Person 2 (Xiong)** — Causal inference analysis (TARNet + DML) on 2.9M arXiv papers; RAG index of causal evidence documents

---

## Theoretical Foundations

| Week | Concept | Application in This Module |
|------|---------|---------------------------|
| Week 5 | MCP / Tool Registry | 6 tools registered in OpenAI function-calling schema |
| Week 8 | ReAct Framework | Thought → Action → Observation loop via `tool_choice="auto"` |
| Week 3 | RAG | FAISS + SentenceTransformer retrieval of causal evidence |
| Week 2/6 | Embeddings & Steering Vectors | Consuming Leo's perplexity / steering scores as tool outputs |
| Week 4 | Causal Inference | Consuming Xiong's DML-PLR ATE = +0.85pp as grounding anchor |

---

## File Structure

```
Shawn's ReAct/
├── main.ipynb              # Full notebook (35 cells): setup → tools → ReAct → viz → results
├── figures/
│   ├── fig1_system_architecture.png   # System overview diagram
│   ├── fig2_react_loop.png            # ReAct loop flowchart
│   ├── fig3_trajectory.png            # Agent trajectory visualization
│   ├── fig4_discovery_results.png     # Task results summary
│   └── fig5_comparison.png            # ReAct vs baseline comparison
└── output/
    ├── discovery_results.json         # Full agent trajectories (8 tasks)
    ├── baseline_results.json          # Direct LLM answers (3 tasks, no tools)
    ├── comparison_scores.json         # LLM-as-judge scores
    ├── task_summary.csv               # Per-task stats (tool calls, steps, model)
    └── top3_opportunities.json        # Top 3 ranked opportunities
```

---

## The 6 Tools (MCP / Week 5)

| Tool | Data Source | Purpose |
|------|------------|---------|
| `search_klab_sweet_spots` | Leo's scored CSV | Find sweet-spot KLab×arXiv pairs by keyword |
| `get_domain_novelty_stats` | Leo's scored CSV | Perplexity & steering stats for an arXiv domain |
| `get_domain_convergence` | Leo's convergence CSV | Temporal convergence trend between two domains (2019–2026) |
| `query_causal_evidence` | Xiong's FAISS index | RAG retrieval of causal inference documents |
| `compute_opportunity_score` | Leo + Xiong combined | Composite 0–1 opportunity score with star rating |
| `get_causal_effect_summary` | Xiong's causal estimates | Summary of ATE estimates by aspect |

**Sweet-spot definition:** arXiv papers with similarity to a KLab paper in [0.3, 0.7] — semantically related but not redundant, and with high perplexity (surprising) and high steering score (interdisciplinary).

---

## The ReAct Loop (Week 8)

```
User Question
     │
     ▼
┌─────────────────────────────────┐
│  GPT-4o (tool_choice="auto")    │
│                                 │
│  Thought: "I should search..."  │
│       │                         │
│       ▼                         │
│  Action: call tool(args)        │
│       │                         │
│       ▼                         │
│  Observation: tool result       │
│       │                         │
│       └──── loop (max 10) ──────┤
│                                 │
│  Final Answer (no tool_calls)   │
└─────────────────────────────────┘
```

Typical tool call sequence per task:
```
search_klab_sweet_spots → get_domain_novelty_stats → get_domain_convergence
    → query_causal_evidence → compute_opportunity_score
```

---

## The 8 Discovery Tasks

| # | Task Name | Tool Calls | Steps | Result |
|---|-----------|-----------|-------|--------|
| 1 | Team Size × NLP | 7 | 8 | ✅ |
| 2 | Science Acceleration × Bioinformatics | 4 | 5 | ✅ |
| 3 | Knowledge Networks × Statistical Learning | 5 | 6 | ✅ |
| 4 | Political Polarization × Economics | 7 | 8 | ✅ |
| 5 | Disease Space × Electrical Engineering | 2 | 3 | ✅ |
| 6 | Citation Prediction × Quantum Physics | 3 | 3 | ⚠️ No final answer |
| 7 | Language Information Density × Physics | 5 | 6 | ✅ |
| 8 | Failure Dynamics × Mathematics | 0 | 0 | ⚠️ No result |

**6 of 8 tasks** completed successfully. Average: **4.1 tool calls / 5.4 steps per task**.

---

## Top 3 Discovered Opportunities

| Rank | Opportunity | Score | Key Signal |
|------|------------|-------|-----------|
| 🥇 1 | Language Information Density × **Physics** | 0.64 | physics avg perplexity=32.07, cs↔physics convergence +0.22 since 2019 |
| 🥈 2 | Disease Space × **Electrical Engineering (eess)** | 0.60 | sweet-spot pairs found, eess convergence confirmed |
| 🥉 3 | Political Polarization × **Economics (econ)** | 0.58 | 7 tool calls exploring multiple angles |

All top opportunities are anchored by the causal estimate: interdisciplinary papers are **+0.85 percentage points** more likely to reach top-10% citations (DML-PLR Lasso, 95% CI: [0.54, 1.15]).

---

## Comparison Experiment: ReAct Agent vs. Direct LLM

3 tasks evaluated by GPT-4o-mini as judge (scores 1–5):

| Dimension | ReAct Agent | Baseline LLM (no tools) | Δ |
|-----------|------------|------------------------|---|
| Specificity | 3.33 | 3.33 | = |
| Novelty | **4.00** | 3.67 | +0.33 |
| Feasibility | **4.00** | **4.00** | = |
| **Evidence** | **3.67** | 2.33 | **+1.34** |

The ReAct agent's primary advantage is in **data-backed evidence**: it grounds recommendations in actual perplexity scores, convergence trends, and causal effect estimates rather than relying solely on LLM prior knowledge.

---

## How to Run

### Requirements

```bash
pip install openai faiss-cpu sentence-transformers pandas numpy matplotlib
```

### Data dependencies (from teammates)

```
Leo/Other Outputs/
├── klab_papers.json
├── convergence_yearly.csv
└── scored_all_4metrics (no abstract).csv   # ~75 MB

Xiong's output/outputs/
├── causal_estimates.csv
├── causal_evidence_docs.json
└── causal_evidence_index.faiss
```

### API key

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."
```

Or use Google Colab:
```python
from google.colab import userdata
os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
```

### Run

Open `main.ipynb` and run all cells top-to-bottom. Intermediate results are cached in `output/` so expensive API calls are not repeated on re-runs.

**No GPU required.** Runs on any CPU machine or free Google Colab instance.

Estimated cost: ~$0.50–1.00 USD (GPT-4o for 8 tasks + comparison experiment).

---

## Demand Characteristics & Methodology Notes

A key concern: does the agent simply reproduce LLM prior knowledge rather than discovering from data?

**Mitigations implemented:**

| Risk | Mitigation | Evidence |
|------|-----------|---------|
| Agent told to find "surprising" opportunities | Blind comparison experiment | ReAct scores +1.34 higher on Evidence vs. direct LLM |
| Results may reflect training data | All scores grounded in Leo's embedding metrics | Tool outputs are computed from actual paper vectors |
| Causal claim may be confounded | Independent causal analysis (Xiong, DML) | ATE = +0.85pp, CI does not include 0 |

The causal estimate from Person 2 serves as an **independent empirical anchor** — it was computed separately from the agent's reasoning, preventing circular validation.

---

## Figures

| Figure | Description |
|--------|------------|
| `fig1_system_architecture.png` | Full system: Leo → Xiong → Tool Registry → ReAct Agent → Output |
| `fig2_react_loop.png` | Step-by-step ReAct loop diagram |
| `fig3_trajectory.png` | Trajectory of tool calls for a representative task |
| `fig4_discovery_results.png` | Tool usage frequency + opportunity scores across 8 tasks |
| `fig5_comparison.png` | Radar + bar chart comparing ReAct vs. baseline LLM |
