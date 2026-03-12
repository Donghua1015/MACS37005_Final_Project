# Slides for Person 2: Causal Evidence Officer
# 共8张slides，对应约2分钟演讲

---

## Slide 1: Title & Role

**Person 2: Causal Evidence Officer**
因果证据官 — 跨学科研究的因果效应分析

Donghua Xiong | MS Data Science

核心问题: Does crossing disciplinary boundaries *cause* higher impact?
Or is the correlation driven by confounders?

方法: TARNet + DML (Week 4) + RAG Tool (Week 3)
数据: 2.9M arXiv papers + Semantic Scholar citations

---

## Slide 2: Data Overview — 2.9M Papers

![fig1](figures/fig1_yearly_distribution.png)

- 2,968,861 papers (2007-2025), 38 broad domains
- Treatment: interdisciplinary (≥2 broad domains) = 28.08%
- Citation data: 98.47% coverage via Semantic Scholar Batch API
- Outcome: Top 10% citations within same category & year

---

## Slide 3: Naive Gap = 3.14pp — But Is It Causal?

![fig6](figures/fig6_naive_advantage.png)

- 跨学科 Top10% rate: 12.45% vs 单领域 9.31%
- Naive gap = +3.14 percentage points
- But confounders exist: field effects, team size, year trends
- Small teams: +2 advantage | Large teams: -1 disadvantage
- Question: How much is real causal effect vs spurious correlation?

---

## Slide 4: TARNet — Causal Effect = 2.10pp

![fig7](figures/fig7_tarnet_results.png)

- Shared representation (128→64→32) + 2 separate output heads
- 84 epochs, early stopping (patience=15)
- **TARNet ATE = 2.10pp** (vs Naive 3.14pp)
- **Confounding bias = 1.04pp** (~1/3 of naive gap is spurious)
- CATE SD = 3.07pp → high heterogeneity across papers

---

## Slide 5: DML — 5/6 Specifications Significant

![fig9](figures/fig9_method_comparison.png)

| Learner | PLR (pp) | IRM (pp) |
|---------|:---:|:---:|
| Lasso | 0.85 | 0.51 |
| Random Forest | 1.68 | 2.72 |
| XGBoost | 0.99 | 2.42 |

- 5/6 significant at p < 0.001 (with 95% CI)
- All causal estimates < Naive → confounding confirmed
- Nonlinear learners > Lasso → nonlinear confounding exists

---

## Slide 6: Who Benefits? Heterogeneous Effects

![fig8](figures/fig8_heterogeneous_effects.png)

- **Small teams (2-3人):** strongest benefit from crossing boundaries
- **Large teams (7+):** near-zero or negative effect
- **Time trend:** premium declining (2019-2024 < 2013-2018)
- **By domain:** enormous variation across fields
- Insight: boundary-crossing is about individual versatility, not team assembly

---

## Slide 7: RAG Causal Evidence Tool

![fig10](figures/fig10_rag_retrieval_quality.png)

- 25 structured evidence documents (10 general + 15 domain-specific)
- all-MiniLM-L6-v2 (384D) + FAISS IndexFlatL2
- MCP tool interface: `query_causal_evidence`
- Agent asks natural language → retrieves causal evidence → LLM synthesizes
- Top-3 retrieval distances: 0.3-0.9 (well-separated from irrelevant docs)

---

## Slide 8: Summary — Key Takeaways

| Metric | Value |
|---|---|
| Naive correlation | +3.14 pp |
| Causal effect (TARNet) | +2.10 pp |
| Causal effect (DML) | +0.51 ~ +2.72 pp |
| Confounding bias | ~1/3 of naive gap |
| Best beneficiary | Small teams |
| Trend | Declining premium |

**Bottom line:** Interdisciplinary research has a real but modest causal benefit.
~1/3 of the naive advantage is spurious. Small teams benefit most. The premium is shrinking.

→ These findings feed into Person 3's DiscoveryAgent as causal evidence tools.
