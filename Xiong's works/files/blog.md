# Section 2: Does Crossing Boundaries Actually Pay Off? Causal Evidence from 2.9 Million Papers

*By Donghua Xiong (Person 2: Causal Evidence Officer)*

## The Question Everyone Assumes They Know the Answer To

"Interdisciplinary research leads to higher impact." You've probably heard this claim a hundred times. Universities love it, funding agencies repeat it, and it sounds intuitively right. But is it actually *true* in a causal sense? Or are we just observing that talented researchers — who happen to produce high-impact work anyway — also tend to cross disciplinary boundaries?

This is a textbook example of **confounding bias** (混淆偏差). In statistics, we know that correlation does not imply causation. A naive comparison between interdisciplinary and single-domain papers conflates the genuine effect of crossing boundaries with the influence of confounders like team size, publication year, and field-specific citation cultures. To untangle these, we need **causal inference** (因果推断) — methods that can estimate what would have happened to the *same* paper had it been single-domain instead of interdisciplinary, or vice versa.

This section uses two causal inference methods from Week 4 of our course — **TARNet** (Treatment-Agnostic Representation Network) and **DML** (Double/Debiased Machine Learning) — on the full arXiv dataset of **2.9 million papers** to answer: after controlling for confounders, does crossing disciplinary boundaries *cause* higher citation impact?

## Data: The Full arXiv Universe

We start with the complete Kaggle arXiv metadata snapshot: **2,968,861 papers**, spanning 38 broad domains and 176 subcategories from 2007 to 2025.

![Fig 1: arXiv growth and interdisciplinary trends](figures/fig1_yearly_distribution.png)
*Fig 1: Left — Exponential growth in arXiv submissions, accelerating after 2015 (the deep learning boom). Right — The interdisciplinary rate follows a U-shaped curve, rising from ~16% (2007) to ~30% (2008), then fluctuating between 25-30% before climbing to ~35% by 2020.*

We define a paper as **interdisciplinary** if it spans 2 or more broad arXiv domains (e.g., a paper tagged both `cs.LG` and `stat.ML` crosses the CS and Statistics domains). By this definition, **28.08%** of all papers are interdisciplinary — a substantial treatment group that avoids the extreme imbalance problem in causal inference.

![Fig 2: Domain distribution and interdisciplinary tendency](figures/fig2_category_distribution.png)
*Fig 2: Left — CS dominates with 730K+ papers, followed by Math (560K) and Condensed Matter (350K). Right — Interdisciplinary tendency varies enormously: math-ph, eess, and econ exceed 70%, while astro-ph, math, and cond-mat are below 20%. This heterogeneity is a key confounder we must control.*

One counterintuitive finding emerges immediately:

![Fig 3: Author count distribution](figures/fig3_author_distribution.png)
*Fig 3: Interdisciplinary papers have **fewer** authors on average (3.44 vs 3.80). Crossing boundaries isn't about assembling large teams — it's driven by individuals who think across fields.*

![Fig 4: Cross-domain co-occurrence heatmap](figures/fig4_cross_domain_heatmap.png)
*Fig 4: The top cross-domain pair is CS × Math (130K papers), followed by CS × EESS (95K). CS serves as the largest "bridge" discipline, connecting with nearly every other field.*

## Citation Data: 98.47% Coverage via Semantic Scholar

Using the Semantic Scholar Batch API (500 papers/batch, ~1s response, ~1s interval), we retrieved citation counts for **2,845,648 papers** (98.47% coverage). The citation distribution follows a classic **power law** (幂律分布): mean 28.90, median only 7, maximum 221,897.

![Fig 5: Citation distribution](figures/fig5_citation_distribution.png)
*Fig 5: Left — Log-scale citation distribution showing the heavy right tail. Center — Interdisciplinary papers (orange) are slightly right-shifted. Right — Median citations by domain: physics fields (hep-th: 12, astro-ph: 11) substantially outpace CS (6) and math (4).*

The **naive comparison** shows interdisciplinary papers have +33% higher median citations (8 vs 6) and +10.5% higher mean citations (31.03 vs 28.07). But how much of this is causal?

![Fig 6: Naive interdisciplinary advantage by year and team size](figures/fig6_naive_advantage.png)
*Fig 6: Left — The naive advantage is positive in every year but declining over time (from +6 in 2012-2013 to +1 in 2023-2024). Right — Small teams (2-3 people) show the largest advantage (+2), while large teams (7+) show a slight **disadvantage** (-1). These patterns demand causal verification.*

## Causal Inference: Separating Signal from Noise

### Setup

- **Treatment (T):** Is the paper interdisciplinary (n_broad_domains ≥ 2)?
- **Outcome (Y):** Does it rank in the **Top 10%** by citations within its primary category and year? (This clever outcome design inherently controls for field-specific and temporal citation culture differences.)
- **Covariates (X):** 22 features — log team size, year, primary category (one-hot), time period dummies
- **Identification:** Selection on Observables (可观测变量下的条件独立性假设)
- **Effective sample:** 2,500,429 papers

The **naive gap** is 3.14 percentage points (pp): 12.45% of interdisciplinary papers reach Top 10%, versus 9.31% of single-domain papers.

### Method 1: TARNet (Week 4)

TARNet (Treatment-Agnostic Representation Network) uses a shared representation layer to learn common features, then splits into two separate output heads — one predicting Y(0) (the counterfactual outcome under no treatment) and one predicting Y(1) (under treatment). The treatment-masked loss ensures each head learns only from its factual observations. We implemented TARNet in PyTorch (the course used TensorFlow/Keras; the underlying math — BCE loss, Adam optimizer, backpropagation — is identical).

**Result:** After 84 epochs of training (early stopping with patience=15):

> **TARNet ATE (Average Treatment Effect) = 2.10 pp**

This means interdisciplinary papers are 2.10 percentage points more likely to reach the Top 10% — a real but modest effect. Critically, the **confounding bias is 1.04 pp**: about one-third of the naive 3.14pp gap was spurious, driven by confounders rather than genuine causal impact.

![Fig 7: TARNet results](figures/fig7_tarnet_results.png)
*Fig 7: Left — Training curve (84 epochs, early stopping). Center — CATE (Conditional Average Treatment Effect) distribution: most papers benefit modestly, but the spread (SD = 3.07pp) reveals high heterogeneity. Right — CATE by team size: small teams benefit most.*

### Method 2: DML (Week 4)

DML (Double/Debiased Machine Learning, Chernozhukov et al. 2018) provides what TARNet cannot: **confidence intervals** (置信区间) and **p-values**. It uses orthogonalization and cross-fitting to remove regularization bias. We ran 6 specifications: 3 nuisance learners (Lasso, Random Forest, XGBoost) × 2 DML models (PLR — Partially Linear Regression, and IRM — Interactive Regression Model) on a stratified subsample of 500K papers.

| Learner | PLR ATE (pp) | PLR 95% CI | IRM ATE (pp) | IRM 95% CI |
|---------|:---:|:---:|:---:|:---:|
| Lasso | 0.85 | [0.54, 1.15] | 0.51 | [-0.05, 1.06] |
| Random Forest | 1.68 | [1.38, 1.99] | 2.72 | [2.59, 2.84] |
| XGBoost | 0.99 | [0.65, 1.32] | 2.42 | [2.28, 2.56] |

**5 out of 6 specifications are significant at p < 0.001.** The one exception (Lasso-IRM, p = 0.073) uses the most restrictive linear learner with the most flexible model — its marginal significance is expected. Nonlinear learners (RF, XGBoost) consistently estimate larger effects than Lasso, suggesting nonlinear confounding interactions that linear models underfit.

![Fig 9: Method comparison](figures/fig9_method_comparison.png)
*Fig 9: All causal estimates (orange/blue) fall below the naive gap (gray), confirming confounding bias exists. DML confidence intervals (horizontal lines) mostly exclude zero, supporting a genuine positive causal effect.*

## Who Benefits Most? Heterogeneous Effects

The average effect hides dramatic variation across subgroups.

![Fig 8: Heterogeneous causal effects](figures/fig8_heterogeneous_effects.png)
*Fig 8: Left — CATE heatmap by team size × time period: small teams consistently benefit, large/mega teams see near-zero or negative effects. Right — CATE by primary domain: the causal return to crossing boundaries varies enormously across fields.*

Key findings:
- **Small teams (2-3 people)** gain the most from interdisciplinary work — consistent with our earlier observation that boundary-crossing is driven by individual versatility, not team assembly
- **Large teams (7+)** see near-zero or even negative effects — perhaps because large teams already aggregate diverse expertise internally
- **The interdisciplinary premium is declining** over time (2019-2024 effects weaker than 2013-2018), possibly because crossing boundaries has become commonplace, reducing its "scarcity premium" (稀缺溢价)

## From Analysis to Tool: RAG Causal Evidence System

Raw numbers in a CSV aren't useful to an AI agent. To make our causal findings actionable for Person 3's DiscoveryAgent, we built a **RAG** (Retrieval-Augmented Generation, Week 3) pipeline that converts our results into a queryable knowledge base.

We structured all findings into **25 evidence documents** (10 general + 15 domain-specific), encoded them using **all-MiniLM-L6-v2** (384-dimensional vectors, matching Leo's embedding pipeline), and indexed them with **FAISS** (Facebook AI Similarity Search) for efficient retrieval. The complete pipeline: natural language query → vector similarity search → retrieve top-k evidence → LLM generates a synthesized answer grounded in our causal data.

![Fig 10: RAG retrieval quality](figures/fig10_rag_retrieval_quality.png)
*Fig 10: Left — L2 distances for Top-3 retrieved documents (0.3-0.9) are well-separated from non-relevant documents (0.6-1.8). Right — Heatmap confirms accurate semantic matching: each query retrieves its most relevant evidence document.*

The tool is packaged as an **MCP** (Model Context Protocol, Week 5) interface named `query_causal_evidence`, ready for Person 3 to integrate into the ReAct agent. When the DiscoveryAgent asks "Is it worth pursuing CS × Biology interdisciplinary research for a 3-person team?", our tool retrieves the relevant causal evidence and generates a data-grounded answer — not speculation, but findings backed by 2.9 million papers.

## Summary

| What we measured | Result |
|---|---|
| Naive correlation | +3.14 pp |
| Causal effect (TARNet) | +2.10 pp |
| Causal effect (DML range) | +0.51 to +2.72 pp |
| Confounding bias | ~1.04 pp (~1/3 of naive gap) |
| Who benefits most | Small teams (2-3 people) |
| Who benefits least | Large teams (7+) |
| Trend | Premium declining over time |

Crossing disciplinary boundaries does cause higher citation impact — but the effect is roughly one-third smaller than naive correlations suggest, it varies enormously by team size and field, and it's shrinking over time. These nuanced, causally-grounded findings are exactly what a DiscoveryAgent needs to make informed recommendations rather than repeating the oversimplified "interdisciplinary = better" narrative.
