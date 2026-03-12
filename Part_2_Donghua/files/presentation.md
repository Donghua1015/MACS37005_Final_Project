# Presentation Script — Person 2: Causal Evidence Officer
# 每页对应一张slide，共8页，总时长约2分钟

---

## Slide 1 (~15s)

Hi everyone, I'm Donghua Xiong, the Causal Evidence Officer for our team.

My job is to answer a deceptively simple question: does crossing disciplinary boundaries actually *cause* higher research impact? Everyone assumes the answer is yes — but naive correlations can be misleading due to confounding factors. I used TARNet and DML from Week 4, plus a RAG tool from Week 3, to get a rigorous causal answer from 2.9 million arXiv papers.

---

## Slide 2 (~15s)

Here's our dataset. Nearly 3 million arXiv papers from 2007 to 2025, spanning 38 broad domains. About 28% of papers cross two or more domains — that's our treatment group. I retrieved citation counts for 98.47% of these papers using the Semantic Scholar Batch API. The outcome variable is whether a paper ranks in the top 10% of citations within its own category and year — this design cleverly controls for field-specific citation cultures.

---

## Slide 3 (~15s)

The naive comparison shows a 3.14 percentage point gap — interdisciplinary papers reach top 10% at 12.45% versus 9.31% for single-domain papers. But look at the breakdown: small teams show a positive advantage, while large teams actually show a slight *dis*advantage. And the gap is shrinking over time. These patterns suggest confounders are at work. The question is: how much of the 3.14 point gap is genuine causal effect?

---

## Slide 4 (~15s)

TARNet gives us the answer. Using a shared representation layer with two separate output heads — one for each treatment condition — and training on 2 million papers with early stopping, we get a causal ATE of 2.10 percentage points. That means about one-third of the naive gap — 1.04 points — was confounding bias from field effects, team size, and year trends. The CATE distribution shows high heterogeneity: some papers benefit a lot, others not at all.

---

## Slide 5 (~15s)

To get confidence intervals, I ran DML with three learners and two model specifications — six estimates total. Five out of six are significant at p less than 0.001. The causal effect ranges from 0.51 to 2.72 percentage points depending on the specification. All estimates fall below the naive 3.14 line, confirming that confounding bias is real. Nonlinear learners consistently estimate larger effects than Lasso, revealing nonlinear confounding interactions.

---

## Slide 6 (~15s)

Who benefits most? The heterogeneity analysis reveals a striking pattern. Small teams of 2-3 people gain the most from crossing boundaries — consistent with our earlier finding that interdisciplinary papers actually have *fewer* authors. Large teams see near-zero or even negative effects — they probably already have diverse expertise internally. And the premium is declining over time, possibly because interdisciplinary work has become the norm rather than the exception.

---

## Slide 7 (~15s)

To make these findings actionable for our DiscoveryAgent, I built a RAG tool. I structured all causal results into 25 evidence documents, encoded them with SentenceTransformer, and indexed them with FAISS. The tool is packaged as an MCP interface called `query_causal_evidence`. When the agent asks "Is it worth doing cross-domain work in a small team?", it retrieves causally-grounded evidence — not speculation. The retrieval quality plot confirms accurate semantic matching.

---

## Slide 8 (~15s)

To summarize: interdisciplinary research does have a real positive causal effect on citation impact — but it's roughly one-third smaller than naive correlations suggest. The effect is highly heterogeneous: small teams benefit most, large teams barely benefit at all, and the premium is shrinking over time. These nuanced findings are exactly what a DiscoveryAgent needs to give researchers informed, evidence-based recommendations. Thank you.
