# Does Crossing Boundaries Actually Pay Off? A Causal Deep Dive into 2.9 Million Papers

*Section by Donghua Xiong, Causal Analyst*

---

Everyone loves interdisciplinary research. Funding agencies celebrate it, universities reward it, and keynote speakers praise it. But here is a question that surprisingly few people ask with scientific rigor. Does crossing disciplinary boundaries actually *cause* higher research impact? Or are we just seeing a mirage created by confounding factors?

As the causal analyst on our DiscoveryAgent team, we took on this challenge using 2.9 million arXiv papers, two causal inference frameworks from deep learning, and a retrieval system that turns our findings into tools for an autonomous AI detective. This post walks through how we moved from raw correlation to causal evidence, and what that evidence reveals about who really benefits from going interdisciplinary.

![fig1_yearly_distribution.png](figures/fig1_yearly_distribution.png)
*Figure 1. Annual volume of arXiv papers from 2007 to 2025 and the changing rate of interdisciplinary publications over time.*

## The Dataset: Nearly 3 Million Scientific Lives

We started with the full arXiv metadata dump. After cleaning and filtering to papers published between 2007 and 2025, we had 2,890,002 papers spanning 38 broad domains and 176 subcategories. Among these, 28.08% cross two or more broad domains. We defined this as our treatment variable.

![fig2_category_distribution.png](figures/fig2_category_distribution.png)
*Figure 2. Top 20 primary categories by paper count and their respective interdisciplinary rates.*

The landscape is far from uniform. Computer Science dominates with over 730,000 papers, followed by Mathematics and Condensed Matter Physics. Interdisciplinary rates vary wildly across fields. Mathematical Physics and Electrical Engineering have rates above 70%, while Astrophysics and pure Mathematics stay below 20%.

To measure impact, we needed citation data. We used the Semantic Scholar Batch API to retrieve citation counts for all 2.89 million papers. The process took about 3.7 hours, querying 500 papers per batch. We achieved 98.47% coverage, with citation data for 2,845,648 papers.

```python
# Semantic Scholar Batch API query loop
for batch_start in range(0, len(paper_ids), BATCH_SIZE):
    batch = paper_ids[batch_start:batch_start + BATCH_SIZE]
    response = requests.post(
        "https://api.semanticscholar.org/graph/v1/paper/batch",
        json={"ids": [f"ArXiv:{pid}" for pid in batch]},
        params={"fields": "citationCount,year"}
    )
```

Rather than using raw citation counts, which are heavily skewed by field norms, we defined a binary outcome. A paper is a "hit" if it ranks in the top 10% of citations within its own category and publication year. This relative measure ensures that a highly cited Physics paper and a highly cited Economics paper are compared fairly.

![fig3_author_distribution.png](figures/fig3_author_distribution.png)
*Figure 3. Author count distributions for single-domain versus interdisciplinary papers. Interdisciplinary work tends to involve slightly larger teams.*

## The Naive Story: 3.14 Percentage Points

Before any causal analysis, the raw numbers look encouraging. Interdisciplinary papers reach top-10% status at a rate of 12.45%, compared to 9.31% for single-domain papers. That is a 3.14 percentage point gap.

But dig deeper and the picture gets complicated.

![fig6_naive_advantage.png](figures/fig6_naive_advantage.png)
*Figure 4. The naive interdisciplinary citation advantage broken down by publication year and team size. Small teams gain the most while large teams see diminishing returns.*

Small teams of 2 to 3 people show a strong interdisciplinary advantage of about +2 percentage points. Large teams of 7 or more actually show a slight disadvantage. And the gap has been shrinking over time. In the early 2010s, interdisciplinary papers had a clear edge. By 2020 and beyond, the advantage had narrowed to about +1 percentage point.

These patterns scream confounding. Maybe productive researchers simply tend to work across fields. Maybe certain hot domains naturally span multiple categories. Maybe team composition drives both interdisciplinarity and impact simultaneously. We needed causal inference to separate the signal from the noise.

![fig4_cross_domain_heatmap.png](figures/fig4_cross_domain_heatmap.png)
*Figure 5. Cross-domain co-occurrence heatmap for the top 20 arXiv domains. CS and Math form the most frequent pairing with over 130,000 joint papers.*

## Causal Estimation: TARNet

We turned to TARNet, the Treatment-Agnostic Representation Network from Week 4 of our course. TARNet learns a shared representation of confounders through a neural network trunk, then splits into two separate prediction heads for treated and control outcomes.

$$\begin{gathered} L = \frac{1}{n}\sum_{i=1}^{n}\left[T_i \cdot \mathcal{L}\!\left(Y_i,\, f_1\!\left(\Phi(X_i)\right)\right) + (1-T_i) \cdot \mathcal{L}\!\left(Y_i,\, f_0\!\left(\Phi(X_i)\right)\right)\right] \\[6pt] \text{where } X_i = \text{covariates (team size, field, year)},\; T_i = \begin{cases}1 & \text{interdisciplinary}\\0 & \text{single-domain}\end{cases},\; Y_i = \text{top-10\% indicator} \\[6pt] \Phi = \text{shared trunk network},\; f_0 = \text{control head},\; f_1 = \text{treated head},\; \mathcal{L} = \text{binary cross-entropy} \end{gathered}$$

Our architecture uses a trunk of three layers mapping 22 input features down to a 32-dimensional shared representation, with BatchNorm and ELU activations. Each treatment head then maps this representation to a probability through two more layers with Dropout.

```python
class TARNet(nn.Module):
    def __init__(self, input_dim=22):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ELU(),
            nn.Linear(128, 64),  nn.BatchNorm1d(64),  nn.ELU(),
            nn.Linear(64, 32),   nn.BatchNorm1d(32),  nn.ELU(),
        )
        self.head0 = nn.Sequential(  # single-domain outcome
            nn.Linear(32, 64), nn.ELU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ELU(), nn.Dropout(0.3),
            nn.Linear(32, 1),  nn.Sigmoid()
        )
        self.head1 = nn.Sequential(  # interdisciplinary outcome
            nn.Linear(32, 64), nn.ELU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ELU(), nn.Dropout(0.3),
            nn.Linear(32, 1),  nn.Sigmoid()
        )
```

We trained on 2 million papers, validated on 500,000, and used early stopping with patience of 15. Training converged at epoch 84 with a best validation loss of 0.3157.

![fig7_tarnet_results.png](figures/fig7_tarnet_results.png)
*Figure 6. TARNet training curve, the distribution of individual treatment effects, and CATE estimates by team size.*

The result is striking. TARNet estimates the Average Treatment Effect at **2.10 percentage points**, compared to the naive gap of 3.14. That means roughly one-third of the observed advantage, about 1.04 percentage points, was confounding bias from team size, field selection, and temporal trends.

But the analysis does not end with a single number. The standard deviation of the Conditional Average Treatment Effect across papers is 3.07 percentage points. This high heterogeneity means some papers benefit enormously from crossing boundaries while others do not benefit at all.

## Double Machine Learning: Six Specifications, One Conclusion

To get proper confidence intervals and test robustness, we ran Double/Debiased Machine Learning on a stratified subsample of 500,000 papers. We crossed three learners with two model specifications, producing six independent causal estimates.

$$\begin{gathered} \hat{\theta} = \frac{1}{n}\sum_{i=1}^{n}\frac{Y_i - \hat{g}(X_i)}{T_i - \hat{m}(X_i)} \\[6pt] \text{where } \hat{\theta} = \text{causal effect estimate},\; Y_i = \text{outcome},\; T_i = \text{treatment} \\[6pt] \hat{g}(X_i) = \text{outcome nuisance model } \mathbb{E}[Y|X],\; \hat{m}(X_i) = \text{propensity nuisance model } \mathbb{E}[T|X] \end{gathered}$$

```python
from doubleml import DoubleMLPLR, DoubleMLIRM
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

# Example: DML with Random Forest learner, PLR specification
dml_plr_rf = DoubleMLPLR(
    dml_data,
    ml_l=RandomForestRegressor(n_estimators=200),
    ml_m=RandomForestClassifier(n_estimators=200),
    n_folds=5
)
dml_plr_rf.fit()
```

![fig9_method_comparison.png](figures/fig9_method_comparison.png)
*Figure 7. Comparison of all causal effect estimates with 95% confidence intervals. Every method places the true effect below the naive correlation line.*

Five out of six specifications are significant at p < 0.001. The causal effect ranges from 0.51 to 2.72 percentage points depending on specification. Every single estimate falls below the naive 3.14 line, confirming that confounding bias is real and substantial.

An interesting pattern emerges from the results. Nonlinear learners like Random Forest and XGBoost consistently estimate larger effects than Lasso. This reveals that the relationship between confounders and outcomes is genuinely nonlinear. Simple linear adjustments underestimate the true causal effect.

## Who Benefits? The Heterogeneity Puzzle

This is where the analysis gets really interesting. We broke down the treatment effects by team size, time period, and domain.

![fig8_heterogeneous_effects.png](figures/fig8_heterogeneous_effects.png)
*Figure 8. Heterogeneous treatment effects across time periods, team sizes, and primary categories. The heatmap reveals where interdisciplinarity pays off the most.*

The team size pattern is the most provocative finding. Solo researchers and small teams of 2 to 3 people benefit the most from crossing boundaries. The effect peaks for small teams at roughly 1.5 to 2.5 percentage points. But for large teams of 7 or more, the causal effect drops to near zero or even turns negative.

Why? We believe the answer lies in what interdisciplinarity actually represents at different scales. For a small team, crossing a domain boundary means the researchers themselves are stretching their expertise, bringing genuinely new perspectives. For a large team, having members from different departments is nearly automatic. The boundary-crossing is structural rather than intellectual.

The temporal trend reveals something different. The interdisciplinary premium has been declining steadily from 2007 to 2024. As cross-domain work becomes the norm rather than the exception, its competitive advantage erodes. This finding carries a practical warning for DiscoveryAgent. Simply recommending "go interdisciplinary" is not enough. The agent needs to identify *where* and *how* boundary-crossing still pays off.

![fig5_citation_distribution.png](figures/fig5_citation_distribution.png)
*Figure 9. Citation distributions on a log scale, split by interdisciplinary status, alongside median citations for the top 15 categories.*

## Turning Findings into Tools: The RAG Causal Evidence System

Raw numbers in a table are useful for a paper but useless for an AI agent making real-time decisions. To bridge this gap, we built a Retrieval-Augmented Generation system that packages all our causal findings into a queryable knowledge base.

We structured our results into 25 evidence documents. Ten are general documents covering dataset statistics, TARNet results, DML results, heterogeneity patterns, and temporal trends. Fifteen are domain-specific documents, one for each of the top arXiv fields, containing field-level causal effects and interdisciplinary rates.

```python
# Encode evidence documents and build FAISS index
from sentence_transformers import SentenceTransformer
import faiss

encoder = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim
embeddings = encoder.encode(documents)
index = faiss.IndexFlatL2(384)
index.add(embeddings)

def retrieve_evidence(query, k=3):
    q_vec = encoder.encode([query])
    distances, indices = index.search(q_vec, k)
    return [(documents[i], distances[0][j]) for j, i in enumerate(indices[0])]
```

The tool is packaged as an MCP interface called `query_causal_evidence`. When our DiscoveryAgent wonders whether a particular cross-domain direction is worth pursuing, it can ask in natural language and receive causally grounded evidence rather than speculation.

![fig10_rag_retrieval_quality.png](figures/fig10_rag_retrieval_quality.png)
*Figure 10. RAG retrieval quality evaluation. The L2 distance histogram and query-document similarity heatmap confirm accurate semantic matching across all test queries.*

We validated retrieval quality across six test queries. The top-3 retrieved documents have L2 distances of 0.3 to 0.9, well separated from irrelevant documents at 0.6 to 1.8. The system correctly routes domain-specific questions to the right evidence documents and general methodology questions to the appropriate overview documents.

Here is a sample interaction. When we asked "I am in a 3-person team doing CS and Statistics cross-domain research. Does interdisciplinarity help us?", the system retrieved evidence on small team benefits, CS-specific effects, and the overall causal premium, synthesizing a grounded answer that small teams indeed benefit the most from boundary-crossing, with an estimated effect of 1 to 2.5 percentage points in these domains.

## What We Learned

Our causal analysis delivers a nuanced message. Interdisciplinary research does carry a real positive effect on citation impact. But the effect is roughly one-third smaller than naive correlations suggest. The benefit concentrates among small teams of 2 to 3 people and has been declining over recent years as cross-domain work becomes mainstream.

For DiscoveryAgent, these findings translate into actionable intelligence. The agent should not blindly recommend interdisciplinary work. Instead, it should consider team composition, field dynamics, and temporal trends before making a recommendation. The causal evidence tool we built ensures that every suggestion is backed by data, not just intuition.

As international students navigating a new academic system, we found something personally resonant in these numbers. The biggest gains from crossing boundaries come not from assembling large diverse teams but from individual researchers willing to stretch beyond their comfort zones. That is a lesson that extends well beyond arXiv statistics.

---

*Code and data available in our [GitHub repository]. All figures generated from main.ipynb.*

