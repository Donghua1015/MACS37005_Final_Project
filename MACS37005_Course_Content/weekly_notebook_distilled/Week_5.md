# Week 5 Distilled Reference: Sampling, Fine-Tuning, Benchmarking, and Tools
Course: MACS 37005 — Thinking with Deep Learning (UChicago, 2026)
Instructor: James Evans | TAs: Avi, Gio, Jesse, Shiyang
Modules completed: Sampling, FineTuning_LoRA, Tools_and_MCP (3 of 4)

---

## Core Techniques

Week 5 covers four interconnected areas. Module 1 covers probabilistic and non-probabilistic sampling — random, stratified, weighted, cluster, and systematic sampling on tabular data, plus graph sampling methods (PageRank-based, random walk, Metropolis-Hastings, snowball/forest-fire) via `littleballoffur`, and bootstrap resampling to construct confidence intervals around word embedding similarities. The key intuition is that sampling strategy determines what a model sees, and biased samples produce biased representations; bootstrap CIs quantify how much embedding geometry shifts with different training corpora.

Module 2 implements Low-Rank Adaptation (LoRA) and Quantized LoRA (QLoRA) for parameter-efficient fine-tuning. The intuition: weight updates during fine-tuning are low-rank in practice, so rather than updating all W, inject two trainable matrices A (d×r) and B (r×d) where r << d. QLoRA additionally quantizes the frozen base model to 4-bit NF4 (NormalFloat4) while keeping the LoRA adapters in FP16, enabling much larger models on limited GPU memory. Both are implemented via Hugging Face `peft` on `facebook/galactica-125m` trained on arXiv CS abstracts.

Module 3 introduces Centered Kernel Alignment (CKA) as a representation similarity metric invariant to orthogonal transformations and isotropic scaling. The intuition: compare the geometry of activation spaces across different models or layers without requiring alignment of individual neurons. Also introduces structured agent benchmarking frameworks (AgentBench, WebArena, SWE-Bench) and the problem of selection/cultural/temporal bias in benchmark design.

Module 4 implements tool-using agents and the Model Context Protocol (MCP). The intuition: LLM agents need standardized interfaces to external data and functions; MCP (Anthropic's open standard, JSON-RPC based) separates tool concerns from model concerns. Also implements DSPy for declarative, compiled prompt optimization, contrasting with manual prompt engineering.

---

## Key Code Patterns

**Stratified sampling (sklearn)**
```python
from sklearn.model_selection import train_test_split
sample, _ = train_test_split(df, train_size=0.10, stratify=df[['category_col']])
```
Proportional stratum representation in sample.

**Weighted sampling (pandas)**
```python
weighted_sample = df.sample(n=1000, weights='numeric_col')
```
Probability proportional to column value; mean of weight column will shift dramatically.

**Graph sampling (littleballoffur)**
```python
from littleballoffur import PageRankBasedSampler, RandomWalkSampler, MetropolisHastingsRandomWalkSampler, SnowBallSampler
sampler = MetropolisHastingsRandomWalkSampler(number_of_nodes=target_n)
subgraph = sampler.sample(graph)
```
MHRW corrects for degree bias in random walk; snowball for social network reachability.

**Bootstrap CI for embedding similarity**
```python
from sklearn.utils import resample
from gensim.models import Word2Vec
diffs = []
for _ in range(20):
    boot = resample(texts)
    m = Word2Vec(boot, vector_size=100, window=10)
    diffs.append(m.wv.similarity('word_a', 'word_b'))
sorted_diffs = sorted(diffs)
ci_lower, ci_upper = sorted_diffs[1], sorted_diffs[18]  # 90% CI from 20 samples
```
Quantifies stability of pairwise similarity under corpus resampling.

**Imbalanced classification (imbalanced-learn)**
```python
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
X_res, y_res = RandomUnderSampler().fit_resample(X_train, y_train)
# Only resample training set; test set must remain natural distribution
```

**LoRA application (peft)**
```python
from peft import LoraConfig, get_peft_model, TaskType
lora_config = LoraConfig(
    r=8, lora_alpha=32,
    target_modules=["k_proj", "q_proj", "v_proj", "o_proj"],
    lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()  # should show ~0.35% of total params
```

**QLoRA 4-bit quantization**
```python
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True, bnb_4bit_quant_type='nf4'
)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb_config, device_map="auto")
```
Base model loads in 955 MB instead of ~500 MB FP16.

**Linear CKA**
```python
def linear_CKA(X, Y):
    X = X - X.mean(axis=0); Y = Y - Y.mean(axis=0)
    K = X @ X.T; L = Y @ Y.T
    def hsic(K, L):
        n = K.shape[0]; H = np.eye(n) - np.ones((n, n)) / n
        return np.trace(K @ H @ L @ H) / (n - 1) ** 2
    return hsic(K, L) / np.sqrt(hsic(K, K) * hsic(L, L))
```
Returns scalar in [0, 1]; 1 = identical representation geometry.

**Tool registry pattern**
```python
@dataclass
class Tool:
    name: str; description: str; parameters: dict; function: Callable
    def to_schema(self): return {"type": "function", "function": {"name": self.name, ...}}
    def execute(self, **kwargs): return self.function(**kwargs)

class ToolRegistry:
    def register(self, tool): self.tools[tool.name] = tool
    def list_schemas(self): return [t.to_schema() for t in self.tools.values()]
    def execute(self, name, **kwargs): return self.tools[name].execute(**kwargs)
```

**OpenAI tool-use agent loop**
```python
for iteration in range(max_iterations):
    response = client.chat.completions.create(
        model="gpt-4o-mini", messages=messages, tools=registry.list_schemas(), tool_choice="auto"
    )
    msg = response.choices[0].message
    if not msg.tool_calls:
        return msg.content
    for tc in msg.tool_calls:
        result = registry.execute(tc.function.name, **json.loads(tc.function.arguments))
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})
```

**MCP server base class**
```python
class MCPServer(ABC):
    def __init__(self, name, version="1.0.0"): ...
    @abstractmethod
    def read_resource(self, uri: str) -> str: ...
    @abstractmethod
    def call_tool(self, name: str, arguments: dict) -> Any: ...
```

**DSPy RAG pipeline**
```python
import dspy
lm = dspy.LM('openai/gpt-4o-mini', api_key=api_key)
dspy.configure(lm=lm)

class AnswerResearchQuestion(dspy.Signature):
    context = dspy.InputField(desc="Retrieved papers context")
    stats = dspy.InputField(desc="Category statistics")
    question = dspy.InputField(desc="Research question")
    answer = dspy.OutputField(desc="Comprehensive answer")

class CoTRAG(dspy.Module):
    def __init__(self): self.answer = dspy.ChainOfThought(AnswerResearchQuestion)

from dspy.teleprompt import BootstrapFewShot
compiled = BootstrapFewShot(metric=metric, max_bootstrapped_demos=4).compile(CoTRAG(), trainset=trainset)
```

**SafeToolWrapper (input validation + rate limiting)**
```python
dangerous_patterns = [r'__[a-z]+__', r'import\s+', r'eval\s*\(', r'exec\s*\(']
for pattern in dangerous_patterns:
    if re.search(pattern, value, re.IGNORECASE): return False  # block
```

---

## Hyperparameters and Design Choices

**Word2Vec (Module 1)**
- vector_size=100, window=10, min_count=10, workers=1 (workers=1 for reproducibility), seed=42
- Bootstrap uses 20 iterations; 90% CI extracts sorted_diffs[1] and sorted_diffs[18]
- Sub-sampling into 20 non-overlapping chunks for alternative CI construction

**LoRA (Module 2)**
- Base model: facebook/galactica-125m (125M params, causal LM)
- Dataset: arXiv CS abstracts (10,000 papers; 8K/1K/1K split), max_length=128
- Best configuration identified: r=16, lora_alpha=32, lora_dropout=0.05, lr=5e-4
- Default baseline: r=8, lora_alpha=32, lora_dropout=0.05, lr=2e-4
- target_modules: ["k_proj", "q_proj", "v_proj", "o_proj"] (all attention projections)
- Batch size: 8 per device, 3 epochs, fp16=True, max_grad_norm=1.0

**LoRA rank sensitivity:**
- r=4: 221K trainable params, val PPL 17.60
- r=8: 442K params, val PPL 17.50 (good baseline)
- r=16: 885K params, val PPL 17.44 (marginal gain, double the params)
- Diminishing returns: r=4 to r=8 drops PPL by 0.10; r=8 to r=16 drops only 0.06

**Learning rate sensitivity (most important hyperparameter):**
- lr=5e-5: val PPL 17.74 (underfits)
- lr=2e-4: val PPL 17.50 (solid default)
- lr=5e-4: val PPL 17.48 (best)
- Very low sensitivity to lora_alpha (16 vs 32 vs 64 differ by <0.05 PPL)
- Very low sensitivity to dropout (0.0 vs 0.05 vs 0.1 differ by <0.02 PPL)

**QLoRA specifics:**
- bnb_4bit_quant_type='nf4' (NormalFloat4) outperforms uniform quantization for normally-distributed weights
- bnb_4bit_use_double_quant=True: quantizes the quantization constants themselves for additional memory saving
- bnb_4bit_compute_dtype=torch.float16: dequantizes to FP16 during forward pass

**Regularization lesson:**
- Weight decay 0.0 vs 0.01 vs 0.1: identical test PPL (17.34)
- LoRA adapter L2 norm decreases slightly with higher weight decay (11.529 → 11.255) but no performance effect
- Conclusion: explicit regularization is redundant for LoRA; the rank constraint itself regularizes the update

**Imbalanced dataset:**
- Imbalance ratio: 38:1 (majority:minority) for cs.CL detection on arXiv
- Baseline accuracy (majority class always): 97.43% — misleadingly high
- Undersampling: best for recall (97.3% minority recall); loses majority data
- Oversampling: intermediate (79.3% recall); inflates training set with duplicates
- Rule: never resample test data, only training data

---

## Results and Findings

**Module 1: Sampling**
- Bootstrap CI for Word2Vec similarity(book, stori) on hobbies corpus: 90% CI = (0.9959, 0.9999), CI width 0.004. Embeddings are extremely stable on this corpus.
- arXiv bootstrap (20k docs, 20 samples): similarity(neural, network) CI width = 0.088; similarity(quantum, comput) CI width = 0.067. More semantically distinct pairs are less stable.
- Weighted sampling by n_authors massively inflates the statistic: mean authors shifts from 4.67 (natural) to 63.19 (weighted), a 13x increase. Demonstrates how sampling weights control the effective population.
- Stratified sampling by category preserves all 171 categories vs only 129 in random sampling — critical for studying rare fields.
- Undersampling outperforms oversampling for minority recall on arXiv cs.CL (97.3% vs 79.3%), but oversampling achieves better majority precision (98.8% vs 95.6%). Trade-off is task-dependent.

**Module 2: LoRA/QLoRA**
- LoRA with r=8 reduces trainable parameters from 125M to 442K (0.35%) while reducing perplexity by 19.5% (21.76 → 17.52).
- QLoRA adds only +0.36 PPL vs full-precision LoRA (17.88 vs 17.52) while cutting base model memory from ~500 MB FP16 to 955 MB 4-bit — counterintuitive that 4-bit memory is larger than expected because bitsandbytes overhead.
- Training time: LoRA = 202.5s, QLoRA = 360.2s (78% slower due to dequantization overhead).
- Surprising: learning rate matters far more than rank or regularization. lr=5e-4 achieves the same PPL as lr=2e-4 with r=16 vs r=8.
- LoRA is an implicit regularizer: explicit weight decay has zero effect on test PPL. This is because the low-rank constraint already limits the magnitude of weight updates.

**Module 3: Benchmarking (template only)**
- Benchmark composition bias simulation: same agent scores 88% on English-heavy benchmark vs 75% on balanced benchmark — a 17% inflation from composition alone, not capability.
- CKA matrix between two randomly initialized networks of the same architecture: early layers are more similar (input processing is universal) while later layers diverge (task-specific).

**Module 4: Tool Use and MCP**
- Multi-tool chain successfully computes cs.LG = 9.2% of arXiv papers (search + stats + calculator).
- MCP find_convergence identifies cs.AI + cs.LG as strongest co-occurring category pair (299 papers).
- Tool safety: SafeToolWrapper blocked 4 of 5 injection patterns; one passed (insufficient coverage of all attack vectors).
- Agent failure mode: when querying a nonexistent category, agent hit max_iterations=5 by repeating similar queries — no error recovery or query reformulation. Demonstrates need for error-aware planning.
- DSPy compiled CoT achieved 0.838 combined score vs 0.875 for naive prompt on GPT-4o-mini; DSPy advantage grows larger on weaker models (llama2-13b: 9% → 47% per Khattab et al. 2023).
- BootstrapFewShot improved keyword retrieval score: 0.575 → 0.675. Critical debugging: forward() method signature must exactly match trainset field names or zero traces are collected.

---

## Connection to Chorus Project

Module 1's bootstrap embedding stability analysis directly informs Chorus's embedding pipeline: before trusting cosine similarity or angular distance as a measure of domain proximity, bootstrap CIs should be computed to verify that the embedding geometry is stable enough to support geometric analysis of converging research domains. Unstable similarities would make the "converging but underexplored" detection unreliable. Module 4's MCP architecture provides the right abstraction for Chorus's DiscoveryAgent: the ArxivResearchServer pattern (exposing search, category statistics, and find_convergence as MCP tools) maps precisely onto the Chorus use case where a RAG agent needs standardized access to the publication graph — the find_convergence tool in Exercise 2 directly parallels the cross-domain proximity analysis at the core of the Chorus enhancement. Module 2's perplexity metric is directly relevant as a surprise/novelty score: papers in underexplored intersections should have higher perplexity under a domain-specific language model, consistent with the Zhang & Evans 2025 perplexity-based surprise metric; fine-tuning a domain-specific model via LoRA and measuring cross-domain perplexity offers a concrete implementation path.

---

## Potential Final Project Integration

Module 2 (LoRA/QLoRA fine-tuning) is a strong candidate for one of the "three model types from three separate weeks." The implementation is complete and reusable: fine-tune facebook/galactica-125m (or a larger model via QLoRA) on KnowledgeLab publication abstracts to build a domain-specific language model, then use its perplexity as a surprise metric for cross-domain paper recommendations. A paper scoring high perplexity under the domain LM but having semantic embedding proximity to the domain centroid would be exactly the "converging but underexplored" signal the Chorus enhancement targets. The code pattern is: LoraConfig + get_peft_model + TrainingArguments + Trainer.evaluate() → exp(eval_loss) = domain perplexity score for any candidate paper. This cleanly separates as Week 5's contribution distinct from the embedding geometry component (which draws from Kozlowski et al. 2019) and any agent/retrieval component from other weeks.

Module 1 (bootstrap sampling + Word2Vec stability) is a secondary candidate: it provides an evaluation methodology — use bootstrap CIs to validate that the embedding-based domain proximity scores are stable, not artifacts of corpus sampling. This strengthens the scientific credibility of the Chorus system's recommendations without requiring it to be the primary modeling contribution.

Module 4 (tool use / MCP) is the strongest architectural candidate for the DiscoveryAgent implementation layer itself: the Tool + ToolRegistry + agent loop pattern, and especially the find_convergence MCP tool design, provides a complete blueprint for building a Chorus DiscoveryAgent that calls external scholarly APIs (Semantic Scholar, CrossRef), computes co-occurrence statistics, and chains geometric + perplexity analyses within a single agentic loop using gpt-4o-mini with tool_choice="auto."
