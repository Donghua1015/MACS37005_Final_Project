# Week 3 Reference: Programming LLMs, Multi-Agent Simulation, AutoGen, Concordia, RAG

Source notebook: Week_3.ipynb (5 modules, 208 code cells, 169 markdown cells)
Mandatory: Modules 1, 2, 5. Choose one of: Module 3, Module 4.

---

## MODULE 1: Programming LLMs with Language and Prompting

### Core Techniques and Intuition

Module 1 covers the full prompting toolkit for OpenAI-compatible models. The central intuition is that LLM behavior is programmable through natural language: the same base model can be made to act as a classifier, a creative writer, a reasoner, or a domain expert purely through prompt design. The module benchmarks six prompting strategies on an arXiv paper classification task (100 papers, 5 categories: cs, math, physics, astro-ph, cond-mat) using gpt-3.5-turbo.

Zero-shot prompting: send a task description and input, accept output directly. Sufficient for many tasks with clear categories.

Few-shot prompting: provide k labeled examples before the test input. The model learns the format and label distribution from demonstrations. Effective when ground-truth examples are cheap to obtain.

Interactive multi-shot: build a rolling context string, appending question-answer pairs so the model conditions on prior exchanges. Useful for multi-turn disambiguation.

Chain of Thought (CoT, Wei et al. 2022): include intermediate reasoning steps in few-shot examples so the model decomposes the problem before answering. The zero-shot variant appends "Let's think step by step" to trigger decomposition without examples.

Actor-Critic pipeline: one LLM call (Actor) produces a classification with reasoning; a second call (Critic) evaluates the decision and suggests corrections; a third call (Refined Actor) incorporates the critique. Designed to replicate human self-monitoring but adds latency and can degrade accuracy.

### Key Code Patterns

```python
# Basic OpenAI chat call (Module 1 template)
from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)
response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
text = response.choices[0].message.content

# System persona + user message (few-shot setup)
messages = [
    {"role": "system", "content": "You are my breakfast assistant..."},
    {"role": "user", "content": query}
]

# LangChain few-shot (gpt-3.5-turbo-instruct, completion API)
from langchain_openai import OpenAI
model = OpenAI(model="gpt-3.5-turbo-instruct")
story = model.generate(prompts=[few_shot_prompt])

# Zero-shot classifier function signature
def classify_paper_zeroshot(abstract, categories, client, model="gpt-3.5-turbo", temperature=0): ...

# Actor-Critic pipeline function signature
def actor_critic_classify(abstract, categories, client, verbose=False): ...
# Returns: {"initial_prediction": str, "critique": str, "final_prediction": str}
```

### Hyperparameters and Design Choices

temperature=0 used throughout for classification (deterministic). Tested temperature range [0, 0.3, 0.7, 1.0] for sensitivity analysis. Shot count varied as k in {0, 1, 3, 5}. A single example (1-shot) sometimes introduced bias rather than signal, causing worse performance than 0-shot. The Actor-Critic pipeline used 3 sequential API calls per sample; overhead did not translate to accuracy gains. CoT prompt structure: identify key concepts → analyze field membership → classify.

### Results and Findings

Classification accuracy on 100 arXiv papers, 5 categories:

- 5-shot: 78% (best)
- 0-shot: 76%
- 3-shot: 76%
- CoT: 76% (no improvement over 0-shot)
- 1-shot: 74% (worse than 0-shot — single example introduced bias)
- Actor-Critic: 58% (worst — 18% drop from 0-shot)

Category-level insight: physics papers had 0% recall (all misclassified as cs), while astro-ph, math, and cond-mat each reached 95% recall. The model treated physics and cs as nearly synonymous. Actor-Critic analysis: the Critic changed 38% of predictions (19/50 samples); of these, 11 broke correct predictions, 6 changed wrong to different wrong, and only 2 fixed genuine errors. This shows Critic interference is destructive in low-ambiguity classification tasks. Counterintuitive finding: more complexity (Actor-Critic, CoT) did not improve over simple zero-shot; few-shot with real examples was the only reliable gain.

### Connection to Chorus Project

The Actor-Critic and few-shot patterns from Module 1 directly inform DiscoveryAgent and MatchmakerAgent design in Chorus. A DiscoveryAgent could use a Critic pass to sanity-check candidate "sweet-spot" domain pairs against known literature before surfacing them. The lesson that Critic interference hurts classification suggests building the Critic as a filter (veto gate) rather than a rewriter in Chorus. Few-shot prompting with real validated research combinations as examples could improve MatchmakerAgent relevance ranking.

### Potential Final Project Integration

Module 1's prompting techniques qualify as one of the three required model types. Specifically, the few-shot or CoT prompting pipeline used for paper classification can be repurposed as the Chorus recommendation explanation layer: given a candidate domain pair identified by the geometric/perplexity analysis, a CoT-prompted LLM generates a structured justification explaining why the combination is converging but underexplored. This makes the system's output interpretable to researchers.

---

## MODULE 2: Multi-Agent Social Simulation with LLMs

### Core Techniques and Intuition

Module 2 implements LLM-based "digital doubles" — agents with demographic and ideological backgrounds injected via system prompts — to simulate human survey responses and social interactions. The theoretical basis is Argyle et al. (2024) "Out of One, Many" (Political Analysis), which validates LLM agents against two criteria: (1) Social Science Turing Test: generated responses are indistinguishable from parallel human texts; (2) Backward Conditioning Test: conditioning on known demographics reliably shifts LLM outputs in directions matching real group behavior.

The module replicates two Argyle et al. analyses: free-form partisan text generation (Rothschild et al. "Pigeonholing Partisans" survey data, n=2,112) and vote prediction using ANES 2012/2016/2020. Demographic variables (ideology, party ID, race, gender, income, age) are serialized into a natural-language prompt that serves as the agent's identity.

The Homework extends this to domain expert simulation and a Consolidating vs. Disrupting agent debate directly relevant to the Chorus project.

### Key Code Patterns

```python
# text_generate wrapper (Module 2 standard, gpt-4o-mini)
def text_generate(messages):
    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    return response.choices[0].message.content

# Republican/Democrat agent definition
republican_agent = [
    {"role": "system", "content": "Assume that you are a Republican. Now, you are engaging in a virtual political discourse."},
    {"role": "user", "content": "Initiate the debate..."}
]

# Multi-round discussion between two named agents
def simulate_agent_discussion(topic, agent1, agent2, num_rounds=5, client=None):
    # Returns list of {"round": int, "speaker": str, "content": str}

# Domain expert judgment simulation
def simulate_expert_judgment(abstract, expert_domain, expert_description, client):
    # Returns {"judgment": "Yes"/"No", "confidence": int, "reasoning": str}

# LLM-as-evaluator for scoring open-ended responses
def evaluate_response(direction, persona, response, client):
    # Returns {"relevance": int, "expertise": int, "coherence": int, "insight": int}
```

### Hyperparameters and Design Choices

Sample size for initial experiments: n=20 (from full dataset). Vote prediction: 20 individuals from ANES 2020. Expert simulation: 4 domains (CS, Physics, Math, Statistics) judging all papers (binary Yes/No). Consolidating vs. Disrupting discussion: 5 rounds. LLM evaluator scores on 1-5 scale across 4 dimensions (relevance, expertise, coherence, insight). For local alternative to GPT: DeepSeek-R1-Distill-Qwen-7B loaded via transformers with AutoModelForCausalLM and AutoTokenizer, run on CUDA (A100 recommended in Colab).

### Results and Findings

Vote prediction (n=20 ANES 2020 sample): F1 = 0.952. The model correctly predicted voting behaviors even for edge cases.

Domain expert simulation (Module 2 Homework Task 1): Overall accuracy 61.88%, F1 = 0.548. High recall (93%) but low precision (39%): simulated experts correctly found their own domain papers but also claimed cross-domain papers. This mirrors real academic behavior where researchers see relevance everywhere.

Open-ended response simulation (Task 2): LLM peer review scored all responses highly — Relevance: 5.0, Expertise: 5.0, Coherence: 4.0, Insight: 4.0. Each persona (ML researcher, Social Scientist, Philosopher) maintained consistent epistemic framing, demonstrating that system-prompt identity injection reliably constrains output style and vocabulary.

Consolidating vs. Disrupting agent debate (Task 3, topic: using perplexity metrics to identify transformative research): Consolidator showed higher sentiment (4.6/5), openness (4.6), constructiveness (5.0); Disruptor maintained higher conviction (5.0) throughout. The agents debated integration vs. isolation of radical ideas, with the Consolidator eventually incorporating quantitative framing while the Disruptor pushed for more extreme departures from existing paradigms. LLM summary identified this as a productive tension that moved both agents toward richer positions.

Counterintuitive: the Consolidating agent was more open and emotionally positive than the Disrupting agent, which matched aggressive, defensive conviction scores. This inversion of expected temperament is a known LLM simulation artifact.

Partisan word generation: LLM-generated and human-generated words for Democrat and Republican showed high overlap (e.g., "liberal", "conservative") but LLM outputs were more neutral and less emotionally loaded than human survey responses.

### Connection to Chorus Project

Module 2's multi-agent simulation framework is directly applicable to Chorus as a validation layer. A simulated debate between Consolidating and Disrupting researcher agents about a candidate domain combination can serve as a qualitative stress test before the system surfaces it to real users. The Homework Task 3 simulation — which explicitly modeled this debate about perplexity metrics for research discovery — demonstrated that role-differentiated agents can generate complementary critiques that neither agent alone would produce, providing richer evaluation signals than single-agent LLM review.

### Potential Final Project Integration

The multi-agent social simulation approach qualifies as one of the three required model types. Specifically, the Consolidating vs. Disrupting agent debate pipeline (simulate_agent_discussion) can serve as a post-hoc validation module in Chorus: after the geometric/perplexity analysis identifies candidate domain combinations, a two-agent debate evaluates feasibility and novelty. Quantitative analysis of sentiment and conviction trajectories across debate rounds could function as a quality score for shortlisting recommendations to present to real researchers.

---

## MODULE 3: Multi-Agent Conversational Simulation with AutoGen

### Core Techniques and Intuition

AutoGen (library: ag2, formerly autogen) provides a declarative framework for multi-agent conversation where each agent is a ConversableAgent or AssistantAgent with a system prompt and an LLM backend. Agents exchange messages until a termination condition fires (e.g., the message content contains "TERMINATE"). The key intuition is that complex collaborative tasks can be decomposed into sequential agent handoffs rather than monolithic prompts.

Four patterns demonstrated: (1) pairwise chat (initiate_chat between two agents, max_turns limit); (2) sequential pipeline (initiate_chats list with sender/recipient/message triplets for multi-step onboarding); (3) critic-writer loop (AssistantAgent writer + Critic that terminates on "TERMINATE"); (4) research collaboration digital doubles (two AssistantAgents representing real sociologists — Austin C. Kozlowski and Donghyun Kang — discussing a new research project).

### Key Code Patterns

```python
from autogen import ConversableAgent, AssistantAgent, initiate_chats

config_list = [{"model": "gpt-4o-mini", "api_key": OPENAI_API_KEY}]
llm_config = {"config_list": config_list}

# Pairwise chat
agent_a = ConversableAgent(name="cathy", system_message="...", llm_config=llm_config,
                            human_input_mode="NEVER")
chat_result = agent_b.initiate_chat(recipient=agent_a, message="...", max_turns=10)

# Sequential onboarding pipeline
chats = [{"sender": agent1, "recipient": proxy, "message": "...", "max_turns": 2,
           "summary_method": "last_msg"},
         {"sender": agent2, "recipient": proxy, "message": "...", "max_turns": 2}]
chat_results = initiate_chats(chats)

# Critic terminates on keyword
critic = AssistantAgent(name="Critic",
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    system_message="Review the content. Respond with TERMINATE if satisfied.")
```

### Hyperparameters and Design Choices

model: gpt-4o-mini throughout (cheap, effective). max_turns controls conversation depth; 10 for free conversation, 2 for structured onboarding steps. human_input_mode="NEVER" for fully automated runs. summary_method="last_msg" for chaining context between sequential steps. Termination via string matching on "TERMINATE" is fragile; more robust alternatives would use structured output or a dedicated termination agent.

### Results and Findings

Qualitative only. Comedian pairwise chat: agents maintained distinct comic voices and built on each other's jokes. Customer onboarding sequential pipeline: each agent correctly handled its specialized step (demographics → interests → product recommendation) without leaking context inappropriately. Writer-Critic loop: blog post improved through feedback rounds, with Critic providing specific, actionable suggestions rather than vague praise. Research collaboration digital doubles of Kozlowski and Kang: agents referenced each other's actual published work (embedding geometry, cultural dimensions) and synthesized a plausible novel research direction, demonstrating that system-prompt biographical injection can produce coherent disciplinary voices.

### Connection to Chorus Project

AutoGen's sequential pipeline and critic-writer loop map directly onto Chorus's multi-agent architecture. The initiate_chats pattern could orchestrate the pipeline: EmbeddingAgent → DiscoveryAgent → MatchmakerAgent → ExplanationAgent as a sequential chain. The ConversableAgent with "NEVER" human input enables fully automated batch processing of research domain pairs. The digital double of Austin C. Kozlowski is directly relevant since Kozlowski et al. (2019) is a foundational citation in my project.

### Potential Final Project Integration

AutoGen multi-agent orchestration can serve as the architectural backbone of the Chorus enhancement rather than as a standalone model type. It is most useful as infrastructure that wires together the three required model types (e.g., embedding model, RAG, simulation). If the project requires demonstrating an agentic pipeline, AutoGen's initiate_chats provides a clean way to present the system as a chain of specialized agents rather than a monolithic script.

---

## MODULE 4: Situated Multi-Agent Simulation with Concordia

### Core Techniques and Intuition

Concordia (library: gdm-concordia, from DeepMind) adds three layers that AutoGen lacks: time (explicit clock with date/hour), place (location constraints that determine who can interact with whom), and formative memory (agents have episodic memories injected at initialization representing their biography). The key intuition is that social behavior is inseparable from context: the same agent will act differently depending on where they are, what time it is, and what they remember about past events.

Concordia requires two components: a model (anything implementing sample_text) and an embedder (for memory retrieval). The framework uses prefabs (predefined agent and game master configurations): conversational__Entity for agents and situated_in_time_and_place__GameMaster for the environment. Dramaturgic scenes separate conversation (open dialogue) from decision (fixed action choices evaluated by a scoring function). Reflective agents add an explicit self-monitoring step before each action.

### Key Code Patterns

```python
# Install
# pip install gdm-concordia sentence-transformers openai

from concordia.prefabs import entity as entity_prefabs
from concordia.prefabs import game_master as game_master_prefabs
from concordia.prefabs.simulation import generic as simulation
from concordia.typing import prefab as prefab_lib, scene as scene_lib

# Custom OpenAI wrapper for Concordia
class SimpleOpenAIModel:
    def __init__(self, model_name="gpt-4o-mini", api_key=None):
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
    def sample_text(self, prompt, max_tokens=1000, temperature=0.7): ...

# Agent config
agents = [prefab_lib.InstanceConfig(
    prefab="conversational__Entity", role=prefab_lib.Role.ENTITY,
    params={"name": "Mina", "goal": "...", "formative_memories": [...]})]

# Game master config
world_gm = prefab_lib.InstanceConfig(
    prefab="situated_in_time_and_place__GameMaster",
    role=prefab_lib.Role.GAME_MASTER,
    params={"name": "World", "locations": [...], "start_time": "..."})

# Dramaturgic scenes
conversation_scene = scene_lib.SceneTypeSpec(name="conversation", game_master="World")
decision_scene = scene_lib.SceneTypeSpec(name="decision", game_master="World",
    action_spec=entity_lib.choice_action_spec(options=["Option A", "Option B", "Option C"]))

# Scoring function (shapes collective outcomes)
def action_to_scores(joint_action):
    choice = joint_action.get("Mina", "")
    if "Validity-first" in choice: return {"Mina": 3, "Eli": 2, "Noor": 2}
    elif "Scale-first" in choice: return {"Mina": 1, "Eli": 1, "Noor": 1}
    return {"Mina": 2, "Eli": 1, "Noor": 1}
```

### Hyperparameters and Design Choices

Model: gpt-4o-mini (wrapped as SimpleOpenAIModel). Formative memory injection: ages 5, 10, 15-18, 21-23, 28-30 plus shared event memories. Number of conversation rounds before decision: 4 (reduced from larger values to avoid boundary errors). Three scoring function variants tested to show how reward structure shapes collective behavior. Reflective agent designation: only one agent per simulation made reflective to enable clean comparison.

### Results and Findings

Dramaturgic scenes with three scoring functions: (1) balanced scoring → Mina chose Theory-first (scores 2/1/1); (2) Scale-first rewarded → Mina still chose Theory-first (1/1/1); (3) Validity-first rewarded → Mina switched to Validity-first. This shows agents are sensitive to payoff structures but the effect is non-linear: incentives must clearly dominate before agents change strategy.

Reflective vs. non-reflective: reflective agent generated explicit structured internal monologue — observed facts, trust risks, guardrails, next move intent — before acting. Non-reflective agent acted directly on goal and observations. Expected behavioral difference: reflective agents show more consistent long-term strategy, less susceptibility to social pressure in early rounds.

Location constraints in Concordia: agents in different parts of the environment (library vs. lab) do not interact directly, creating natural information asymmetries that produce emergent sub-group consensus before broader negotiation.

Formative memory retrieval: at decision points, agents recalled childhood and career memories that shaped their stance. This produced plausible but sometimes stereotyped behavior (a limitation flagged in homework).

Concordia Homework Task (directly Chorus-relevant): reformulated the Chorus research question as: "How do researchers with different disciplinary backgrounds evaluate cross-disciplinary research directions identified as sweet spots?" Four agents: Dr. Consolidator (Senior ML, skeptical), Dr. Disruptor (Computational Social Scientist, advocates unconventional), Dr. Validator (Methodologist, focuses feasibility), Dr. Integrator (Interdisciplinary, facilitates synthesis). Environment: research institute with shared journal collection, citation network as memory. Key simulation insight: could reveal deliberation processes and consensus-building dynamics that pure quantitative metrics (distance-surprise matrix) cannot capture.

### Connection to Chorus Project

Concordia's situated simulation is the richest validation tool available for the Chorus project. A simulation placing researcher agents in an environment where they share Chorus recommendations and deliberate about which to pursue would reveal social dynamics that the embedding space analysis cannot detect: how interdisciplinary teams navigate risk tolerance, how memory of past failed cross-domain bets affects receptivity to new recommendations. The dramaturgic decision phase with scoring functions directly models the "will researchers act on this recommendation?" question that Chorus ultimately needs to answer.

### Potential Final Project Integration

Concordia multi-agent simulation qualifies as one of the three required model types if structured as an evaluation module. After the embedding + perplexity pipeline identifies candidate domain combinations, a Concordia simulation with researcher digital doubles evaluates which combinations survive deliberation under realistic social constraints. The scoring function can be calibrated against historical data on which cross-domain combinations actually resulted in publications, making this a semi-quantitative validation layer.

---

## MODULE 5: Retrieval-Augmented Generation (RAG)

### Core Techniques and Intuition

RAG combines dense retrieval with generative LLMs to ground responses in a specific document collection. The core intuition is that LLMs have strong reasoning capabilities but unreliable factual recall for specific or recent content; retrieval supplies the facts, generation supplies the reasoning. The pipeline has three stages: (1) offline indexing — embed all documents with a sentence-transformer, store vectors in a FAISS index; (2) online retrieval — embed the query, find top-k nearest neighbors by L2 distance; (3) generation — concatenate retrieved documents as context in the LLM prompt.

Module 5 implements this from scratch using sentence-transformers (all-MiniLM-L6-v2, 384-dimensional embeddings), faiss-cpu (IndexFlatL2), and openai (gpt-3.5-turbo). The Homework scales this to 200 arXiv ML papers and systematically varies k-value and chunking strategy.

### Key Code Patterns

```python
# Install
# pip install sentence-transformers faiss-cpu openai numpy

from sentence_transformers import SentenceTransformer
import faiss, numpy as np
from openai import OpenAI

# Offline: embed + index
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
document_embeddings = embedding_model.encode(documents, convert_to_numpy=True)
# document_embeddings.shape = (n_docs, 384)

dimension = document_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(document_embeddings)

# Online: retrieve
def retrieve_documents(query, k=3):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)
    return [documents[i] for i in indices[0]]

# RAG generation
def rag_generate(query, k=3, model="gpt-3.5-turbo"):
    retrieved = retrieve_documents(query, k)
    context = "\n\n".join(retrieved)
    messages = [
        {"role": "system", "content": f"Use this context:\n{context}"},
        {"role": "user", "content": query}
    ]
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content, retrieved

# Chunking strategies (Task 3)
chunks_para = docs_200                          # full abstract per chunk
chunks_sent = [s for doc in docs_200 for s in doc.split(". ")]   # sentence-level
chunks_fixed = [doc[i:i+400] for doc in docs_200 for i in range(0, len(doc), 400)]  # 400-char windows

# Helper for k-value sweep
def retrieve_k(query, k, idx, docs):
    query_emb = embedding_model.encode([query], convert_to_numpy=True)
    _, indices = idx.search(query_emb, k)
    return [docs[i] for i in indices[0]]
```

### Hyperparameters and Design Choices

Embedding model: all-MiniLM-L6-v2, 384-dimensional. FAISS index type: IndexFlatL2 (exact L2 search, appropriate for small corpora; for large scale would use IndexIVFFlat or HNSW). k values tested: 1, 3, 5, 10, 20. Chunking strategies: paragraph (full abstract per chunk), sentence-level (split on ". "), fixed-400 (400-character windows). Document collection sizes: 10 papers (Task 1-2), 200 papers (Task 3). Generation model: gpt-3.5-turbo. LLM evaluator for scoring: prompts another GPT call to rate accuracy (1-5), helpfulness (1-5), hallucination (1-3, lower is better).

### Results and Findings

Task 2 — RAG vs. vanilla LLM on 10 arXiv papers:
- Vanilla LLM accuracy: 4.80/5.00; RAG accuracy: 4.00/5.00
- Vanilla LLM relevance: 5.00/5.00; RAG relevance: 4.20/5.00
- RAG hallucination: 2.20/3.00 (less hallucination); Vanilla hallucination: 1.20/3.00 (more hallucination)
- RAG outperformed on Q1 (Artificial Immune Systems in film recommendation — obscure, document-specific topic)

Task 3 — k-value sweep on 200 papers:
- k=1: 4.2 average accuracy (insufficient context)
- k=3, 5, 10: 5.0 (optimal)
- k=20: 4.6 (slight degradation from noise)
- Optimal k range: 3-10

Task 3 — chunking strategies on 200 papers:
- Paragraph (200 chunks): 4.0 accuracy (lowest, but fastest)
- Sentence-level (1313 chunks): highest accuracy
- Fixed-400: mixed results, moderate performance

Key lessons: RAG does not universally outperform vanilla LLM. On common ML topics, LLM parametric knowledge is sufficient and RAG adds noise. RAG's primary advantage is hallucination suppression and performance on document-specific queries. Document coverage is the binding constraint: too few documents means many queries find no relevant content. RAG honestly reports "I don't have information about this" when retrieved context is irrelevant, which is a feature for scientific applications.

### Connection to Chorus Project

Module 5's RAG pipeline is the direct technical foundation of Chorus itself. The specific improvement I am implementing — proactive recommendation of "converging but underexplored" domain combinations — would extend the existing RAG system by adding a geometric analysis layer before retrieval: rather than retrieving documents similar to the user's query, Chorus would also proactively push documents from domains that are geometrically converging in embedding space but statistically underrepresented in the user's recent queries. The FAISS IndexFlatL2 pattern is directly reusable for the embedding space distance matrix computation that underlies my surprise metric.

### Potential Final Project Integration

Module 5's RAG implementation is almost certainly one of the three required model types, as it is the most central to the Chorus project. The from-scratch implementation pattern (SentenceTransformer + FAISS + OpenAI) can be extended by: (1) replacing the static document collection with Chorus's live knowledge base; (2) adding a proactive retrieval layer that identifies documents from underexplored domain intersections using geometric analysis; (3) using the LLM evaluator pattern (evaluate_response) as a quality gate for generated recommendations. The k-value and chunking experiments provide empirical guidance: use k=3-10, prefer sentence-level chunking for precision, and expect RAG to underperform vanilla LLM on general knowledge questions while excelling on corpus-specific ones.

---

## Cross-Module Summary for Final Project

Three candidate model types for the required "three from three separate weeks" requirement:

Week 3, Module 1 (prompting / LLM): CoT or few-shot LLM call for recommendation explanation — takes a candidate domain pair and generates a structured justification. Low computational cost, easy to deploy.

Week 3, Module 2 or 4 (multi-agent simulation): Consolidating vs. Disrupting agent debate as post-hoc validation of candidate recommendations. Provides qualitative stress-testing and can be run asynchronously.

Week 3, Module 5 (RAG): Core retrieval system, extended with geometric proactive retrieval layer. This is the primary deliverable for the Chorus enhancement.

The most critical cross-module insight: the LLM evaluator pattern from Module 2 (LLM-as-judge scoring responses 1-5) reappears in Module 5 (evaluate_response for RAG quality). This same pattern can serve as the automatic evaluation framework for the Chorus enhancement — generate candidate recommendations, evaluate with an LLM judge, filter by score, surface top-k to the user.
