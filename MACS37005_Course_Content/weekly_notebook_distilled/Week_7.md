# Week 7 Reference: Reinforcement Learning to Optimize AI Agents

## Core Techniques

Week 7 covers four RL tiers, each building on the last.

Module 1 establishes Deep RL Foundations. The central abstraction is the Markov Decision Process (MDP): an agent observes a state, selects an action according to a policy, receives a reward, and transitions to a new state. The return G_t = sum of discounted future rewards (discount factor gamma) is what policies try to maximize. The notebook progresses through five algorithms: Q-Learning (tabular, off-policy temporal difference), REINFORCE (Monte Carlo policy gradients), DQN (Q-Learning with a neural network approximator and experience replay to break sample correlation), Actor-Critic (a policy network + a value network in parallel), and PPO (Proximal Policy Optimization, which clips the policy update ratio to the range [1-epsilon, 1+epsilon] to prevent destabilizing large steps). The Bandit problem (single-state RL) is the entry point, with three exploration strategies: Epsilon-Greedy (random with probability epsilon), UCB (upper confidence bound adds a bonus for uncertainty), and Thompson Sampling (Bayesian posterior over expected reward).

Module 2 covers RL for Alignment. A reward model is trained from human preference pairs using the Bradley-Terry objective: given a chosen and a rejected response, the RM learns to score chosen higher. The RM is then used to supervise PPO fine-tuning of an LLM. Constitutional AI (RLAIF) replaces human labels with AI self-critique against written principles. RATE (Rewrite-based Attribute Treatment Estimators) applies causal inference logic — analogous to difference-in-differences — to estimate the Average Treatment Effect of specific text attributes on RM scores by generating attribute-added and attribute-removed rewrites and comparing scores.

Module 3 introduces GRPO (Group Relative Policy Optimization) and the Societies of Thought (SoT) framework. GRPO is a critic-free variant of PPO: for each prompt, sample G completions, score them with a reward function, compute per-completion advantage as (r_i - mean(r)) / std(r), and apply the PPO clipped objective. No value network is needed. DeepSeek-R1 used GRPO on a 671B model to elicit emergent chain-of-thought reasoning — self-verification, error correction, and "aha moment" trace segments — without any supervised reasoning examples. Kim et al. (2025) propose the SoT framework: reasoning models simulate internal communities of perspectives (self-questioning, self-correction, verification, perspective-shift, conflict, reconciliation) rather than simply outputting longer text.

Module 4 covers Multi-Agent RL. The Iterated Prisoner's Dilemma (IPD) operationalizes the cooperation-vs-defection tension: individual reward gradients toward defection while team reward induces cooperation. Theory of Mind (Cross et al. 2025 Hypothetical Minds) maintains a hypothesis set over opponent strategies, updating with a Rescorla-Wagner rule, allowing an agent to adapt to opponent strategy switches. Counterfactual credit assignment (Bo et al. 2024) defines each agent's credit as the team reward minus the team reward it would have produced if that agent were absent — a Shapley-value-inspired approach that sharpens individual incentives.

---

## Key Code Patterns

Q-table discretization for continuous state spaces:
`discretize(state, bounds, bins=20)` — maps each continuous dimension to an integer index, returns a tuple as the Q-table key. Needed whenever applying tabular RL to gym environments with Box observation spaces.

Q-Learning update (one line):
`q_table[state + (action,)] += lr * (reward + gamma * max_future_q - q_table[state + (action,)])`

DQN with experience replay — the three-part structure is the reusable pattern:
1. `class DQN(nn.Module)` with `nn.Sequential(Linear, ReLU, Linear, ReLU, Linear)` and a forward pass
2. `class ReplayBuffer` with `deque(maxlen=capacity)`, `.push()` and `.sample(batch_size)` methods
3. Training loop: epsilon-greedy action → push to buffer → sample batch → compute `current_q = q_net(states).gather(1, actions.unsqueeze(1)).squeeze()` → compute `target_q = rewards + gamma * target_net(next_states).max(1)[0] * (1 - dones)` → `F.mse_loss(current_q, target_q)` → backprop → periodic `target_net.load_state_dict(q_net.state_dict())`

PPO via stable-baselines3 (minimal invocation):
```python
model = PPO("MlpPolicy", "CartPole-v1", n_steps=256, batch_size=64,
            n_epochs=10, learning_rate=3e-4, gae_lambda=0.98, gamma=0.99,
            ent_coef=0.0, verbose=1)
model.learn(total_timesteps=25600)
```

Reward model training via TRL:
`from trl import RewardTrainer, RewardConfig` — wraps `AutoModelForSequenceClassification` with `num_labels=1`; dataset must have `chosen` and `rejected` columns; `RewardConfig` mirrors TrainingArguments.

GRPO training via TRL:
`from trl import GRPOTrainer` on `Qwen/Qwen2.5-0.5B-Instruct`, dataset from `AI-MO/NuminaMath-TIR`; reward_functions list: accuracy checker (exact match) + format checker (presence of `<think>` tag).

SoT behavior pattern classifier (regex dict, reusable as-is):
```python
behavior_patterns = {
    "self_questioning": r"(let me think|wait|hmm|let's see|consider)",
    "self_correction": r"(actually|no,? wait|that's wrong|let me reconsider)",
    "verification":    r"(let me check|verify|double.check|to confirm)",
    "perspective_shift": r"(but what if|alternatively|on the other hand|another way)",
    "conflict":        r"(however|contradicts|doesn't match|conflicts|incompatible)",
    "reconciliation":  r"(therefore|so combining|the answer is|in conclusion|thus)"
}
```

IPD Q-learning with configurable reward type:
`class QLearningAgent` with `defaultdict(lambda: np.zeros(2))` Q-table, `act(state)` using epsilon-greedy, `update(state, action, reward, next_state)` with TD-0 update. State is `(my_last_action, opponent_last_action)`.

Counterfactual credit:
`Credit_i = R_team - R_team_without_i` — remove agent i from the action vector, recompute team reward, difference is individual contribution.

---

## Hyperparameters and Design Choices

Q-Learning on Mountain Car: `EPISODES=10000`, `LEARNING_RATE=0.5`, `DISCOUNT=0.99`, `EPSILON_START=1.0`, `EPSILON_END=0.01`, `EPSILON_DECAY=2000` steps. `bins=20` for state discretization. High learning rate (0.5) works here because the tabular update is low-variance; continuous-valued Q-networks require much smaller lr.

DQN on CartPole: `gamma=0.99`, `epsilon` decayed multiplicatively by `0.995` per episode down to `0.01`, `batch_size=64`, `replay capacity=10000`, `target_update_freq=10` episodes, `hidden=128`, `lr=1e-3` with Adam. Key design choice: the target network freeze prevents oscillation; without it CartPole DQN diverges.

PPO via stable-baselines3: `n_steps=256`, `batch_size=64`, `n_epochs=10`, `learning_rate=3e-4`, `gae_lambda=0.98`, `gamma=0.99`, `ent_coef=0.0`. Achieves mean reward ~350+ on CartPole by iteration 200 starting from ~21. The clipping epsilon default is 0.2 in SB3.

Reward model training (TRL RewardConfig): `num_train_epochs=3`, `learning_rate=1e-4`, `per_device_train_batch_size=4`, `gradient_accumulation_steps=4`, `warmup_ratio=0.1`, `eval_steps=500`, `max_length=512`. Model: `Qwen/Qwen2.5-0.5B-Instruct` as sequence classifier with `num_labels=1`.

GRPO lesson learned (critical): a 0.5B model is too small for GRPO to produce emergent reasoning from scratch. The model collapsed into mode collapse after ~50 steps (repetitive token output, all rewards → 0, advantage → 0, gradient → 0). Gandhi et al. (2025) corroborate: RL self-improvement requires the model to already exhibit baseline cognitive behaviors (verification, backtracking). Remedy: supervised fine-tuning on reasoning traces before GRPO, or start with a 1.5B+ model.

IPD alpha sweep: `alpha` in `R = alpha * R_individual + (1-alpha) * R_team` is the key design variable. At alpha=1 (pure individual), defection dominates. At alpha=0 (pure team), cooperation emerges. The sweep is continuous: partial alignment yields partial cooperation. Counterfactual credit achieves ~60-70% cooperation vs ~20-30% under uniform credit splitting in the 4-agent variant.

---

## Results and Findings

Q-Learning on Mountain Car converges after roughly 8000 episodes with the epsilon decay schedule given. Performance is very sensitive to the `bins` parameter — too coarse (bins=10) and the value function is aliased, too fine (bins=40) increases the state space and slows learning.

DQN on CartPole achieves an average reward above 300 by episode 250 from a starting average near 20. The target network update frequency matters: updating every episode causes instability, every 10 is stable.

PPO via stable-baselines3 convergence was fast and smooth — reaching average reward ~350 within 200 training iterations (~51k total timesteps). The `explained_variance` metric in the PPO logs tracks whether the value function is predictive; a value near 1 indicates the critic has learned the return structure.

RATE attribute ATEs (on social science prompts scored by the Qwen-trained RM):
- verbose: ATE = +2.51 (±6.26) — weak and noisy signal
- formal: ATE = +18.78 (±4.88) — strong, consistent effect; the RM strongly rewards formal register
- humorous: ATE = -5.03 (±3.46) — RM penalizes humor
- (helpfulness defined as fourth attribute but ATE not shown in outputs)

Surprising finding from RATE: formality has a much larger causal effect on RM scores than verbosity. Common assumption is that longer, more detailed responses score better, but the bigger driver is register (formal vs informal tone). This exposes a potential alignment failure where the RM rewards sounding authoritative over being genuinely useful.

SoT analysis comparing DeepSeek-R1-Distill-Qwen-1.5B vs Qwen2.5-1.5B-Instruct on interdisciplinary research prompts:
- self-questioning: 13 vs 5 (reasoning model 2.6x higher)
- perspective-shift: 8 vs 1 (reasoning model 8x higher)
- conflict: 4 vs 0 (reasoning model non-zero, base model zero)
- reconciliation: 0 vs 0 on open-ended prompts — both zero; open-ended research questions do not converge to a single answer unlike math problems
- Both models produced traces of similar length (~390 words), so length alone does not explain the quality difference

Surprising finding: reconciliation was non-zero (3) on math prompts for the reasoning model but zero for both on social science prompts. Math has ground truth that allows convergence; open-ended research questions sustain unresolved tension — perspective-shift and conflict persist rather than resolving.

---

## Connection to Chorus Project

GRPO's reward function structure directly maps onto the Chorus enhancement architecture: the accuracy reward (does the recommendation appear in future papers?), math-bridge reward (does the reasoning trace use mathematical formalism to connect domains?), and format reward (does the agent produce structured reasoning in think tags?) can be assembled as a multi-signal reward `R = R_accuracy + lambda_1 * R_math_bridge + lambda_2 * R_format` to fine-tune the DiscoveryAgent's language backbone. The SoT regex classifier is immediately reusable as an evaluation metric for Chorus reasoning traces — perspective-shift and conflict counts serve as proxy measures for genuine interdisciplinary deliberation quality, complementing the embedding-space geometric metrics from Kozlowski et al. and Zhang & Evans. The RATE causal methodology offers a diagnostic tool for auditing whatever reward model is used to train Chorus: by estimating ATEs of attributes like domain-specificity, mathematical density, and hedging language, one can verify the RM is rewarding genuine discovery rather than stylistic proxies.

---

## Potential Final Project Integration

Week 7 qualifies as one of the three required model types in at least two ways. First, GRPO training can serve as the RL fine-tuning component of the DiscoveryAgent — the multi-signal reward combining accuracy, math-bridge, and format objectives trains Chorus's recommendation backbone to optimize for the specific evaluation criteria of the Chorus project rather than general instruction-following. This is a distinct model paradigm from the embedding models (Week 5/6) and the RAG retrieval architecture already in Chorus. Second, the RATE methodology can serve as the evaluation model type: it is a causally-grounded measurement framework that quantifies which textual attributes drive recommendation quality scores, allowing principled ablation of the Chorus reward function. Either component satisfies the requirement of a model from a separate week, and both connect directly to the perplexity-based surprise metrics (GRPO rewards can encode surprise as a component, and RATE can estimate the causal effect of "interdisciplinary novelty" on RM scores).
