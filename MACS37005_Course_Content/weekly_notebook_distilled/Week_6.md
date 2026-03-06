# Week 6 Reference: Mechanistic Interpretability, Sparse Autoencoders, and Steering Vectors

## Core Techniques

Week 6 covers three modules, each representing a distinct paradigm for understanding and manipulating internal representations of transformer language models.

**Module 1: Mechanistic Interpretability with TransformerLens.** The central object is the HookedTransformer, a GPT-2-compatible architecture that exposes every intermediate activation via named hook points. The key analytic target is induction heads, a two-layer circuit where a "previous token head" in layer 0 (e.g., head 0.7) attends to the prior token and writes to the residual stream, while an "induction head" in layer 1 (e.g., heads 1.4 and 1.10) uses K-composition with that output to implement a pattern-completion mechanism. The module then reverse-engineers this circuit by directly multiplying weight matrices: the OV circuit (W_E @ W_V @ W_O @ W_U) reveals copying behavior, and the QK circuit (W_pos @ W_Q @ W_K^T @ W_pos^T) reveals positional attending. The intuition is that attention heads perform linear operations on the residual stream, and the full computation of a circuit can be expressed as a low-rank factored matrix, enabling exact algebraic analysis without running the model. Composition scores based on Frobenius norms quantify how much one head's output subspace aligns with a later head's input subspace.

**Module 2: Autoencoders and Sparse Autoencoders.** Beginning with basic autoencoders on MNIST (dense bottleneck for compression, convolutional for feature extraction, denoising for noise removal), the module moves to Sparse Autoencoders applied to the residual stream of a language model. The core motivation is superposition: a d-dimensional model can represent more than d near-orthogonal features when features are sparse. An SAE disentangles these by learning an overcomplete dictionary (d_sae >> d_in) with an L1 sparsity penalty that encourages most latent activations to be zero at any given input. The architecture is h' = W_dec @ ReLU(W_enc @ (h - b_dec) + b_enc) + b_dec. The loss is L_reconstruction + sparsity_coeff * L_sparsity. Decoder columns are normalized to prevent magnitude cheating. Dead latents are addressed by resampling: resetting dead encoder/decoder columns to point toward high-reconstruction-error inputs. The pilot implementation trained an ArxivSAE on GPT-2 layer 6 activations from 20,000 scientific abstracts across four domains.

**Module 3: Steering Vectors via Contrastive Activation Addition (CAA).** The key intuition, grounded in linear representation hypothesis (Park et al. 2023, Kim/Evans/Schein 2025), is that semantic concepts are encoded as directions in activation space. A steering vector is extracted by computing the difference of mean activations across contrastive conditions (positive vs. negative examples of a behavior), then injected at inference by adding alpha * steering_vector to the residual stream at a chosen layer. The nnsight library provides a context manager interface for accessing and patching intermediate activations without modifying model source code. The module demonstrates this for sentiment steering on GPT-2 and sycophancy reduction on Llama-2 (7B and 13B) using the CAA framework from Rimsky et al. The pilot applies CAA to interdisciplinary reasoning: using arXiv multi-category papers as positive condition and single-domain papers as negative condition to extract a cross-domain steering direction.

---

## Key Code Patterns

TransformerLens model loading and activation caching:
```python
model = HookedTransformer.from_pretrained("gpt2", device=device)
logits, cache = model.run_with_cache(tokens, names_filter=["blocks.6.hook_resid_post"])
h = cache["blocks.6.hook_resid_post"]  # shape [batch, seq, d_model]
```
Collects any named intermediate activation without modifying model code.

Hook-based intervention (add steering vector):
```python
def hook_fn(activations, hook):
    activations[:, :, :] += alpha * steering_vec[None, None, :]
    return activations
model.add_hook("blocks.6.hook_resid_post", hook_fn)
# or inline:
loss = model.run_with_hooks(tokens, fwd_hooks=[("blocks.1.attn.hook_pattern", hook_fn)])
```
Hooks fire on forward pass; returning modified tensor patches the activation in-place.

nnsight context manager for activation access:
```python
from nnsight import LanguageModel
model = LanguageModel("openai-community/gpt2", device_map="cuda", dispatch=True)
with model.generate(prompt, max_new_tokens=20):
    layer_output = model.transformer.h[4].output[0].save()
```
Saves intermediate activations without the model's own caching overhead.

SAE forward pass and loss:
```python
class ArxivSAE(nn.Module):
    # W_enc: [d_in, d_sae], W_dec: [d_sae, d_in]
    def forward(self, h):
        z = F.relu(h @ self.W_enc + self.b_enc)          # [batch, d_sae]
        h_hat = z @ self.W_dec_normalized + self.b_dec   # [batch, d_in]
        L_recon = (h - h_hat).pow(2).mean()
        L_sparse = self.sparsity_coeff * z.abs().mean()
        return h_hat, z, L_recon + L_sparse
```
W_dec_normalized normalizes decoder columns to unit L2 norm per latent dimension.

Contrastive steering vector extraction:
```python
def compute_mean_activation(sentences, layer_modules):
    # returns dict[layer_module -> mean_activation_vector [768]]
    ...
pos_acts = compute_mean_activation(POSITIVE_SENTENCES, layer_modules)
neg_acts = compute_mean_activation(NEGATIVE_SENTENCES, layer_modules)
steering_vec = pos_acts[layer] - neg_acts[layer]  # shape [768]
```
The difference vector points from "negative" to "positive" behavior in activation space.

SAE-based cross-domain steering vector construction:
```python
cross_domain_feat_ids = [feat_ids with 2-3 active domains]
steering_vector = mean([sae.W_dec_normalized[fid] for fid in cross_domain_feat_ids])
```
Averages decoder directions of cross-domain latents to construct a semantically grounded steering vector.

Interdisciplinarity metric via entropy:
```python
def compute_interdisciplinarity_v2(text, gpt2, centroid_matrix, device):
    # project mean activation onto 4 domain centroids, softmax, compute entropy
    h_mean = cache[HOOK_NAME].mean(dim=1)                      # [768]
    cos_sims = F.cosine_similarity(h_mean, centroid_matrix)    # [4]
    probs = F.softmax(cos_sims, dim=0)
    return -(probs * probs.log()).sum().item()                  # entropy in nats
```
Higher entropy means the text activates more domains equally, i.e., more interdisciplinary.

Factored matrix for circuit analysis:
```python
from transformer_lens.utils import FactoredMatrix
full_OV = FactoredMatrix(model.W_E @ model.W_V[layer, head], model.W_O[layer, head] @ model.W_U)
# access without materializing full [50k, 50k] matrix:
submatrix = full_OV[left_indices, left_indices].AB
```

---

## Hyperparameters and Design Choices

SAE architecture: d_in=768 (GPT-2 residual stream), d_sae=6144 (8x expansion ratio). The 8x ratio follows Anthropic's approach: too small and the SAE cannot disentangle all features; too large and dead latents proliferate. The pilot found 3,033 active out of 6,144 features (mean L0=967.7) and 3,111 dead features, suggesting the expansion could be reduced or resampling improved.

Sparsity coefficient lambda=5e-4. This is a relatively small penalty, appropriate when d_in is large (768) and you want to capture genuine feature activation rather than forcing extreme sparsity. Higher lambda increases sparsity but risks increasing reconstruction loss and creating more dead latents.

Layer selection for activation extraction: layer 6 out of 12 (middle of GPT-2). Middle layers encode the most abstract semantic content; earlier layers encode surface syntax and later layers encode task-specific predictions. The pilot confirms this: layer 6 produces distinguishable domain centroids across physics, CS, math, and biology.

Steering layer: layers 4-5 for the sentiment experiment, layer 6 for the interdisciplinary experiment. The lesson is that steering is most effective at the same layer where the concept is encoded. Steering too early (layers 1-2) produces incoherent text; too late (layers 10-11) has limited effect because the forward pass is nearly complete.

Steering factor (alpha): effective range was +/-5 for sentiment on GPT-2 (cherry-picked). For interdisciplinary steering the pilot used alpha in [-3, +3] scaled by a base magnitude. Larger alpha produces qualitatively observable shifts but risks incoherence or repetition. The 95% CI bands across 13 prompts show monotonically increasing interdisciplinarity for positive alpha.

Math-debiasing: the cross-domain steering vector was 47.7% math-associated because mathematical notation appears naturally across all scientific fields. Two fixes were compared: (1) projection removal improved the metric from 0.855 to 0.920; (2) contrastive subtraction v_cross - 1.0 * v_math improved it further to 0.953. Contrastive subtraction is more aggressive because it subtracts the full scaled math direction rather than only its parallel component.

Resampling: simple resampling (random new vector) is sufficient for toy model settings. Advanced resampling (weighted by reconstruction loss) helps in high-dimensional real model settings where uniform random initialization is less likely to point toward underrepresented regions of the data manifold.

---

## Results and Findings

Module 1 results: GPT-2 Small has induction heads in middle layers (approximately layers 4-7) with induction scores above 0.6. Heads 1.4 and 1.10 in the 2-layer toy model achieve combined OV circuit top-1 accuracy of 95.6% on copying tasks (vs. 30.79% for either head alone), confirming that rank-64 OV matrices benefit from being split across two heads to approximate a higher-rank copying operation. Head 0.7 is the strongest previous-token head; ablating it collapses the induction score from 0.68 to near zero, while ablating other heads has minimal effect.

Module 2 pilot results: SAE trained on 20,000 arXiv abstracts (4 domains), 1,269,894 activation tokens. Reconstruction loss converged to 0.021. Of 6,144 latents, 3,033 were active (mean L0=967.7); 91 were sharp discriminative features (3.0%), meaning they activated significantly more in one or a subset of domains. Feature taxonomy: 13 single-domain (one domain >80% relative activation), 43 two-domain, 35 three-domain, 2,742 broad (fire across all domains indiscriminately), 3,111 dead. One Math-specific feature had a 33x activation ratio relative to other domains. Cross-domain features cluster around Physics+Math, CS+Biology, and Physics+CS+Biology combinations.

Module 3 steering results: with positive alpha (+2, after math debiasing), interdisciplinarity entropy score increases from baseline 0.855 to 0.953, measured via logit-based domain probability entropy. Per-domain analysis shows math probability decreases slightly under positive steering while biology and CS probabilities increase, consistent with the debiased cross-domain direction. Qualitative examples: at alpha=+2, "In this paper we study the" completes with text bridging neuroscience and social science or molecular biology and computing, rather than staying within one field. At alpha=-2, outputs narrow to single-domain scientific language.

Surprising finding: broad features (90.4% of active latents) are unhelpful for domain discrimination and for steering. Directly using all active SAE features as a steering basis introduced noise. Restricting to the 91 discriminative features substantially improved metric validity. This mirrors the polysemantic latents problem in Anthropic's toy model paper: most learned features are not monosemantic.

---

## Connection to Chorus Project

The SAE trained on scientific literature activations is directly applicable to the Chorus embedding pipeline: run Chorus's document corpus through a language model, collect residual stream activations at a middle layer, and train an SAE to decompose the corpus into sparse interpretable features. Cross-domain features identified by the SAE (those that activate significantly across two or more research domains) serve as candidates for "converging but underexplored" combinations, which is exactly the discovery signal your DiscoveryAgent needs. The entropy-based interdisciplinarity metric from Module 3 could be repurposed as an evaluation function within the MatchmakerAgent to score how cross-domain a retrieved or recommended document is, giving a continuous measure rather than a binary classification. Additionally, steering vectors derived from cross-domain SAE decoder directions could be used to bias RAG retrieval: injecting the steering vector into the query representation before nearest-neighbor search would systematically bias retrieval toward cross-domain documents, directly operationalizing the "proactive recommendation" behavior Chorus aims to provide.

---

## Potential Final Project Integration

Module 2's SAE satisfies one of the three required model types for the final project. Concretely, you can present the ArxivSAE trained on a language model's layer activations from Chorus's document corpus as your representation learning / feature decomposition model, noting that it is a distinct architecture (overcomplete dictionary learning with L1 sparsity) from embedding models and from generative LLMs. Module 3's steering vector / CAA technique is a second candidate: it is a distinct class of method (activation engineering at inference time) that could be framed as your probing / intervention model. If your three weeks are Module 2 (SAE), Module 3 (steering vectors), and a third week covering embedding space geometric analysis or topic modeling (e.g., the Kozlowski/Evans approach), you would have a clean separation of representation learning, intervention method, and geometric analysis. The pilot implementation in this notebook is particularly strong evidence that the methods compose: you can train an SAE, identify cross-domain latents, aggregate their decoder directions into a steering vector, and measure the resulting output's interdisciplinarity with an entropy metric, all in a single reproducible pipeline on arXiv data directly relevant to Chorus's knowledge domain.
