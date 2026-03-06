# Week 2 Reference: Neural Embeddings, BERT, and Interpretability

Source: MACS 37005 Week_2.ipynb (915 cells). Instructors: James Evans; TAs: Shiyang Lai, Gio Choi, Avi Oberoi, Jesse Zhou. Key libraries: gensim 4.x, transformers (HuggingFace), shap, sklearn, torch.

## Core Techniques

Week 2 covers three tiers of text representation, moving from shallow to deep: Word2Vec (skip-gram and CBOW), Doc2Vec, Procrustes-aligned embedding comparison, BERT contextual embeddings, BERT fine-tuning for sequence classification, and SHAP-based transformer interpretability. A fifth module on diffusion models is included but is less central to applied NLP workflows.

The key intuition running through all three tiers is distributional semantics: meaning is encoded in statistical regularities of co-occurrence. Word2Vec captures this at the word level through a shallow two-layer neural network trained with either skip-gram (predict context from target) or CBOW (predict target from context). The self-supervised signal comes from treating nearby words as positives and randomly sampled words as negatives (negative sampling). Crucially for the Chorus project, this architecture encodes semantic geometry—directions in embedding space correspond to semantic relationships, and cosine similarity in that space proxies conceptual closeness.

Doc2Vec extends the same mechanism to document-level representations by adding a memory vector (the document ID) to the context window during training, letting the model average over word contributions to generate a document embedding. Two variants exist: DM (Distributed Memory, analogous to CBOW) and DBOW (Distributed Bag of Words, analogous to skip-gram). The ArXiv and APS paper datasets in the notebook make this directly applicable to academic literature.

Procrustes alignment is the geometric bridge technique. When two separate embedding spaces are trained on different corpora or time slices, their coordinate systems are arbitrary rotations of each other. Procrustes alignment finds the optimal orthogonal rotation matrix via SVD (singular value decomposition) of the dot product matrix between aligned vocabulary subsets, then applies that rotation so embeddings from both spaces become comparable. Post-alignment, cosine distance (1 - cosine similarity) between a word's two representations measures semantic drift.

BERT (Bidirectional Encoder Representations from Transformers) represents the third tier. Unlike Word2Vec, BERT produces contextual embeddings—the same token gets a different vector depending on its surrounding context, resolving polysemy. BERT uses masked language modeling (MLM) and next-sentence prediction (NSP) as pre-training objectives on Wikipedia + BookCorpus. The notebook fine-tunes bert-base-uncased on the CoLA (Corpus of Linguistic Acceptability) task and extracts sentence-level embeddings via [CLS] token pooling or mean pooling over all token hidden states. SHAP (SHapley Additive exPlanations) is then used to explain which tokens drive a specific model output, applying game-theoretic Shapley value attribution to transformer pipelines.

## Key Code Patterns

Word2Vec training from gensim:
`Word2Vec(tokenized_texts, vector_size=100, window=10)` — trains from scratch on a tokenized corpus.

Semantic similarity and analogy:
`model.wv.most_similar(positive=['king','woman'], negative=['man'], topn=5)` — word arithmetic in embedding space.

Vector lookup and similarity:
`cosine_similarity(model.wv['word1'].reshape(1,-1), model.wv['word2'].reshape(1,-1))[0][0]` — pairwise cosine between two word vectors.

Doc2Vec with TaggedDocument:
`TaggedDocument(words=row['normalized_words'], tags=[doi, year, keyword])` — tags can be any identifier; used to retrieve document vectors later by tag.

Doc2Vec training:
`Doc2Vec(documents=tagged_docs, vector_size=100, window=5, min_count=2, workers=4, epochs=40)` — DBOW by default; `dm=0` for explicit DBOW.

Document retrieval:
`model.docvecs.most_similar([query_vector], topn=N)` — nearest neighbors in document space.

Procrustes alignment pipeline:
```python
in_base, in_other = intersection_align_gensim(base_model, other_model)
base_vecs = normalize(in_base.wv.vectors)
other_vecs = normalize(in_other.wv.vectors)
m = other_vecs.T @ base_vecs
u, _, v = np.linalg.svd(m)
ortho = u @ v
other_model.wv.vectors = normalize(other_model.wv.vectors) @ ortho
```

Semantic drift measurement:
`drift = 1 - cosine_similarity(base_model.wv[word], aligned_other_model.wv[word])` — high drift indicates the word's meaning shifted between the two corpora.

BERT tokenization:
`tokenizer.encode_plus(text, add_special_tokens=True, max_length=256, padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt')` — standard BERT input preparation.

BERT sentence embedding ([CLS] pooling):
`outputs = model(input_ids, attention_mask=mask, output_hidden_states=True); cls_vec = outputs.hidden_states[-1][0, 0, :].cpu().numpy()` — last hidden state, first token.

BERT sentence embedding (mean pooling):
`sentence_vec = np.mean([tok.cpu().numpy() for tok in outputs[0][0]], axis=0)` — averages all token embeddings from last hidden layer.

BERT fine-tuning setup:
`BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2, output_hidden_states=True)` with `AdamW(model.parameters(), lr=2e-5, eps=1e-8)`.

Linear warmup scheduler:
`get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)` — standard BERT training recipe.

SHAP for transformers:
`explainer = shap.Explainer(model, tokenizer); shap_values = explainer(texts); shap.plots.text(shap_values)` — produces token-level attribution scores.

Matthews Correlation Coefficient evaluation:
`matthews_corrcoef(true_labels, predicted_labels)` — preferred metric for imbalanced classification; used alongside standard accuracy and confusion_matrix.

Visualization pipeline:
`pca = PCA(n_components=50); reduced = pca.fit_transform(embedding_matrix); tsne = TSNE(n_components=2, perplexity=30, random_state=42); coords = tsne.fit_transform(reduced)` — PCA first reduces dimensionality before t-SNE for efficiency.

## Hyperparameters and Design Choices

Word2Vec: vector_size=100, window=10, no explicit min_count specified in the main hobbies corpus demo but min_count=1 when loading a pre-trained model. These are not heavily tuned in the notebook—they are standard defaults. Window size 10 is larger than the typical 5, which captures broader discourse-level context; useful for academic text where domain terms appear within long phrases.

Doc2Vec: vector_size=100, window=5, min_count=2, workers=4, epochs=40. The 40-epoch setting is notably higher than the Word2Vec default (usually 5–10 epochs), reflecting that document-level representations require more iterations to stabilize. min_count=2 eliminates hapax legomena without aggressively pruning.

t-SNE: perplexity=min(30, len(words)-1), so perplexity adapts to vocabulary subset size. The PCA-first-then-t-SNE pipeline is the standard efficiency trick—t-SNE's O(n²) cost makes applying it directly to 768-dim BERT vectors slow.

BERT fine-tuning: MAX_LEN=256 (not 512, which halves memory and compute), lr=2e-5, eps=1e-8, epochs=4, num_warmup_steps=0. The 2e-5 learning rate is the canonical "don't blow up the pre-trained weights" choice for BERT classification. Using output_hidden_states=True is necessary to extract intermediate representations for sentence embedding after fine-tuning. The notebook uses bert-base-uncased (110M parameters), not bert-large, making it feasible on a single GPU or Colab.

SHAP: no dedicated hyperparameters—the explainer wraps the pipeline object and tokenizer directly. The main design choice is using shap.Explainer (the new unified API) rather than older class-specific explainers like shap.DeepExplainer.

Lessons: Pre-trained embeddings (loaded via gensim.models.KeyedVectors.load_word2vec_format) outperform from-scratch Word2Vec on small corpora. Procrustes alignment quality depends critically on vocabulary intersection size—small intersections produce unreliable rotation matrices. BERT mean pooling tends to produce more semantically uniform sentence embeddings than [CLS] pooling when the model has not been explicitly trained with a sentence-pair objective (like SBERT).

## Results and Findings

- Word2Vec trained on the 448-document hobbies corpus demonstrates clean analogical reasoning (king - man + woman ≈ queen) and semantic grouping via doesnt_match(), confirming the geometry holds even on small corpora.
- Doc2Vec on the ArXiv 10,000-paper subset successfully clusters by category (cs.LG, cs.CL, cs.AI, cs.SI) in PCA/t-SNE projections, with most-similar-document retrieval returning thematically coherent results.
- Procrustes-aligned Word2Vec on ASCO medical abstracts (grouped by year) surfaces meaningful semantic drift: oncology terminology that shifts in usage over time becomes visible as high (1 - cosine similarity) divergence scores. This is a direct computational analogue to the "semantic change" literature.
- Fine-tuned bert-base-uncased on CoLA achieves MCC reported via matthews_corrcoef alongside accuracy and a full classification_report. The exact MCC value varies by run, but the notebook demonstrates that even 4 epochs is sufficient for the model to converge on this binary classification task.
- SHAP explanations on the sentiment pipeline reveal that negation tokens (not, never) flip the sign of nearby sentiment-bearing tokens' Shapley values, a confirmation of BERT's context sensitivity that static embeddings miss entirely.
- Counterintuitive: Cosine similarity heatmaps across BERT sentence embeddings of related documents often show high within-category similarity even for documents in very different scientific subfields, suggesting that BERT's pre-training may over-smooth domain-specific distinctions compared to domain-adapted models.
- Using bert-large-uncased-whole-word-masking-finetuned-squad for question answering shows strong span extraction, but the notebook warns that the model's confidence (logits) is not calibrated—high logit scores do not reliably indicate factual correctness.

## Connection to Chorus Project

The Procrustes alignment and Doc2Vec components are directly relevant to the Chorus embedding pipeline. To identify "converging but underexplored" domain combinations, Chorus could embed corpora from distinct research domains (analogous to the ASCO year-slice approach), align the embedding spaces using smart_procrustes_align_gensim, and then measure inter-domain proximity by computing cosine similarity between domain centroid vectors after alignment. Domains that are geometrically close but lack bridging papers are candidates for underexplored convergence.

BERT sentence embeddings (either via fine-tuned bert-base-uncased [CLS] pooling or mean pooling) provide a stronger foundation than Word2Vec for representing individual papers in the Chorus retrieval index, since contextual embeddings handle polysemous terminology (e.g., "network" meaning social network vs. neural network vs. protein network) that confounds static embeddings. The DiscoveryAgent in Chorus could use BERT-encoded paper embeddings as its base representation layer, feeding those vectors into the geometric analysis components.

SHAP attribution provides an interpretability layer for Chorus recommendations: after identifying a candidate cross-domain paper, SHAP can highlight which tokens (keywords, author-specific phrases) drove the cross-domain similarity score, making the recommendation legible to researchers rather than a black-box output.

## Potential Final Project Integration

Week 2 qualifies as one of the three required model types for the final project, specifically contributing the static neural embedding component (Word2Vec/Doc2Vec via gensim) and the contextual embedding component (BERT via transformers). In terms of the three-week model diversity requirement, this week most naturally represents the embedding/representation learning layer. The Procrustes alignment technique directly implements the "geometric analysis" half of the project's core methodological claim—that converging domains will exhibit decreasing inter-domain cosine distance over time, detectable by aligning year-sliced embedding spaces and tracking centroid drift. The Doc2Vec architecture is also the most straightforward baseline for encoding research abstracts as fixed-length vectors before applying the perplexity-based surprise metric (Zhang & Evans 2025) on top. If the final project implements three models from three weeks, Week 2 can be represented by Doc2Vec or fine-tuned BERT as the document representation model, which then feeds into downstream anomaly detection or graph-based convergence analysis from later weeks.
