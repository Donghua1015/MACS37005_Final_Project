# Week 4 Reference: Causal Inference with Deep Learning

Course: MACS 37005, Week 4
Instructor: James Evans; TAs: Shiyang Lai, Gio Choi, Avi O
Applied dataset: arXiv papers (2.9M total; 105,002 sampled); also IHDP simulation, 401(k) SIPP data, Moral Machine experiment


## Core Techniques (Four Modules)

### Module 1: S-Learner, T-Learner, TARNet (Heterogeneous Treatment Effects under Selection on Observables)

The central problem is estimating causal effects when treatment assignment is not random but can be accounted for by observed covariates X. The identifying assumption is "selection on observables" (unconfoundedness): treatment T is independent of potential outcomes Y(0), Y(1) given X.

S-Learner: A single neural network trained on concatenated [X, T] input. At inference, counterfactual outcomes are estimated by clamping T=0 and T=1 separately. CATE_i = Y_hat(1|X_i) - Y_hat(0|X_i). Weakness: the network may ignore T if its signal is weak relative to X.

T-Learner: Two independent networks, one fit on control observations (T=0) and one on treated (T=1). Stronger treatment signal, but does not encourage covariate balance across groups.

TARNet (Treatment-Agnostic Representation Network, Shalit et al. 2017): A shared "representation" trunk phi(X) (3 dense layers, no regularization) feeds into two separate "hypothesis" heads, one for Y(0) and one for Y(1) (each 2 dense layers with L2 regularization). The shared trunk learns a balanced representation of X across treatment groups; the separate heads prevent information leakage between potential outcomes. This is the key architectural improvement over the T-Learner.

Evaluation metric: sqrt(PEHE) = sqrt(mean((CATE_pred - CATE_true)^2)), applicable only in simulations. In real data, ATE = mean(CATE) is reported.

### Module 2: Causal Mediation with cGNF (Causal Graphical Normalizing Flows)

When we want to understand mechanisms — not just total effects — we need mediation analysis. The goal is to decompose total effect A -> Y into a Natural Direct Effect (NDE, A -> Y not through M) and Natural Indirect Effect (NIE, A -> M -> Y). The challenge is that standard regression-based (Baron-Kenny) approaches fail when there are exposure-induced confounders (L is affected by A and also confounds M -> Y).

Normalizing flows are deep learning models that learn an invertible transformation from a complex data distribution p(X) to a standard normal. cGNF (Malone et al.) extends this to causal settings by learning a flow that respects the causal graph (DAG). Once trained, interventional distributions p(Y | do(A=a)) are simulated by Monte Carlo: freeze A at a given value, sample the remaining variables from the learned flow.

Three-step workflow: process() -> train() -> sim(). The DAG is specified as a CausalGraphicalModel adjacency matrix and passed to process(). The sim() function handles ATE (mediator=None), NDE/NIE (mediator=['M']), and Path-Specific Effects (mediator=['L','M'] in causal order).

### Module 3: Double/Debiased Machine Learning (DML, Chernozhukov et al. 2018)

DML's core insight: naive application of ML to estimate the causal parameter theta_0 in y = theta_0 * d + g_0(x) + epsilon is invalid because regularization bias makes the estimator converge slower than 1/sqrt(n). Two fixes applied in sequence: (1) Orthogonalization — partial out g_0(x) and m_0(x) from y and d respectively, yielding residuals V = D - m_hat(X) and U = Y - g_hat(X); then regress U on V. This removes the confounding signal from g_0 and removes regularization bias. (2) Sample splitting (cross-fitting) — nuisance models g_hat and m_hat are estimated on one fold, residuals computed on another, to remove bias induced by in-sample overfitting.

The doubleml package implements three model variants:
- DoubleMLPLR: Partially Linear Regression — theta is the coefficient on D in a linear regression after partialing out g_0 and m_0. Assumes additive separability.
- DoubleMLIRM: Interactive Regression Model — allows fully heterogeneous treatment effects (g_0(D,X) without separability). Uses IPW-like reweighting.
- DoubleMLIIVM: Interactive IV Model — for LATE estimation when D is endogenous and an instrument Z is available (e.g., eligibility e401 as instrument for participation p401).

### Module 4: Prediction-Powered Inference (PPI, Angelopoulos et al.)

PPI addresses the problem of expensive human labels. Given n small labeled samples {(X_i, Y_i)} and N large unlabeled samples {X_tilde_i} with LLM predictions {Y_hat_tilde_i}, PPI constructs valid confidence intervals for a population parameter theta by correcting for the bias of the LLM predictions using the labeled sample. The confidence interval is always valid regardless of LLM accuracy, but narrows more when LLM-human agreement (PPI correlation rho) is high.

Key metric: effective sample size n_0 = n(k+1) / (k(1-rho^2)+1), where k = N/n. When rho=0.7 and k=10, n_0 = 1443 (80% gain over n=800). SE ratio = sqrt(1 - (k/(1+k)) * rho^2).

The "mixed subjects design" contrasts with "silicon subjects design": silicon subjects alone are fast and cheap but can be massively biased (up to 363% of true effect in the Moral Machine experiment). PPI guarantees valid coverage (nominal 95%) while silicon-only intervals lose coverage as N grows.


## Key Code Patterns

S-Learner architecture (Sequential API, tensorflow.keras):
```python
s_learner = tf.keras.models.Sequential([
    Dense(200, activation='elu'),
    Dense(200, activation='elu'),
    Dense(200, activation='elu'),
    Dense(100, activation='elu', kernel_regularizer=regularizers.l2(0.01)),
    Dense(100, activation='elu', kernel_regularizer=regularizers.l2(0.01)),
    Dense(1, activation=None, kernel_regularizer=regularizers.l2(0.01)),
])
# Input: concat [X, T]; at inference clamp T=0 or T=1
```

T-Learner architecture (Functional API):
```python
def make_tlearner(input_dim, reg_l2):
    x = Input(shape=(input_dim,))
    y0_h = Dense(100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(x)
    y1_h = Dense(100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(x)
    # two more hidden layers each, then output heads
    concat_pred = Concatenate(1)([y0_pred, y1_pred])
    return Model(inputs=x, outputs=concat_pred)
```

TARNet architecture (key structural difference from T-Learner):
```python
def make_tarnet(input_dim, reg_l2):
    x = Input(shape=(input_dim,))
    phi = Dense(200, activation='elu', kernel_initializer='RandomNormal')(x)  # shared, NO reg
    phi = Dense(200, activation='elu', kernel_initializer='RandomNormal')(phi)
    phi = Dense(200, activation='elu', kernel_initializer='RandomNormal')(phi)
    y0_h = Dense(100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(phi)  # separate heads WITH reg
    y1_h = Dense(100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(phi)
    # two more hidden layers each, then output heads
    concat_pred = Concatenate(1)([y0_pred, y1_pred])
    return Model(inputs=x, outputs=concat_pred)
```

Custom treatment-masked loss (used by T-Learner and TARNet):
```python
def regression_loss(concat_true, concat_pred):
    y_true = concat_true[:, 0]
    t_true = concat_true[:, 1]
    y0_pred = concat_pred[:, 0]
    y1_pred = concat_pred[:, 1]
    loss0 = (1. - t_true) * mse(y_true, y0_pred)  # only control loss for T=0
    loss1 = t_true * mse(y_true, y1_pred)           # only treated loss for T=1
    return loss0 + loss1
# For binary Y, replace mse with BinaryCrossentropy
```

CATE estimation after model.predict():
```python
concat_pred = model.predict(X)
y0_pred, y1_pred = concat_pred[:, 0], concat_pred[:, 1]
CATE = y1_pred - y0_pred
ATE = CATE.mean()
```

cGNF three-step workflow:
```python
from cGNF import process, train, sim
process(path, dataset_name, dag_name, test_size=0.2, cat_var=['C'], seed=None)
train(path, dataset_name, model_name,
      trn_batch_size=128, val_batch_size=2048, learning_rate=1e-4,
      nb_epoch=50000, nb_estop=50,
      emb_net=[90,80,70,60,50], int_net=[50,40,30,20,10], seed=8675309)
sim(path, dataset_name, model_name, n_mce_samples=50000,
    treatment='A', cat_list=[0,1], outcome='Y',
    mediator=None,    # None=ATE; ['M']=NDE/NIE; ['L','M']=PSE
    inv_datafile_name='ATE')
```

DML with doubleml (PLR example):
```python
import doubleml
data_backend = doubleml.DoubleMLData(df, y_col='Y', d_cols='D', x_cols=feature_list)
dml_model = doubleml.DoubleMLPLR(data_backend, ml_l=lasso, ml_m=lasso_class, n_folds=3)
dml_model.fit(store_predictions=True)
print(dml_model.summary)  # coef, se, t-stat, p-value, 2.5%, 97.5%
# Also: DoubleMLIRM for heterogeneous effects; DoubleMLIIVM for IV/LATE
```

DML nuisance learner wrappers (sklearn Pipeline for standardization):
```python
lasso = make_pipeline(StandardScaler(), LassoCV(cv=5, max_iter=10000))
mlp = Pipeline([("scaler", StandardScaler()),
                ("regressor", KerasRegressor(model=create_mlp_model, epochs=50))])
```

PPI effective sample size and SE ratio:
```python
def n0(rho, n, k):   # k = N/n
    return (n * (k + 1)) / (k * (1 - rho**2) + 1)
def se_ratio(rho, k):
    return np.sqrt(1 - (k / (1 + k)) * rho**2)
```

PPI confidence interval (ppi-python package):
```python
from ppi_py import ppi_logistic_ci, classical_logistic_ci
ppi_ci = ppi_logistic_ci(X_labeled, Y_labeled, Yhat_labeled,
                          X_unlabeled, Yhat_unlabeled, alpha=0.05)
```


## Hyperparameters and Design Choices

Module 1 (TARNet/T-Learner/S-Learner):
- Optimizer: SGD with Nesterov momentum (sgd_lr=0.01 to 0.001, momentum=0.9)
- Activation: elu throughout (better than relu for avoiding dead neurons)
- Regularization: L2 on hypothesis heads only (reg_l2=0.01); NO regularization on representation trunk
- Batch size: 32-64 (smaller batches improve gradient quality for causal models)
- Early stopping: patience=20-30 on val_loss; ReduceLROnPlateau factor=0.5, patience=5-15
- Epochs: 100-300 with early stopping
- TARNet v2 improvements: BatchNormalization in representation trunk, Dropout(0.2) in heads, Adam optimizer (lr=0.0005), smaller batch size=32 for better gradients, lower reg_l2=0.001
- Key lesson: TARNet with too-strong shared representation can collapse both heads to the overall mean, making it underestimate treated-group outcomes. T-Learner is more robust when the shared representation underfits.

Module 2 (cGNF):
- emb_net = [90, 80, 70, 60, 50], int_net = [50, 40, 30, 20, 10] (decreasing width)
- learning_rate = 1e-4, nb_epoch = 50000, nb_estop = 50
- n_mce_samples = 50000 for Monte Carlo estimation
- Diagnostic check: transform data to latent space; each variable should approximate N(0,1). Non-normal latent distributions indicate poor model fit.
- Training time: 10-30 minutes on CPU; GPU recommended.

Module 3 (DML):
- n_folds=3 for cross-fitting (standard; 5 also used)
- Lasso: LassoCV(cv=5, max_iter=10000), with StandardScaler
- Random Forest: n_estimators=500, max_depth=7, max_features=3, min_samples_leaf=3
- XGBoost: n_estimators=35, eta=0.1 (regression); separate params per nuisance function via set_ml_nuisance_params()
- 1-layer MLP: n_hidden=32, lr=0.01, epochs=50, batch=32
- 2-layer MLP: n_hidden1=64, n_hidden2=32, dropout=0.2, lr=0.01, epochs=50, batch=64
- Key lesson: 1-layer MLP failed to converge in the 401(k) dataset (~10k samples) but succeeded on 105k arXiv samples. Data size is the main driver of MLP viability.
- Key lesson: Tree-based methods (RF, XGBoost) show the most stable ATE across PLR vs IRM (gap < 0.05 pp); Lasso shows the largest gap (0.41 pp); MLPs intermediate (0.17-0.29 pp).

Module 4 (PPI):
- PPI correlation rho for GPT-4 Turbo on Moral Machine decisions ranged 0.049 to 0.353 (modest interchangeability)
- At rho=0.353 and k=10, effective sample size gain is limited (max 11,275 from 10,000 human + 100,000 LLM)
- Silicon-only bias can be as large as 363% of the true effect (e.g., pedestrian vs. passenger attribute)
- PPI maintains nominal 95% coverage regardless of LLM accuracy; silicon-only loses coverage as N grows


## Results and Findings

Module 1 applied to arXiv papers (T = interdisciplinary, Y = published):
- Naive correlation: +2.00 pp
- S-Learner: ATE = 3.52 pp, CATE std = 0.30 pp (near-zero heterogeneity — likely ignoring T signal)
- T-Learner: ATE = 5.12 pp, CATE std = 7.72 pp (highest heterogeneity, likely upward biased)
- TARNet: ATE = 4.19 pp, CATE std = 5.22 pp (balanced estimate, moderate heterogeneity)
- All three exceed the naive estimate, indicating downward bias from confounders (year, primary_category)
- Surprising: S-Learner failed to learn any heterogeneity. CATE distribution was almost a point mass.

Module 1 pilot on citation impact (T = novel combination, Y = top 10% citations, n=4,800 arXiv papers):
- TARNet failed — shared representation collapsed both heads toward overall mean (8.12% predicted vs 11.68% observed for treated group)
- T-Learner succeeded — ATE = 2.22 pp (vs naive 2.95 pp); controlled for confounders
- Heterogeneous effects: peaked in 2015-2019 for all team sizes; large teams (7+) benefit most (4.93 pp in middle period); medium teams (4-6) recently outperform large teams (2.92 vs 2.68 pp)
- Surprising: TARNet's shared trunk can backfire when treatment group size is unbalanced or the outcome space is bimodal

Module 2 (cGNF mediation, synthetic DGP):
- True ATE = 1.0; cGNF estimated ATE approximately 1.0 (accurate)
- True NDE = 0.45, NIE = 0.55; cGNF recovered both within margin
- Path-Specific Effects with exposure-induced confounder L: cGNF handles this correctly; traditional Baron-Kenny approach is biased

Module 3 applied to 401(k) data (n=9,915 SIPP households):
- Baseline naive estimate: $19,559
- DML estimates (PLR + IRM with 6 learners): $7,000-$10,000 range (substantially attenuated)
- All learners except 1-layer MLP converged; estimates stable across methods
- DML for arXiv (PLR): Lasso=3.35pp, RF=3.04pp, XGBoost=3.12pp, 1-layer MLP=3.17pp, 2-layer MLP=3.20pp
- All highly significant (p < 1e-26); all methods in tight agreement (range ~0.31 pp across PLR)
- DML gives more conservative estimates than T-Learner/TARNet (3.1-3.4 pp vs 4.2-5.1 pp)

Module 4 (PPI, Moral Machine experiment):
- Human AMCEs match Awad et al. 2018 ground truth
- GPT-4 Turbo silicon estimates biased up to 363% of true effect (pedestrian vs passenger: predicted 49 pp, true 11 pp)
- PPI coverage maintained at 95%; silicon-only coverage approaches 0% at large N
- At rho=0.7, k=10: n_0 = 1,443 from n=800 (80% effective gain); SE ratio = 71.4%
- Bias and high PPI correlation are not mutually exclusive: the most-biased attribute also had the second-highest rho


## Connection to Chorus Project

Module 1 (TARNet/T-Learner) directly supports Chorus's DiscoveryAgent: once candidate "converging domain pairs" are surfaced using embedding geometry, TARNet can estimate the heterogeneous treatment effect of pursuing a novel cross-domain combination on downstream impact (citations, publications), conditioned on researcher attributes (team size, institutional context). The CATE from TARNet gives a personalized payoff signal, enabling Chorus to rank recommendations not just by novelty (embedding distance) but by expected individual benefit.

Module 3 (DML) complements the CATE pipeline with valid confidence intervals: rather than point estimates of impact, Chorus can communicate uncertainty-aware recommendations. DML's orthogonalization also helps when the confounding structure between "currently working in a domain" and "eventual impact" is complex and high-dimensional — exactly the situation with arXiv-scale scientific corpora.

Module 4 (PPI) is most relevant for Chorus evaluation: when ground-truth impact labels (citations, publication outcomes) are expensive to collect for emerging research combinations, PPI allows using LLM-generated "plausible impact" scores as unlabeled augmentation. This lets Chorus's evaluation pipeline scale without waiting years for citations to accumulate, while still guaranteeing valid statistical inference.


## Potential Final Project Integration

Module 1 (TARNet): Yes, qualifies as one of the "three types of models from three separate weeks." It can serve as the causal effect estimator in Chorus's DiscoveryAgent — given a recommended cross-domain pair (A=1, novel combination) vs baseline (A=0), TARNet estimates personalized CATE on citation impact. The architecture (shared representation + separate outcome heads) maps directly onto the problem of learning a domain-invariant embedding while maintaining distinct outcome models per treatment arm.

Module 3 (DML): Yes, qualifies as a separate week's model. DML with DoubleMLPLR or DoubleMLIRM can be used as a validation layer for Chorus recommendations: given a proposed "underexplored" domain pair, DML estimates the causal effect of entering that combination on impact outcomes with valid confidence intervals. Because DML works with any sklearn-compatible ML learner, it naturally accepts embedding features as covariates X.

Module 4 (PPI): Yes, qualifies as a third week's model. PPI is the ideal evaluation framework for Chorus: human expert judgments on recommendation quality are expensive (small n), but LLM-generated quality scores are cheap (large N). PPI corrects for LLM bias using the small human-labeled set, producing valid confidence intervals for Chorus's precision@k metric without requiring large human evaluation budgets.
