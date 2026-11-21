# Adult Income Prediction with MLPs (PyTorch)

An experimental study on the [UCI Adult](https://www.openml.org/d/1590) dataset (OpenML `adult`, version 2; 48,842 rows, 14 features), predicting whether a person earns more than 50K per year.

The goal was not only to train an MLP, but to understand **how data-handling choices affect it**: missing values, categorical encoding, class imbalance and oversampling.

**Primary metric: F1 of the positive class (>50K).** About 76% of samples belong to the negative class, so accuracy alone is misleading (a model that always predicts "<=50K" already scores ~76%). Accuracy, precision and recall are reported alongside.

## Questions explored

1. Does the way missing values are handled change the data and the model?
2. Do learned embeddings beat one-hot encoding for categorical features?
3. Does oversampling (SMOTENC) help with the imbalance, and how does the decision threshold interact with it?
4. How separable is this data in the first place?

## Setup

|Item|Value|
|---|---|
|Split|Stratified: 56.25% train / 18.75% validation / 25% test (`random_state=42`)|
|Numeric features|`StandardScaler` (fit on train only)|
|Categorical features|One-hot (MLP) or `OrdinalEncoder` + embeddings (EmbMLP)|
|Architecture|3 hidden layers (128 -> 64 -> 32), BatchNorm + ReLU + Dropout(0.2)|
|Optimizer|Adam, lr = 1e-3, batch size 512, up to 40 epochs|
|Early stopping|Patience 20 on validation loss, best weights restored|
|Class weighting|`BCEWithLogitsLoss(pos_weight = n_neg / (2·n_pos))` in the non-SMOTE runs; none in the SMOTENC runs|
|Seeds|`torch`, `numpy`, `random` = 42|

## Experiments

### 1. Missing values

Missing values appear in `workclass`, `occupation` and `native-country`. Two datasets were built:

- **Without missing values:** every row containing a missing value is dropped.
- **Missing handled:** a `crosstab` of each feature against the target guided the decision.
    - `workclass` and `occupation`: the missing group has its own income profile (for `occupation`, only ~9.4% earn >50K versus roughly 24% overall), so missingness carries information. It is kept as an explicit `unknown` category.
    - `native-country`: the missing group (857 rows) has a >50K rate of ~25.7%, close to the overall rate, so it adds little information. These rows are dropped.

FAMD projections of the two variants look broadly similar (component signs are arbitrary, so the mirrored layout is not a real difference), and the datasets differ by only a few thousand rows. Model results are almost identical.

### 2. Embeddings

Categorical features were also modelled with learned embeddings (`EmbMLP`) instead of one-hot vectors. It did not improve results on the same data and split.

### 3. SMOTENC

SMOTENC (which supports mixed numeric/categorical data) was applied **to the training set only**; validation and test sets keep the original distribution. The training prior becomes 50/50 while the real prior is ~24% positive, so the default threshold of 0.5 is not appropriate. Two thresholds were compared, judged on the train and validation results:

- **0.75:** roughly the accuracy-optimal threshold under this prior shift.
- **0.6:** a lower threshold that favours recall, which suits an F1 objective better. It gave the higher validation F1 and was chosen.

### 4. Separability

PCA and t-SNE on the numeric features show heavily overlapping classes, which is consistent with the performance plateau around 85% accuracy.

## Results

Non-SMOTE models use threshold 0.5. Train metrics are the last training epoch (dropout active); validation and test are computed on the full sets. Single run, seed 42.

**F1 across splits and test-set details**

|Model|Train F1|Valid F1|Test F1|Test Acc|Test Recall|Test Precision|
|---|---|---|---|---|---|---|
|MLP, without missing values|0.7305|0.6989|0.6967|0.8484|0.7027|0.6909|
|MLP, missing handled|0.7258|0.7041|0.6973|0.8485|0.7304|0.6671|
|EmbMLP, without missing values|0.7196|0.7008|0.6943|0.8438|0.7156|0.6742|
|MLP + SMOTENC (th = 0.6)|0.8717|0.6882|0.6822|0.8304|0.7348|0.6367|
|MLP + SMOTENC (th = 0.75)|0.8245|0.6691|0.6615|0.8457|0.6085|0.7246|

**Validation set**

|Model|Accuracy|F1|Recall|Precision|
|---|---|---|---|---|
|MLP, without missing values|0.8482|0.6989|0.7111|0.6872|
|MLP, missing handled|0.8513|0.7041|0.7405|0.6712|
|EmbMLP, without missing values|0.8461|0.7008|0.7273|0.6761|
|MLP + SMOTENC (th = 0.6)|0.8328|0.6882|0.7449|0.6396|
|MLP + SMOTENC (th = 0.75)|0.8482|0.6691|0.6192|0.7276|

## Key findings

- **Preprocessing choices barely mattered.** Missing-value handling and embeddings changed test F1 by less than 0.5 points, within the range a different seed could produce.
- **SMOTENC looked great on training data but not on held-out data.** Train F1 reached 0.82-0.87 against 0.66-0.69 on validation/test. Part of this is the shift in class distribution between the oversampled training set and the untouched evaluation sets (positive-class F1 and precision naturally rise when the positive class is 50% instead of 24%), and part is that synthetic samples are interpolations and easier to classify.
- **With SMOTENC the threshold matters a lot.** Moving from 0.75 to 0.6 raised test F1 from 0.6615 to 0.6822 (recall 0.61 -> 0.73) at the cost of accuracy (0.846 -> 0.830). No SMOTENC variant beat the plain MLPs on F1.
- **The data is hard to separate with these features.** All variants plateau around 85% accuracy and ~0.70 F1.

## Limitations

- Single run per configuration; differences below ~1 point should not be interpreted as real.
- Only two thresholds were compared for SMOTENC; no systematic threshold search.
- No unweighted baseline for the non-SMOTE models, and SMOTENC was not combined with `pos_weight` in the final notebook (an earlier exploratory run with `pos_weight` at th = 0.75 reached a test F1 of about 0.68).
- No ROC-AUC / PR-AUC and no tree-based baseline (e.g. XGBoost), so the "ceiling" claim is supported only by the MLP variants tested here.
- The datasets differ in row composition, so their test sets are not identical.
- t-SNE/PCA were computed on the numeric features only.

## Run it

```bash
pip install torch numpy pandas scikit-learn imbalanced-learn prince matplotlib tqdm
jupyter notebook adult_income_mlp.ipynb
```

The dataset is downloaded automatically via `fetch_openml("adult", version=2)`.