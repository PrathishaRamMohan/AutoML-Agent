# Data Quality Checks Before Modelling

## Identifier columns
Columns whose values are unique for every row (customer IDs, order numbers, UUIDs) carry no generalisable signal. A model can memorise them and appear to perform well in training while failing on new data. Drop ID-like columns before training. A practical rule: if the number of unique values equals the number of rows and the column is not a continuous measurement, treat it as an identifier.

## Constant and near-constant columns
A column with a single unique value has zero variance and cannot help any model. Near-constant columns (one value in more than 99% of rows) rarely help and can destabilise some estimators. Remove them during preprocessing.

## Duplicate rows
Exact duplicate rows inflate cross-validation scores when copies land in both training and validation folds. Report the duplicate count and consider de-duplicating before splitting.

## Data leakage
Data leakage happens when a feature contains information that would not be available at prediction time, usually because it was recorded after the outcome occurred. Typical examples are "account_closed" when predicting churn, "refund_issued" when predicting fraud, or "discharge_date" when predicting hospital admission.

Warning signs of leakage:
- A single feature achieves near-perfect accuracy or ROC-AUC (above 0.97) on its own.
- The absolute correlation between a numeric feature and the target is above 0.95.
- Model performance is far better than domain experts would expect.

The correct action is to remove the leaking feature and to confirm with a domain expert when it is recorded. Never keep a leaking feature just because it improves validation scores; the deployed model will fail.

## High-cardinality categorical columns
Categorical columns with many unique values (more than about 50) create very wide one-hot matrices. Options are frequency encoding, target encoding with cross-fitting, grouping rare levels into "Other", or dropping the column if it is ID-like.
