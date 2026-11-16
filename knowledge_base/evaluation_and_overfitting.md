# Model Evaluation and Overfitting

## Hold-out test set
Split off a test set (typically 20%) at the very beginning and do not touch it until final evaluation. All preprocessing, model selection and tuning must use only the training portion. The test score is then an honest estimate of performance on new data.

## Detecting overfitting
Compare the training score with the test score. A gap larger than about 0.10 (for metrics on a 0 to 1 scale such as ROC-AUC, F1 or accuracy) indicates overfitting: the model has memorised noise in the training data. Remedies include stronger regularisation, shallower trees, fewer features, more data, and early stopping.

## Underfitting
If both training and test scores are low and close to the baseline, the model is underfitting. Remedies include more expressive models, better features, and less regularisation.

## Confusion matrix
For classification, the confusion matrix shows true positives, false positives, true negatives and false negatives. It reveals whether errors are concentrated in one class, which aggregated metrics can hide.

## Feature importance
Permutation importance measures how much the test score drops when a feature's values are randomly shuffled. It is model-agnostic and computed on held-out data, so it reflects what the model actually relies on. Impurity-based importances from trees are biased towards high-cardinality features.

## Deployment readiness checklist
- The model beats the baseline by a meaningful margin.
- No sign of leakage (suspiciously perfect scores).
- Train/test gap is acceptable.
- Important features make domain sense.
- A human reviewer has approved the model and the evaluation report.
