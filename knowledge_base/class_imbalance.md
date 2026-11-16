# Handling Class Imbalance

## Detecting imbalance
Compute the ratio between the majority and minority class counts. A ratio above about 3:1 is moderately imbalanced; above 10:1 is severely imbalanced. With imbalance, plain accuracy is misleading: a model that always predicts the majority class can score high accuracy while being useless.

## Metrics to use
For imbalanced binary problems prefer ROC-AUC, PR-AUC (average precision), F1-score, balanced accuracy, and recall on the minority class. Always compare against a majority-class baseline (DummyClassifier).

## Class weighting
Most scikit-learn classifiers accept class_weight="balanced", which reweights samples inversely to class frequency. XGBoost uses scale_pos_weight, typically set to (number of negatives / number of positives) for binary problems. Class weighting is simple, does not change the data, and is usually the first thing to try.

## Resampling
Random oversampling of the minority class, undersampling of the majority class, or synthetic methods such as SMOTE can help. Resampling must be applied only inside the training folds, never before the train/test split, otherwise the evaluation is optimistic.

## Stratified splitting
Use stratified train/test splits and StratifiedKFold cross-validation so every fold keeps the original class proportions.

## Decision threshold
The default 0.5 probability threshold is rarely optimal for imbalanced data. Tune the threshold on validation data to balance precision and recall according to business cost.
