# Hyperparameter Tuning

## Bayesian optimisation with Optuna
Optuna uses the Tree-structured Parzen Estimator (TPE) to propose promising hyperparameters based on earlier trials. It usually finds good configurations in far fewer trials than grid search. Each trial should be scored with cross-validation on the training set only; the held-out test set must never be used during tuning.

## XGBoost search space
- n_estimators: 100 to 600
- max_depth: 2 to 8 (shallower trees reduce overfitting)
- learning_rate: 0.01 to 0.3, log scale
- subsample and colsample_bytree: 0.6 to 1.0 (row and column subsampling add randomness and reduce variance)
- min_child_weight: 1 to 10 (larger values make the model more conservative)
- reg_lambda (L2) and reg_alpha (L1): 1e-3 to 10, log scale

## Random Forest search space
- n_estimators: 100 to 500
- max_depth: 3 to 20 or None
- min_samples_leaf: 1 to 10 (larger values smooth predictions)
- max_features: "sqrt", "log2", or a fraction

## Logistic Regression search space
- C: 1e-3 to 100, log scale
- penalty: L2 is the safe default

## Regularised tuning when overfitting is detected
If the gap between training and test scores is large, shrink the search space towards simpler models: lower max_depth (2 to 4), higher min_child_weight or min_samples_leaf, stronger reg_lambda, lower learning_rate with more trees, and subsample below 0.9. Re-tune and re-evaluate.

## Budget
25 to 50 trials is usually enough for small tabular problems. Use a timeout to cap total tuning time.
