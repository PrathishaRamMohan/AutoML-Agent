# Model Selection for Tabular Data

## Start with baselines
Always compare against a trivial baseline: DummyClassifier (most frequent class) for classification and DummyRegressor (mean) for regression. A model that does not clearly beat the baseline has not learned anything useful.

## Logistic Regression / Linear Regression
Fast, interpretable, and a strong baseline when relationships are roughly linear. Needs scaled features and one-hot encoding. Regularisation strength C controls overfitting (smaller C means stronger regularisation).

## Random Forest
An ensemble of decorrelated decision trees trained on bootstrap samples. Robust to outliers and unscaled features, handles nonlinear interactions, and rarely overfits badly. Key hyperparameters: n_estimators, max_depth, min_samples_leaf, max_features.

## Gradient Boosting and XGBoost
Boosting builds trees sequentially, each correcting the errors of the previous ones. XGBoost adds regularisation (L1/L2 on leaf weights), efficient histogram-based training, and native handling of missing values. On structured tabular data, gradient-boosted trees are usually the strongest off-the-shelf models. They can overfit if trees are too deep or the learning rate is too high.

## Comparing models fairly
Use the same cross-validation folds and the same preprocessing pipeline for every candidate. Report the mean and standard deviation across folds. Prefer the simpler model when scores are within one standard deviation of each other (Occam's razor), because simpler models are easier to explain and maintain.

## Choosing the metric
Binary classification: ROC-AUC or F1. Multiclass: macro F1 or balanced accuracy. Regression: RMSE, MAE, or R². Choose the metric before looking at results to avoid cherry-picking.
