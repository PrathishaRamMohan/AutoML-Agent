# Missing Values and Feature Encoding

## Imputation for numeric features
Median imputation is robust to outliers and is a safe default for skewed numeric features. Mean imputation is acceptable for roughly symmetric features. When more than about 40% of a column is missing, consider dropping it or adding a binary "was_missing" indicator, because the missingness itself can be informative.

Imputation must be fitted on training data only and then applied to validation and test data. Fitting the imputer on the full dataset leaks information from the test set. Wrapping the imputer inside a scikit-learn Pipeline guarantees this during cross-validation.

## Imputation for categorical features
Use the most frequent category, or a dedicated "Missing" category when missingness may be meaningful.

## One-hot encoding
One-hot encoding is the standard choice for low-cardinality nominal categories (fewer than about 20 levels). Set handle_unknown="ignore" so unseen categories at prediction time do not crash the model.

## Scaling
Linear models, SVMs, k-NN and neural networks are sensitive to feature scale, so standardise numeric features for them. Tree-based models (Random Forest, Gradient Boosting, XGBoost) are invariant to monotonic scaling and do not need it, although scaling does no harm.

## Date and time columns
Raw timestamps should not be used as-is. Extract components such as month, day of week, or "days since" a reference date, or drop them if they only record when the row was created.
