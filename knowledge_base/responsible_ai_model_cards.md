# Responsible AI and Model Cards

## Human oversight
Automated ML systems should keep a human in the loop before a model is deployed. The reviewer checks the evaluation evidence, the features used, and the intended use, and records an explicit approve or reject decision. Every decision should be logged with a timestamp and reviewer name for auditability.

## Model cards
A model card is a short document that accompanies a trained model. It typically contains:
- Model details: algorithm, version, training date, hyperparameters.
- Intended use: the task and the users it is designed for, and uses that are out of scope.
- Training data: source, size, preprocessing, and known limitations.
- Evaluation: metrics on held-out data, baseline comparison, and error analysis.
- Ethical considerations: potential biases, sensitive attributes, and fairness checks.
- Caveats and recommendations: when the model should be retrained or not trusted.

## Monitoring after deployment
Data drift (changes in the input distribution) and concept drift (changes in the relationship between inputs and target) degrade models over time. Track input statistics and live performance, and schedule periodic retraining.

## Sensitive attributes
Be careful with attributes such as age, gender, or location. Even when excluded, they may be reflected in correlated proxy features. Evaluate performance separately across groups where possible.
