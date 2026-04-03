# Classification: CC50 > 411.039342

## Methodology
- Evaluation strategy: `holdout (20% test) + inner CV (3 folds)`
- Dataset path: `data/Данные_для_курсовои_Классическое_МО.xlsx`
- Dataset checksum (SHA256): `87fabec4e77159df7482ca480236712c031500d3935141b3231368d83a67c659`
- Feature count after leakage guard: `210`
- Leakage validation status: `passed`
- Primary metric: `roc_auc`
- Label balance: `{0: 500, 1: 501}`

## Winner
- Model: `random_forest`
- Mode: `direct`
- roc_auc: `0.854455`
- f1: `0.764151`
- balanced_accuracy: `0.750990`

## Leaderboard
| task_name                     | problem_type   | target_column   | primary_metric   | evaluation_strategy   | model_name          | mode   |   fit_seconds | best_params_json                                                                         |   roc_auc |   roc_auc_std |       f1 |   f1_std |   balanced_accuracy |   balanced_accuracy_std |
|:------------------------------|:---------------|:----------------|:-----------------|:----------------------|:--------------------|:-------|--------------:|:-----------------------------------------------------------------------------------------|----------:|--------------:|---------:|---------:|--------------------:|------------------------:|
| classification_cc50_gt_median | classification | CC50, mM        | roc_auc          | holdout               | random_forest       | direct |      21.2458  | {"class_weight": "balanced", "max_depth": 8, "max_features": 0.5, "min_samples_leaf": 1} |  0.854455 |             0 | 0.764151 |        0 |            0.75099  |                       0 |
| classification_cc50_gt_median | classification | CC50, mM        | roc_auc          | holdout               | logistic_regression | direct |       1.13778 | {"C": 0.1, "class_weight": null}                                                         |  0.837327 |             0 | 0.75576  |        0 |            0.735941 |                       0 |
| classification_cc50_gt_median | classification | CC50, mM        | roc_auc          | holdout               | svc                 | direct |       1.67704 | {"C": 1.0, "class_weight": null, "gamma": "scale"}                                       |  0.832871 |             0 | 0.779343 |        0 |            0.765891 |                       0 |
| classification_cc50_gt_median | classification | CC50, mM        | roc_auc          | holdout               | dummy               | direct |       2.04068 | {"strategy": "stratified"}                                                               |  0.447525 |             0 | 0.435644 |        0 |            0.432822 |                       0 |

## Leakage Checklist
- Drop mandatory columns: Unnamed: 0, IC50, CC50, SI and actual target columns with units.
- Reject any feature whose normalized name contains standalone tokens ic50, cc50, si, or selectivity_index.
- Validate that feature matrix contains only numeric descriptor columns.
- Validate before every task and persist a leakage_check.json artifact.