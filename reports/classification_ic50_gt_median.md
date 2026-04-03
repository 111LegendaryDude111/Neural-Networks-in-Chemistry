# Classification: IC50 > 46.585183

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
- roc_auc: `0.785990`
- f1: `0.747664`
- balanced_accuracy: `0.731040`

## Leaderboard
| task_name                     | problem_type   | target_column   | primary_metric   | evaluation_strategy   | model_name          | mode   |   fit_seconds | best_params_json                                                                      |   roc_auc |   roc_auc_std |       f1 |   f1_std |   balanced_accuracy |   balanced_accuracy_std |
|:------------------------------|:---------------|:----------------|:-----------------|:----------------------|:--------------------|:-------|--------------:|:--------------------------------------------------------------------------------------|----------:|--------------:|---------:|---------:|--------------------:|------------------------:|
| classification_ic50_gt_median | classification | IC50, mM        | roc_auc          | holdout               | random_forest       | direct |      20.722   | {"class_weight": null, "max_depth": 8, "max_features": "sqrt", "min_samples_leaf": 1} |  0.78599  |             0 | 0.747664 |        0 |            0.73104  |                       0 |
| classification_ic50_gt_median | classification | IC50, mM        | roc_auc          | holdout               | svc                 | direct |       1.77824 | {"C": 1.0, "class_weight": null, "gamma": "scale"}                                    |  0.778168 |             0 | 0.727273 |        0 |            0.716238 |                       0 |
| classification_ic50_gt_median | classification | IC50, mM        | roc_auc          | holdout               | logistic_regression | direct |       6.22268 | {"C": 0.1, "class_weight": null, "penalty": "l2"}                                     |  0.747327 |             0 | 0.702439 |        0 |            0.696436 |                       0 |
| classification_ic50_gt_median | classification | IC50, mM        | roc_auc          | holdout               | dummy               | direct |       2.03425 | {"strategy": "stratified"}                                                            |  0.462673 |             0 | 0.467662 |        0 |            0.467673 |                       0 |

## Leakage Checklist
- Drop mandatory columns: Unnamed: 0, IC50, CC50, SI and actual target columns with units.
- Reject any feature whose normalized name contains standalone tokens ic50, cc50, si, or selectivity_index.
- Validate that feature matrix contains only numeric descriptor columns.
- Validate before every task and persist a leakage_check.json artifact.