# Classification: SI > 3.846154

## Methodology
- Evaluation strategy: `holdout (20% test) + inner CV (3 folds)`
- Dataset path: `data/Данные_для_курсовои_Классическое_МО.xlsx`
- Dataset checksum (SHA256): `87fabec4e77159df7482ca480236712c031500d3935141b3231368d83a67c659`
- Feature count after leakage guard: `210`
- Leakage validation status: `passed`
- Primary metric: `roc_auc`
- Label balance: `{0: 501, 1: 500}`

## Winner
- Model: `svc`
- Mode: `direct`
- roc_auc: `0.727178`
- f1: `0.655914`
- balanced_accuracy: `0.681238`

## Leaderboard
| task_name                   | problem_type   | target_column   | primary_metric   | evaluation_strategy   | model_name          | mode   |   fit_seconds | best_params_json                                                                            |   roc_auc |   roc_auc_std |       f1 |   f1_std |   balanced_accuracy |   balanced_accuracy_std |
|:----------------------------|:---------------|:----------------|:-----------------|:----------------------|:--------------------|:-------|--------------:|:--------------------------------------------------------------------------------------------|----------:|--------------:|---------:|---------:|--------------------:|------------------------:|
| classification_si_gt_median | classification | SI              | roc_auc          | holdout               | svc                 | direct |       1.75633 | {"C": 1.0, "class_weight": "balanced", "gamma": "scale"}                                    |  0.727178 |             0 | 0.655914 |        0 |            0.681238 |                       0 |
| classification_si_gt_median | classification | SI              | roc_auc          | holdout               | random_forest       | direct |      22.2869  | {"class_weight": "balanced", "max_depth": 8, "max_features": "sqrt", "min_samples_leaf": 5} |  0.710446 |             0 | 0.629834 |        0 |            0.666188 |                       0 |
| classification_si_gt_median | classification | SI              | roc_auc          | holdout               | logistic_regression | direct |       1.03852 | {"C": 0.1, "class_weight": "balanced"}                                                      |  0.658416 |             0 | 0.619289 |        0 |            0.626782 |                       0 |
| classification_si_gt_median | classification | SI              | roc_auc          | holdout               | dummy               | direct |       2.09514 | {"strategy": "stratified"}                                                                  |  0.497871 |             0 | 0.509804 |        0 |            0.502574 |                       0 |

## Leakage Checklist
- Drop mandatory columns: Unnamed: 0, IC50, CC50, SI and actual target columns with units.
- Reject any feature whose normalized name contains standalone tokens ic50, cc50, si, or selectivity_index.
- Validate that feature matrix contains only numeric descriptor columns.
- Validate before every task and persist a leakage_check.json artifact.