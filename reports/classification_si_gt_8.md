# Classification: SI > 8

## Methodology
- Evaluation strategy: `holdout (20% test) + inner CV (3 folds)`
- Dataset path: `data/Данные_для_курсовои_Классическое_МО.xlsx`
- Dataset checksum (SHA256): `87fabec4e77159df7482ca480236712c031500d3935141b3231368d83a67c659`
- Feature count after leakage guard: `210`
- Leakage validation status: `passed`
- Primary metric: `pr_auc`
- Label balance: `{0: 644, 1: 357}`

## Winner
- Model: `random_forest`
- Mode: `direct`
- pr_auc: `0.682265`
- roc_auc: `0.755868`
- f1: `0.567164`
- balanced_accuracy: `0.670866`

## Leaderboard
| task_name              | problem_type   | target_column   | primary_metric   | evaluation_strategy   | model_name          | mode   |   fit_seconds | best_params_json                                                                          |   roc_auc |   roc_auc_std |       f1 |   f1_std |   balanced_accuracy |   balanced_accuracy_std |   pr_auc |   pr_auc_std |
|:-----------------------|:---------------|:----------------|:-----------------|:----------------------|:--------------------|:-------|--------------:|:------------------------------------------------------------------------------------------|----------:|--------------:|---------:|---------:|--------------------:|------------------------:|---------:|-------------:|
| classification_si_gt_8 | classification | SI              | pr_auc           | holdout               | random_forest       | direct |      22.3366  | {"class_weight": "balanced", "max_depth": 16, "max_features": 0.5, "min_samples_leaf": 5} |  0.755868 |             0 | 0.567164 |        0 |            0.670866 |                       0 | 0.682265 |            0 |
| classification_si_gt_8 | classification | SI              | pr_auc           | holdout               | svc                 | direct |       1.72943 | {"C": 1.0, "class_weight": null, "gamma": "auto"}                                         |  0.727121 |             0 | 0.52459  |        0 |            0.652455 |                       0 | 0.653251 |            0 |
| classification_si_gt_8 | classification | SI              | pr_auc           | holdout               | logistic_regression | direct |       1.10071 | {"C": 0.01, "class_weight": null}                                                         |  0.673503 |             0 | 0.503704 |        0 |            0.623708 |                       0 | 0.516753 |            0 |
| classification_si_gt_8 | classification | SI              | pr_auc           | holdout               | dummy               | direct |       2.00293 | {"strategy": "stratified"}                                                                |  0.548611 |             0 | 0.342857 |        0 |            0.496124 |                       0 | 0.384348 |            0 |

## Leakage Checklist
- Drop mandatory columns: Unnamed: 0, IC50, CC50, SI and actual target columns with units.
- Reject any feature whose normalized name contains standalone tokens ic50, cc50, si, or selectivity_index.
- Validate that feature matrix contains only numeric descriptor columns.
- Validate before every task and persist a leakage_check.json artifact.