# Regression: IC50

## Methodology
- Evaluation strategy: `holdout (20% test) + inner CV (3 folds)`
- Dataset path: `data/Данные_для_курсовои_Классическое_МО.xlsx`
- Dataset checksum (SHA256): `87fabec4e77159df7482ca480236712c031500d3935141b3231368d83a67c659`
- Feature count after leakage guard: `210`
- Leakage validation status: `passed`
- Primary metric: `mae`
- Target log1p transform: `True`

## Winner
- Model: `random_forest`
- Mode: `direct`
- mae: `219.159926`
- rmse: `478.325154`
- r2: `0.314080`

## Leaderboard
| task_name       | problem_type   | target_column   | primary_metric   | evaluation_strategy   | model_name    | mode   |   fit_seconds | best_params_json                                                |     mae |   mae_std |    rmse |   rmse_std |        r2 |   r2_std |
|:----------------|:---------------|:----------------|:-----------------|:----------------------|:--------------|:-------|--------------:|:----------------------------------------------------------------|--------:|----------:|--------:|-----------:|----------:|---------:|
| regression_ic50 | regression     | IC50, mM        | mae              | holdout               | random_forest | direct |     14.279    | {"max_depth": null, "max_features": 0.5, "min_samples_leaf": 1} | 219.16  |         0 | 478.325 |          0 |  0.31408  |        0 |
| regression_ic50 | regression     | IC50, mM        | mae              | holdout               | svr           | direct |      0.802243 | {"C": 1.0, "epsilon": 0.01, "gamma": "scale"}                   | 230.241 |         0 | 517.825 |          0 |  0.196116 |        0 |
| regression_ic50 | regression     | IC50, mM        | mae              | holdout               | ridge         | direct |      1.53466  | {"alpha": 1.0}                                                  | 232.612 |         0 | 508.934 |          0 |  0.223484 |        0 |
| regression_ic50 | regression     | IC50, mM        | mae              | holdout               | dummy         | direct |      1.7336   | {"strategy": "median"}                                          | 278.881 |         0 | 629.994 |          0 | -0.189871 |        0 |

## Leakage Checklist
- Drop mandatory columns: Unnamed: 0, IC50, CC50, SI and actual target columns with units.
- Reject any feature whose normalized name contains standalone tokens ic50, cc50, si, or selectivity_index.
- Validate that feature matrix contains only numeric descriptor columns.
- Validate before every task and persist a leakage_check.json artifact.