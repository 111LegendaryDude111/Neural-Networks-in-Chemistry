# Regression: CC50

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
- mae: `291.659883`
- rmse: `510.043973`
- r2: `0.498226`

## Leaderboard
| task_name       | problem_type   | target_column   | primary_metric   | evaluation_strategy   | model_name    | mode   |   fit_seconds | best_params_json                                                |     mae |   mae_std |    rmse |   rmse_std |        r2 |   r2_std |
|:----------------|:---------------|:----------------|:-----------------|:----------------------|:--------------|:-------|--------------:|:----------------------------------------------------------------|--------:|----------:|--------:|-----------:|----------:|---------:|
| regression_cc50 | regression     | CC50, mM        | mae              | holdout               | random_forest | direct |     14.3103   | {"max_depth": null, "max_features": 0.5, "min_samples_leaf": 2} | 291.66  |         0 | 510.044 |          0 |  0.498226 |        0 |
| regression_cc50 | regression     | CC50, mM        | mae              | holdout               | svr           | direct |      0.740137 | {"C": 1.0, "epsilon": 0.01, "gamma": "scale"}                   | 295.34  |         0 | 507.258 |          0 |  0.503693 |        0 |
| regression_cc50 | regression     | CC50, mM        | mae              | holdout               | ridge         | direct |      1.492    | {"alpha": 0.1}                                                  | 377.804 |         0 | 630.535 |          0 |  0.233149 |        0 |
| regression_cc50 | regression     | CC50, mM        | mae              | holdout               | dummy         | direct |      1.6999   | {"strategy": "median"}                                          | 507.246 |         0 | 762.639 |          0 | -0.121839 |        0 |

## Leakage Checklist
- Drop mandatory columns: Unnamed: 0, IC50, CC50, SI and actual target columns with units.
- Reject any feature whose normalized name contains standalone tokens ic50, cc50, si, or selectivity_index.
- Validate that feature matrix contains only numeric descriptor columns.
- Validate before every task and persist a leakage_check.json artifact.