# Regression: SI (direct vs indirect)

## Methodology
- Evaluation strategy: `holdout (20% test) + inner CV (3 folds)`
- Dataset path: `data/Данные_для_курсовои_Классическое_МО.xlsx`
- Leakage validation status: `passed`
- Direct model: `SI ~ descriptors`
- Indirect model: `SI_hat = CC50_hat / IC50_hat`
- Target log1p transform on component regressions: `True`

## Direct vs Indirect
- Best direct model: `random_forest` with `MAE=178.367353`
- Best indirect model: `random_forest` with `MAE=177.646524`
- Overall winner: `indirect::random_forest`

## Combined Leaderboard
| task_name     | problem_type   | target_column   | primary_metric   | evaluation_strategy   | model_name    | mode     |   fit_seconds | best_params_json                                                                                                                                                       |           mae |   mae_std |           rmse |   rmse_std |           r2 |   r2_std |
|:--------------|:---------------|:----------------|:-----------------|:----------------------|:--------------|:---------|--------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------:|----------:|---------------:|-----------:|-------------:|---------:|
| regression_si | regression     | SI              | mae              | holdout               | random_forest | indirect |     30.1022   | {"cc50_component": {"max_depth": null, "max_features": 0.5, "min_samples_leaf": 1}, "ic50_component": {"max_depth": null, "max_features": 0.5, "min_samples_leaf": 1}} | 177.647       |         0 | 1421.37        |          0 | -0.00577561  |        0 |
| regression_si | regression     | SI              | mae              | holdout               | random_forest | direct   |     11.6475   | {"max_depth": null, "max_features": "sqrt", "min_samples_leaf": 2}                                                                                                     | 178.367       |         0 | 1418.41        |          0 | -0.0016022   |        0 |
| regression_si | regression     | SI              | mae              | holdout               | svr           | direct   |      0.765084 | {"C": 1.0, "epsilon": 0.01, "gamma": "scale"}                                                                                                                          | 179.12        |         0 | 1424.78        |          0 | -0.0106163   |        0 |
| regression_si | regression     | SI              | mae              | holdout               | ridge         | direct   |      0.820911 | {"alpha": 100.0}                                                                                                                                                       | 179.923       |         0 | 1424.98        |          0 | -0.0108914   |        0 |
| regression_si | regression     | SI              | mae              | holdout               | dummy         | direct   |      1.74191  | {"strategy": "median"}                                                                                                                                                 | 180.856       |         0 | 1428.46        |          0 | -0.0158427   |        0 |
| regression_si | regression     | SI              | mae              | holdout               | dummy         | indirect |      1.60097  | {"cc50_component": {"strategy": "median"}, "ic50_component": {"strategy": "median"}}                                                                                   | 181.954       |         0 | 1427.89        |          0 | -0.0150313   |        0 |
| regression_si | regression     | SI              | mae              | holdout               | ridge         | indirect |      0.2258   | {"cc50_component": {"alpha": 10.0}, "ic50_component": {"alpha": 1.0}}                                                                                                  |   4.79548e+06 |         0 |    6.79851e+07 |          0 | -2.301e+09   |        0 |
| regression_si | regression     | SI              | mae              | holdout               | svr           | indirect |      1.66383  | {"cc50_component": {"C": 1.0, "epsilon": 0.01, "gamma": "scale"}, "ic50_component": {"C": 1.0, "epsilon": 0.01, "gamma": "scale"}}                                     |   1.40751e+07 |         0 |    1.99547e+08 |          0 | -1.98234e+10 |        0 |

## Leakage Checklist
- Drop mandatory columns: Unnamed: 0, IC50, CC50, SI and actual target columns with units.
- Reject any feature whose normalized name contains standalone tokens ic50, cc50, si, or selectivity_index.
- Validate that feature matrix contains only numeric descriptor columns.
- Validate before every task and persist a leakage_check.json artifact.