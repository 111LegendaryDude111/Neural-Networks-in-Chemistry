# Chemical Activity Prediction Pipeline

В репозитории реализован воспроизводимый ML pipeline для прогноза эффективности химических соединений по дескрипторам. Source of truth для методологии: `PDR.md` и `PDR-tasks.md`.

## Что решается

- Регрессия: `IC50`, `CC50`, `SI`
- Классификация:
  - `IC50 > 46.585183`
  - `CC50 > 411.039342`
  - `SI > 3.846154`
  - `SI > 8`

## Data Contract

- Dataset path: `data/Данные_для_курсовои_Классическое_МО.xlsx`
- SHA256: `87fabec4e77159df7482ca480236712c031500d3935141b3231368d83a67c659`
- Shape: `1001 x 214`
- Descriptor features after leakage filtering: `210`
- Missing values: `36`
- Duplicates: `0`
- Target columns: `IC50, mM`, `CC50, mM`, `SI`
- Fixed thresholds:
  - `IC50 > 46.585183`
  - `CC50 > 411.039342`
  - `SI > 3.846154`
  - `SI > 8`

## Leakage Policy

- В `X` попадают только дескрипторы.
- Запрещённые колонки централизованы в коде: `Unnamed: 0`, `IC50`, `CC50`, `SI`, `IC50, mM`, `CC50, mM`.
- Любые прямые производные targets тоже режутся регулярным anti-leakage guard по нормализованным именам колонок.
- Перед каждой задачей выполняется явная валидация feature matrix и сохраняется `leakage_check.json`.

## Методология

- Regression metrics: `MAE` как primary, дополнительно `RMSE`, `R²`
- Classification metrics: `ROC-AUC`, `F1`, `Balanced Accuracy`
- Для `SI > 8` дополнительно считается и сохраняется `PR-AUC`, он используется как primary metric
- Предпочтительная стратегия: nested CV `5x3`
- Fallback для быстрых прогонов: holdout `20%` + inner CV `3-fold`
- Из-за сильной положительной асимметрии targets в регрессии включён прозрачный `log1p`-target transform
- Для `SI` реализованы оба режима:
  - `direct`: `SI ~ descriptors`
  - `indirect`: `SI_hat = CC50_hat / IC50_hat`

## Структура

```text
.
├── PDR.md
├── PDR-tasks.md
├── README.md
├── requirements.txt
├── data/
├── reports/
├── results/
└── src/
    ├── common/
    ├── eda.py
    ├── regression_ic50.py
    ├── regression_cc50.py
    ├── regression_si.py
    ├── classification_ic50_gt_median.py
    ├── classification_cc50_gt_median.py
    ├── classification_si_gt_median.py
    ├── classification_si_gt_8.py
    └── comparison.py
```

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Если `catboost` не устанавливается, пайплайн не блокируется:

```bash
python -m pip install -r requirements.txt --no-deps
```

и затем все task scripts можно запускать с флагом `--skip-catboost`.

## Запуск

EDA:

```bash
python -m src.eda
```

Регрессия:

```bash
python -m src.regression_ic50
python -m src.regression_cc50
python -m src.regression_si
```

Классификация:

```bash
python -m src.classification_ic50_gt_median
python -m src.classification_cc50_gt_median
python -m src.classification_si_gt_median
python -m src.classification_si_gt_8
```

Сводное сравнение и финальный summary:

```bash
python -m src.comparison
```

### Полезные флаги

Полный протокол:

```bash
python -m src.regression_ic50 --evaluation-strategy nested
```

Быстрый fallback:

```bash
python -m src.regression_ic50 --evaluation-strategy holdout
```

Отключить CatBoost:

```bash
python -m src.classification_si_gt_8 --skip-catboost
```

Запустить только часть моделей:

```bash
python -m src.regression_cc50 --models dummy ridge svr random_forest
```

## Артефакты

- `results/data_contract.json` — зафиксированный data contract
- `results/<task>/leaderboard.csv` — таблица моделей по задаче
- `results/<task>/*_fold_metrics.csv` — CV/holdout метрики по модели
- `results/<task>/winner_confusion_matrix.png` — confusion matrix для задач классификации
- `results/<task>/winner_feature_importance*.csv` — importances / coefficients для победителя, если модель это поддерживает
- `reports/eda_summary.md` — текстовые выводы по EDA
- `reports/<task>.md` — выводы по каждой задаче
- `reports/project_summary.md` — итоговый summary по проекту

## Ограничения и Known Issues

- Dataset маленький, поэтому variance метрик между сплитами может быть заметным.
- `SI` имеет очень сильный skew; direct vs indirect нужно сравнивать только на одинаковой evaluation strategy.
- `CatBoost` опционален: если пакет отсутствует, модели автоматически можно отключить через `--skip-catboost`.
- Репозиторий не публикует raw data в артефактах; сохраняются только агрегированные таблицы, метрики и графики.

