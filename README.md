# ML для химических данных

Проект с EDA и отдельными ноутбуками для 7 задач:
- регрессия `IC50`
- регрессия `CC50`
- регрессия `SI`
- классификация `IC50 > медианы`
- классификация `CC50 > медианы`
- классификация `SI > медианы`
- классификация `SI > 8`

## Структура проекта

- `data/data.xlsx` — исходные данные
- `notebooks/eda.ipynb` — разведочный анализ данных
- `notebooks/regression_ic50.ipynb`
- `notebooks/regression_cc50.ipynb`
- `notebooks/regression_si.ipynb`
- `notebooks/classification_ic50_gt_median.ipynb`
- `notebooks/classification_cc50_gt_median.ipynb`
- `notebooks/classification_si_gt_median.ipynb`
- `notebooks/classification_si_gt_8.ipynb`
- `src/common/preprocessing.py` — общая подготовка данных
- `src/common/training.py` — общий код для обучения и сравнения моделей
- `src/eda.py` — пересчёт EDA из терминала
- `src/comparison.py` — сводка по уже посчитанным задачам
- `results/` — локальные результаты запусков
- `reports/` — локальные текстовые отчёты

## Как запустить

1. Создать виртуальное окружение:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Установить зависимости:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

3. Открыть ноутбуки в Jupyter или VS Code.

4. Сначала запустить `notebooks/eda.ipynb`.

5. Потом запускать нужные ноутбуки по задачам сверху вниз.