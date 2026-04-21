# ML для химических данных

В репозитории оставлено только то, что нужно для решения задания:

- `data/data.xlsx` — исходные данные
- `notebooks/eda.ipynb` — отдельный EDA
- `notebooks/regression_ic50.ipynb`
- `notebooks/regression_cc50.ipynb`
- `notebooks/regression_si.ipynb`
- `notebooks/classification_ic50_gt_median.ipynb`
- `notebooks/classification_cc50_gt_median.ipynb`
- `notebooks/classification_si_gt_median.ipynb`
- `notebooks/classification_si_gt_8.ipynb`
- `src/common/preprocessing.py` — единый источник всей логики препроцессинга
- `src/common/training.py` — общий код для сравнения моделей и сохранения артефактов
- `src/eda.py` — пересчёт таблиц и графиков для EDA
- `src/comparison.py` — сводка по уже рассчитанным задачам

Старые служебные документы, кэш, готовые артефакты и промежуточные отчёты удалены.

## Как устроена работа

Каждый ноутбук соответствует одной задаче и запускается сверху вниз без ручной подготовки:

- сам находит корень проекта;
- подключает `src/`;
- использует общий модуль препроцессинга;
- запускает сравнение нескольких моделей;
- пишет локальные артефакты в `results/` и `reports/`, если они нужны для текущего прогона.

EDA вынесен в отдельный ноутбук: [eda.ipynb](/Users/davidsukhashvili/Desktop/ML/MEPhi/ClassicMl/Neural%20Networks%20in%20Chemistry/notebooks/eda.ipynb).

## Почему препроцессинг в одном месте

Весь отбор признаков, защита от утечек, построение `X` и `y`, импутация, масштабирование и `log1p` для регрессии собраны в одном файле:

[src/common/preprocessing.py](/Users/davidsukhashvili/Desktop/ML/MEPhi/ClassicMl/Neural%20Networks%20in%20Chemistry/src/common/preprocessing.py)

Так проще проверять корректность и поддерживать единое поведение во всех задачах.

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`ipykernel` уже включён в зависимости, поэтому ноутбуки можно сразу запускать через JupyterLab, Jupyter Notebook или VS Code.

## Что запускать

Основной сценарий:

1. Открыть `notebooks/eda.ipynb`.
2. Затем последовательно запускать ноутбуки по задачам.
3. После расчётов при необходимости собрать сводку:

```bash
python -m src.comparison
```

Если нужно пересчитать только EDA без ноутбука:

```bash
python -m src.eda
```

## Что не хранится в репозитории

По умолчанию не хранятся:

- `results/`
- `reports/`
- `__pycache__/`
- локальное окружение `.venv/`

Эти файлы и каталоги создаются локально по ходу работы и не нужны для самой реализации.
