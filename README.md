# Multi-Agent System for Kazakhstan Energy Consumption Prediction Using Machine Learning

Учебный проект Assignment 1: регрессия почасового спроса на электроэнергию в узлах сети Казахстана. Основной ML запуск не требует LLM или API-ключа.

## Данные

- **Источник:** Makpal Assembayeva, Jonas Egerer, Roman Mendelevitch, Nurkhat Zhakiyev, *Spatial electricity market data for the power system of Kazakhstan*, Data in Brief (2019), [DOI](https://doi.org/10.1016/j.dib.2019.103781). [Страница набора и исходный Excel](https://edoc.hu-berlin.de/items/95956d1c-7161-4942-ade0-5621a72c1233), [прямая ссылка на Excel](https://edoc.hu-berlin.de/server/api/core/bitstreams/54874ce7-e732-45bc-a169-726f01851db2/content).
- **Исходный файл:** `data/source_kazakhstan_power_system.xlsx`, вкладка `DEMAND`. `src/data.py` преобразует 336 почасовых записей для 33 узлов сети в `data/energy.csv`: **11 088 строк**.
- **Target:** `demand` — почасовой спрос конкретного узла в числовой шкале исходной таблицы. В исходной вкладке единица измерения явно не подписана, поэтому результаты здесь не обозначаются как кВт·ч или МВт.
- **5 features:** `week` (1 или 2), `hour_in_week` (1–168), `hour_of_day` (1–24), `day_in_week` (1–7), `node` (код узла). Последние два временных признака вычисляются только из номера часа. Числовые признаки проходят импутацию и масштабирование, `node` — импутацию и one-hot encoding. Преобразования обучаются внутри каждого CV fold.
- **Происхождение:** это опубликованный обработанный исследовательский набор на основе статистики Казахстана; узловой спрос был частично **распределён и рассчитан авторами** из национального профиля, региональных итогов и допущений. Это не архив прямых показаний счётчиков по узлам. Мы не генерируем дополнительные наблюдения. Данные охватывают только две репрезентативные недели, поэтому модель нельзя считать прогнозом произвольного будущего дня.

Проверен также [показатель Бюро национальной статистики «Total final consumption»](https://stat.gov.kz/en/open-data/energy/476591/) с [CSV-выгрузкой](https://stat.gov.kz/api/iblock/element/45240/csv/file/en/). В ней 35 годовых значений за 1991–2025 годы. Строки по отраслям являются составными частями целевого показателя: использование их как признаков создало бы прямую утечку. Без них данных недостаточно для осмысленного сравнения 11 моделей с 10-fold CV. Поэтому используется опубликованная таблица спроса по узлам.

## Алгоритмы и проверка

Ridge Regression, Lasso Regression, Elastic Net, K-Nearest Neighbors Regression, Extra Trees Regression, AdaBoost Regression, Gradient Boosting Regression, XGBoost Regression, LightGBM Regression, CatBoost Regression, HistGradientBoosting Regression.

Для всех моделей используется `KFold(n_splits=10, shuffle=True, random_state=42)` и `cross_validate()` с RMSE и R². Table 1 автоматически сохраняется в `results/model_results.csv`, стандартные отклонения — в `results/model_results_detailed.csv`. Победитель выбирается по минимальному среднему RMSE, при равенстве — по максимальному R². Полный Pipeline обучается на всех данных и сохраняется в `models/best_model.joblib`.

**Ограничение оценки:** строки соседних часов и двух недель зависимы. Требуемый заданием случайный KFold перемешивает эти строки, поэтому высокие R² нельзя переносить на будущие периоды или независимые недели. Для реального прогноза понадобятся более длинные измеренные ряды и временная проверка; это вне рамок указанного assignment.

## Установка и запуск

Python 3.12 рекомендуется. В PowerShell из корня репозитория:

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.venv\Scripts\python.exe main.py
```

После активации окружения работает требуемая команда `python main.py`. Она загружает исходную таблицу, выводит `head`, `shape`, `info`, `describe`, пропуски и дубликаты; обучает 11 моделей, печатает Table 1, сохраняет лучшую модель и делает одно предсказание. Графики записываются в `results/plots/`.

Пример отдельного предсказания после обучения:

```powershell
.venv\Scripts\python.exe -c "from src.prediction import predict_energy_consumption; print(predict_energy_consumption({'week': 1, 'hour_in_week': 1, 'node': 'N0000'}))"
```

## Результаты Table 1

Получены локальным запуском `main.py`; полный точный CSV находится в `results/model_results.csv`.

| Algorithm | Number of Features | Number of Targets | K-Fold Validation | RMSE | R² |
|---|---:|---:|---:|---:|---:|
| Extra Trees Regression | 5 | 1 | 10 | 16.518 | 0.997528 |
| XGBoost | 5 | 1 | 10 | 17.398 | 0.997213 |
| CatBoost | 5 | 1 | 10 | 19.228 | 0.996618 |
| HistGradientBoosting | 5 | 1 | 10 | 19.849 | 0.996398 |
| LightGBM | 5 | 1 | 10 | 20.092 | 0.996314 |
| KNN Regression | 5 | 1 | 10 | 20.344 | 0.996233 |
| Ridge | 5 | 1 | 10 | 52.160 | 0.975123 |
| Gradient Boosting Regression | 5 | 1 | 10 | 57.468 | 0.969756 |
| Lasso | 5 | 1 | 10 | 61.092 | 0.965840 |
| AdaBoost Regression | 5 | 1 | 10 | 147.087 | 0.801880 |
| Elastic Net | 5 | 1 | 10 | 312.247 | 0.119097 |

Лучшая модель — Extra Trees Regression. Пример `{week: 1, hour_in_week: 1, node: N0000}` дал `54.794` в шкале исходной таблицы после повторной загрузки Pipeline.

## Multi-Agent система

`src/agents.py` использует установленный `deepagents` API `create_deep_agent()`. Главный Deep Agent делегирует задачи четырём подагентам:

```text
Main Deep Agent
├── data_agent          → загрузка, анализ и подготовка данных
├── training_agent      → 11 моделей и 10-fold CV
├── evaluation_agent    → Table 1, лучший Pipeline
└── prediction_agent    → проверка входа и вызов сохранённой ML модели
```

Каждый подагент вызывает Python tools; LLM не рассчитывает метрики и предсказания. Для интерактивной демонстрации скопируйте `.env.example` в `.env`, установите `MODEL_PROVIDER`, `MODEL_NAME` и ключ выбранного провайдера (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY` либо `GOOGLE_API_KEY`). Затем:

```powershell
.venv\Scripts\python.exe -m src.agents
```

Примеры запросов: `Analyze the dataset`, `Train all required models`, `Which model performed best?`, `Predict week 1, hour 1, node N0000`. `main.py --agent` сначала выполняет ML цикл, затем открывает этот же чат. При отсутствии ключа `main.py` всё равно выполняет основной assignment.
