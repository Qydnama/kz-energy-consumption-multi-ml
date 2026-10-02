# Тестовые сценарии

Команда для автоматических проверок без API-ключа: `python -m unittest discover -s tests -v`. Полный сценарий с настоящим LLM: `python -m src.agents --demo`. Он запускает агентов по порядку и создаёт журнал с распределением вызовов. Отдельные запросы можно вводить через `python -m src.agents`.

| № | Запрос или условие | Ожидаемый результат |
|---:|---|---|
| 1 | `Analyze the dataset` | `data_agent` сообщает 11 088 строк, 5 признаков, target `demand`. |
| 2 | `Rebuild the dataset from the workbook` | `data_agent` пересоздаёт `data/energy.csv`; схема и количество строк сохранены. |
| 3 | `Train all required models` | `training_agent` запускает 11 алгоритмов и 10-fold CV; создаётся `results/model_results.csv`. |
| 4 | `Which model is best?` | `evaluation_agent` читает таблицу, выбирает минимальный RMSE и сохраняет Pipeline. |
| 5 | `Predict week 1, hour 1, node N0000` | `prediction_agent` вызывает сохранённую модель и возвращает конечное число. |
| 6 | `Predict week 3, hour 1, node N0000` | Неверная неделя отклоняется; число не придумывается. |
| 7 | `Predict week 1, hour 169, node N0000` | Час вне 1–168 отклоняется. |
| 8 | `Predict week 1, hour 1, node UNKNOWN` | Неизвестный узел отклоняется. |
| 9 | `Verify the dataset and model artifacts` | `verification_agent` вызывает обе проверки и возвращает два `passed: true` для корректного проекта. |
| 10 | `Run the complete Kazakhstan electricity demand workflow` | Оркестратор последовательно вызывает пять специалистов; журнал содержит их вызовы и доли нагрузки. |

Сценарии 1–5, 9–10 требуют LLM для проверки маршрутизации. Их предметные Python-функции и ошибки входа проверяются автоматическими тестами; финальный ответ LLM дополнительно сверяется с CSV и логом. Сценарии 6–8 могут быть проверены напрямую через `predict_energy_consumption()` без расходов на API.
