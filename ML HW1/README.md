## HW Regression with inference
### Описание проделанных действий
| **Этап**             | **Описание**                                                                                     |
|-----------------------|-------------------------------------------------------------------------------------------------|
| **Анализ данных**    | - Провели исследование данных, включая визуализацию ключевых признаков и корреляций.            |
|                       | - Выявили и исправили проблемы с пропущенными значениями и выбросами.                           |
|                       | - Преобразовали числовые столбцы (например, `mileage`, `engine`) из текстовых значений.         |
| **Модели**           | - Обучили классическую линейную регрессию для предсказания целевой переменной.                  |
|                       | - Настроили гиперпараметры для `Ridge` и `Lasso` регрессий, улучшив метрики качества.  |
|                       | - Попробовали улучшить качество с помощью других моделей  |
|                       | - Провели инференс на тестовых данных.                                                         |
|                       | - Оценили качество моделей с использованием $R^2$ и $MSE$.                                     |
| **Создание пайплайна**| - Разработали пайплайн для автоматизации предобработки данных и обучения модели.                |
|                       | - Включили обработку категориальных признаков и нормализацию числовых данных.                   |
|                       | - Реализовали сохранение и загрузку модели с использованием `joblib`.                          |
| **FastAPI**          | - Создали серверное приложение для взаимодействия с моделью через API.                         |
|                       | - Реализовали эндпоинты для обработки запросов и выдачи предсказаний.                          |
|                       | - Настроили валидацию входных данных с использованием `Pydantic`.                               |
### Результаты
- Провели обработку данных, обнаружили большое кол-во дубликатов, которые удалили, преобразовали данные,
привели их к нужному виду для модели, удалили ненужные столбцы.
- Попробовали применить разные модели, в результате лучше всего себя показала Ridge model, увеличив показатель метрики R2 с базовой
0.59 до 0.65 <br>
Скрин с работой сервиса 
![image](https://github.com/user-attachments/assets/5c3437fc-d886-466d-8c8f-6b0fb5706fa8)
