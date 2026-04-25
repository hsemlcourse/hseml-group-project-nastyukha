[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)
# ML Project — [Название проекта]

**Студент:** Гудкова Ксения Дмитриевна

**Группа:** БИВ238


## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Запуски](#быстрый-старт)
4. [Данные](#данные)
5. [Результаты](#результаты)
7. [Отчёт](#отчёт)


## Описание задачи

<!-- Кратко опишите задачу: что предсказываем, какой датасет, метрика качества -->

**Задача:** Регрессия

**Датасет:** Использовался датасет с данными о количестве взятых в прокат велосипедов с кагла (в качестве архива данных, так как иначе понадобится год чтобы данные собрать):https://www.kaggle.com/datasets/ranjanrakesh51/divvy-bike-sharing-data-jan-24-to-mar-25?resource=download&select=202503-divvy-tripdata.csv. Также использовалась API для получения данных о погодных условиях за тот период. Далее все соединялось в один датасет.

**Целевая метрика:** Испаользовались MAE и R2


## Структура репозитория
Опишите структуру проекта, сохранив при этом верхнеуровневые папки. Можно добавить новые при необходимости.
```
.
├── data
│   ├── processed               # Очищенные и обработанные данные
│   └── raw                     # Исходные файлы
├── models                      # Сохранённые модели 
├── notebooks
│   ├── 01_eda.ipynb            # EDA
│   ├── 02_baseline.ipynb       # Baseline-модель
│   └── 03_experiments.ipynb    # Эксперименты и ablation study
├── presentation                # Презентация для защиты
├── report
│   ├── images                  # Изображения для отчёта
│   └── report.md               # Финальный отчёт
├── src
│   ├── preprocessing.py        # Предобработка данных
│   └── modeling.py             # Обучение и оценка моделей
├── tests
│   └── test.py                 # Тесты пайплайна
├── requirements.txt
└── README.md
```

## Запуск

Этот блок замените способом запуска вашего сервиса.
```bash
# 1. Клонировать репозиторий
git clone <url>
cd <repo-name>

# 2. Создать виртуальное окружение
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# 3. Установить зависимости
pip install -r requirements.txt

# 4. Запуск программы
python src/modeling.py
```

## Данные
- `data/raw/` — исходные файлы
- `data/processed/` — предобработанные данные


## Результаты
Здесь коротко выпишите результаты.
| Модель | [Метрика 1] | [Метрика 2] | Примечание |
|--------|-------------|-------------|------------|
| Baseline | — | — | |
| Лучшая модель | — | — | |

<img width="363" height="114" alt="image" src="https://github.com/user-attachments/assets/52b4f991-7b82-449c-b2f9-ebedb25176b7" />



## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md)
