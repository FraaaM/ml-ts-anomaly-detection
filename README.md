# ml-ts-anomaly-detection

# Отчёт о проделанной работе: Pipeline детекции аномалий во временных рядах
**Проект:** Classical ML Time Series Anomaly Detection  

---

## 1. Обзор проекта

Цель проекта — разработка end-to-end пайплайна для детекции аномалий во временных рядах с использованием классических методов машинного обучения (scikit-learn, pandas, numpy). Пайплайн поддерживает два режима работы:
- **Point-wise detection**: предсказание аномальности отдельной точки во времени
- **Series-level detection**: классификация целого ряда как аномального/нормального

**Ключевые требования:**
- Работа с большими объёмами данных (до 12+ млн точек) на ограниченном железе
- Исключение data leakage при сплите данных
- Гибкость в выборе модели и признаков
- Production-ready CLI интерфейс

---

## 2. Сбор и подготовка данных

### 2.1 Источники данных

Данные получены из публичного бенчмарка [TSB-UAD-Public-v2](https://www.thedatum.org/datasets/TSB-UAD-Public-v2.zip). Проведена ручная валидация:
- ✅ Отбор валидных датасетов с корректной разметкой
- ✅ Удаление дубликатов и битых файлов
- ✅ Фильтрация по минимальной длине серии

### 2.2 Итоговые датасеты

| ID | Название | Источник | Единица анализа | Длина серии | Кол-во серий | Примечание |
|----|----------|----------|----------------|-------------|--------------|------------|
| **R1** | Combined IT-Ops | YAHOO, SMD, IOPS, Exathlon, WSD, NEK | univariate series | 1,500 – 30,000 | ~5000 | Веб-трафик, метрики серверов, логи Spark, сетевые потоки |

### 2.3 Формат хранения

**Основной файл** (`R1.parquet`):
```
Columns:
- series_id: str  # уникальный ID в формате {group}__{dataset}__{orig_id}_{sample_id}
- time_index: int64  # монотонный индекс, начинается с 0 для каждой серии
- value: float64  # сырое значение из источника
- label: int8  # point-wise метка аномалии (0/1)
```

**Файл метаданных** (`R1_metadata.parquet`):
```
Columns:
- series_id: str
- length: int  # длина серии
- num_point_anomalies: int  # количество аномальных точек
- y_i: int8  # series-level label (1 если есть хотя бы одна аномалия)
- is_split: bool  # была ли серия разрезана
- original_length: int  # длина исходной серии до сплита
- source_notes: str  # дополнительная информация (периодичность и т.д.)
- period_detected: Optional[int]  # обнаруженный период
- is_representative: bool  # репрезентативность статистик
- anomaly_ratio: float  # доля аномальных точек
```

**Правила формирования `series_id`:**
```
{group}__{original_dataset}__{original_id}_{sample_id}

Примеры:
- R1__YAHOO__real_42_full        # серия не разрезалась
- R1__Exathlon__chunk0_test_001  # первый чанк разрезанной серии
```

**Правила сплита (применяются один раз при создании пула):**
- Если `L ≤ 15,000` → `sample_id = "full"`
- Если `L > 15,000` → создаются непересекающиеся чанки от 1000 до 32000 точек
- Каждая точка принадлежит ровно одному чанку

---

## 3. Программа семплирования (`sampler-r1-r2.py`)

### 3.1 Архитектура

```
TimeSeriesSampler
├── detect_period()           # FFT-based детекция периодичности
├── compute_optimal_chunk_size() # адаптивный расчёт размера чанка
├── expand_anomaly_cluster()  # расширение чанка для захвата кластеров аномалий
├── check_representativeness() # проверка статистической репрезентативности
├── align_boundary()          # выравнивание границ по периоду
├── _optimize_expansion()     # O(log N) поиск минимального расширения
├── _create_valid_sample()    # 4-tier логика контроля аномалий
├── process_series()          # обработка одной серии
└── process_group()           # обработка всей группы датасетов
```

### 3.2 4-уровневая система контроля аномалий во временном ряде
| Уровень | Диапазон anomaly_ratio | Действие |
|---------|----------------------|----------|
| ✅ Target | ≤ 0.06 | Принять как есть |
| ⚠️ Acceptable | 0.06 – 0.15 | Расширить чанк для "разбавления" аномалий |
| 🔴 Max | 0.15 – 0.27 | Объединить в полную серию или извлечь чистые сегменты |
| ❌ Reject | > 0.27 | Отклонить (если нельзя исправить) |

- anomaly_ratio - доля аномалий в данных (0.06 = 6%)

### 3.3 Ключевые оптимизации

- **Период-аверность**: чанки выравниваются по границам обнаруженного периода
- **Адаптивный размер**: старт от `target_chunk_min`, округление до кратного периоду
- **Экспансия кластеров**: захват связанных аномалий в пределах `anomaly_lookahead`
- **Быстрый поиск расширения**: бинарный поиск + локальная доводка для снижения % аномалий
- **Обработка остатков**: присоединение к последнему чанку с пересчётом меток

### 3.4 Запуск

```bash
python sampler-r1-r2.py \
  --r1-raw ./raw_data/R1 \
  --r2-raw ./raw_data/R2 \
  --output ./data \
  --chunk-min 1500 \
  --chunk-max 35000 \
  --target-ratio 0.06
```

---

## 4. Модели предсказания

### 4.1 Point-wise детекция (`model.py` — режим по умолчанию)

**Задача**: для каждой точки времени предсказать `label ∈ {0, 1}`

**Признаки (9 наиболее информативных):**
```python
# Мгновенные изменения
diff_1, abs_diff, pct_change

# Локальная волатильность (окно 10)
roll_mean_10, dev_from_mean_10, roll_std_10

# Глобальный размах (окно 20)
roll_range_20

# Авторегрессия
lag_1, lag_5
```

**Модель**: `HistGradientBoostingClassifier`

**Гиперпараметры:**
```python
max_depth=6, learning_rate=0.05, max_iter=500,
class_weight='balanced', max_bins=64, min_samples_leaf=50,
l2_regularization=0.1, early_stopping=True
```

**Сплит**: `GroupShuffleSplit` по `series_id` для исключения data leakage

**Оптимизация порога**: автоматический подбор по максимизации F1-score на валидации

### 4.2 Series-level детекция (`model.py` — альтернативный режим)

**Задача**: для каждой серии предсказать `y_i ∈ {0, 1}` (есть ли хотя бы одна аномалия)

**Агрегированные признаки:**
```python
# Статистики значений
value_mean, value_std, value_min, value_max, value_median, value_skew, value_kurt

# Статистики изменений
diff_mean, diff_max, diff_std

# Производные признаки
range = max - min
cv = std / |mean|          # коэффициент вариации
diff_ratio = diff_mean / |mean|
```

**Преимущества series-level подхода:**
- Меньший дисбаланс классов (~41% аномальных серий против ~5% аномальных точек)
- Устойчивость к шуму за счёт агрегации
- Проще интерпретировать результаты для бизнес-решений

**Недостатки:**
- Потеря временной точности (неизвестно, *когда* произошла аномалия)
- Требует отдельной постобработки для локализации

### 4.3 Сравнение режимов

| Критерий | Point-wise | Series-level |
|----------|-----------|--------------|
| Единица предсказания | Точка времени | Целая серия |
| Дисбаланс классов | ~5% позитивов | ~41% позитивов |
| Точность локализации | Высокая | Низкая |
| Устойчивость к шуму | Низкая | Высокая |
| Потребление памяти | Высокое (12.9 млн × 24 признака) | Низкое (~2000 серий × 12 признаков) |
| Время обучения | 6-10 минут | 10 - 20 секунд |
| ROC-AUC (тест) | 0.89 | *ожидаемо >0.90* |
| PR-AUC (тест) | 0.45 | *ожидаемо >0.60* |

---

## 5. Технические решения и оптимизации

### 5.1 Работа с памятью

| Проблема | Решение | Эффект |
|----------|---------|--------|
| `ArrowMemoryError` при `groupby` | Конвертация `PyArrow` типов в `numpy` сразу после чтения | Устранение крашей |
| Фрагментация памяти при `pd.concat` | Обработка батчами + `gc.collect()` каждые 10 итераций | Стабильное потребление ~600 МБ |
| `inf`/`NaN` в признаках | Замена `inf → NaN` + `SimpleImputer(strategy='median')` | Корректная работа пайплайна |
| Переполнение `float32` | Обрезка выбросов по MAD-порогу | Защита от `ValueError` |

### 5.2 Оптимизация признаков

- Сокращение с 34 до 9 признаков: удаление избыточных лагов и окон
- Добавление `z_score`, `ewm_*`, `roll_skew` для улучшения детекции аномалий
- Жёсткое приведение типов: `float32`, `int8`, `category` для экономии памяти

### 5.3 Гибкость конфигурации

Все параметры вынесены в `@dataclass Config`:
```python
@dataclass
class Config:
    data_path: Path
    model_dir: Path
    windows: Tuple[int, ...] = (10, 20)
    lags: Tuple[int, ...] = (1, 5)
    train_sample_frac: float = 1.0  # доля используемых данных в датасете (100%)
    # ... гиперпараметры модели
```

---

## 6. Результаты

### 6.1 (Point-wise, R1)
```
2026-05-23 15:44:47,350 | INFO     | === TRAINING ===
2026-05-23 15:44:48,722 | INFO     | Extracting features...
2026-05-23 15:46:32,726 | WARNING  | Found 507684 NaN values in features. Will be imputed with median.
2026-05-23 15:46:32,746 | INFO     | Features extracted: (12950281, 23), Memory: 1178.79 MB
2026-05-23 15:46:32,749 | INFO     | Splitting data...
2026-05-23 15:46:37,179 | INFO     | Train: 8970142, Val: 1973111, Test: 2007028
2026-05-23 15:46:37,187 | INFO     | Train anomaly ratio: 0.0516
2026-05-23 15:54:47,440 | INFO     | Optimal threshold (F1): 0.8635

2026-05-23 15:54:48,005 | INFO     | Validation metrics:
              precision    recall  f1-score   support

           0       0.98      0.98      0.98   1873536
           1       0.63      0.58      0.60     99575

    accuracy                           0.96   1973111
   macro avg       0.80      0.78      0.79   1973111
weighted avg       0.96      0.96      0.96   1973111

2026-05-23 15:55:21,325 | INFO     | Test metrics:
              precision    recall  f1-score   support

           0       0.98      0.98      0.98   1907675
           1       0.60      0.52      0.56     99353

    accuracy                           0.96   2007028
   macro avg       0.79      0.75      0.77   2007028
weighted avg       0.96      0.96      0.96   2007028

2026-05-23 15:55:21,328 | INFO     | Test ROC-AUC: 0.9249, PR-AUC: 0.5998```
```

### 6.2 (Series-level_model, R1)

```
2026-05-23 15:43:01,255 | INFO     | === TRAINING ===
2026-05-23 15:43:02,431 | INFO     | Aggregating point-level data to series-level features...
2026-05-23 15:43:10,275 | INFO     | Aggregated to 4768 series with 14 features
2026-05-23 15:43:10,383 | INFO     | Splitting series data...
2026-05-23 15:43:10,407 | INFO     | Train: 3336, Val: 716, Test: 716
2026-05-23 15:43:10,410 | INFO     | Train anomaly ratio: 0.3891
2026-05-23 15:43:10,895 | INFO     | Optimal threshold (F1): 0.5057

2026-05-23 15:43:10,905 | INFO     | Validation metrics:
              precision    recall  f1-score   support

           0       0.95      0.93      0.94       438
           1       0.89      0.92      0.91       278

    accuracy                           0.93       716
   macro avg       0.92      0.93      0.92       716
weighted avg       0.93      0.93      0.93       716

2026-05-23 15:43:10,927 | INFO     | Test metrics:
              precision    recall  f1-score   support

           0       0.96      0.92      0.94       437
           1       0.89      0.94      0.91       279

    accuracy                           0.93       716
   macro avg       0.92      0.93      0.93       716
weighted avg       0.93      0.93      0.93       716

2026-05-23 15:43:10,927 | INFO     | Test ROC-AUC: 0.9774, PR-AUC: 0.9665
```

## 7. Запуск пайплайна

### 7.1 Обучение

```bash
# Point-wise режим (по умолчанию)
python model.py train --data ./data/R1.parquet --model-dir ./models

# Series-level режим (раскомментировать в _build_pipeline)
# + использовать агрегированные признаки
```

### 7.2 Предсказание

```bash
# На новых данных
python model.py predict --data ./data/R2.parquet --model-dir ./models \
  --output ./predictions/R2_preds.parquet

# Получить series-level флаги из point-wise предсказаний
# (без переобучения)
preds = pd.read_parquet("predictions.parquet")
series_flags = preds.groupby('series_id')['pred_label'].max().reset_index()
```

## 9. Структура проекта

```
ml-ts-anomaly-detection/
|── raw_data/...
│   
├── data / #(появится после запуска sampler-r1-r2.py)
│   ├── R1_metadata.parquet
│   ├── R1.parquet
│   ├── R2_metadata.parquet
│   └── R2.parquet
├── src/
│   ├── models_point-wise/
│   │   └── anomaly_model.joblib
│   └── models_series-level/
│       └── anomaly_model.joblib
├── point-wise_model.ipynb
├── real_datasets_analysis.ipynb
├── sampler-r1-r2.py
├── series-level_model.ipynb
└── README.md # ОТЧЕТ
```

---

## 10. Выводы

1. ✅ Построен рабочий end-to-end пайплайн для детекции аномалий во временных рядах
2. ✅ Реализованы два режима: point-wise (точная локализация) и series-level (устойчивая классификация)
3. ✅ Достигнута стабильная работа на ограниченном железе (<2 ГБ RAM) за счёт оптимизаций памяти и признаков
4. ✅ Получены метрики: ROC-AUC = 0.89, Recall = 53% при дисбалансе 5% 