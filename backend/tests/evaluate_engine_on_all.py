# backend/tests/evaluate_engine_on_all.py
import sys, os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from tqdm import tqdm
from config.settings import TRAINING_DATA_DIR, RAW_DATA_DIR, REFERENCE_DIR, FAISS_DIR
from backend.models.retriever import Retriever
from backend.models.engine import DecisionEngine

# ---------- Настройки ----------
MODEL = "artifacts/models/bge-m3-frozen-3epoch"
CLASSIFIER_PATH = Path("artifacts/models/berta_classifier_improved")
INPUT_PATH = RAW_DATA_DIR / "all_nomenclature.xlsx"
OUTPUT_PATH = Path("artifacts/full_nomenclature_report.xlsx")
BATCH_SIZE = 1000       # пауза каждые N записей, чтобы видеть прогресс
# --------------------------------

# Загружаем все данные
all_data = pd.read_excel(INPUT_PATH, dtype=str)
# Ожидаем колонки 'nomenclature' и, возможно, 'okpd2_code'
if 'nomenclature' not in all_data.columns:
    raise KeyError("Ожидается столбец 'nomenclature'")
print(f"Всего записей: {len(all_data)}")

# Инициализируем Retriever и Engine (модели загрузятся один раз)
retriever = Retriever(
    model_name=MODEL,
    index_path=FAISS_DIR / "okpd_index.faiss",
    id_map_path=FAISS_DIR / "id_map.csv",
)
engine = DecisionEngine(retriever, classifier_path=CLASSIFIER_PATH)

# Прогоняем предсказания
predicted_codes = []
modes = []
confidences = []
classifier_codes = []
classifier_probs = []

for i, row in tqdm(all_data.iterrows(), total=len(all_data), desc="Предсказание"):
    query = str(row['nomenclature']).strip()
    res = engine.predict(query)
    predicted_codes.append(res['predicted_code'])
    modes.append(res['mode'])
    confidences.append(res['confidence'])
    classifier_codes.append(res.get('classifier_code'))
    classifier_probs.append(res.get('classifier_prob'))

    # Пауза для отслеживания памяти (необязательно)
    if (i+1) % BATCH_SIZE == 0:
        pass  # можно напечатать промежуточный статус

# Добавляем результаты в DataFrame
all_data['predicted_code'] = predicted_codes
all_data['mode'] = modes
all_data['confidence'] = confidences
all_data['classifier_code'] = classifier_codes
all_data['classifier_prob'] = classifier_probs

# Если есть текущий код, считаем расхождения
if 'okpd2_code' in all_data.columns:
    all_data['current_code'] = all_data['okpd2_code']
    all_data['match'] = all_data['predicted_code'] == all_data['current_code']
    # Здесь можно добавить анализ ставок НДС, если загрузить vat_exempt_codes
    print("\nРасхождения с текущим кодом:")
    mismatch = all_data[~all_data['match']]
    print(f"  Всего расхождений: {len(mismatch)} ({len(mismatch)/len(all_data)*100:.1f}%)")
    print(f"  Из них AUTO: {(mismatch['mode']=='AUTO').sum()}")

# Общая статистика
print("\nРаспределение режимов:")
for mode in ['AUTO', 'REVIEW', 'MANUAL']:
    count = (all_data['mode'] == mode).sum()
    pct = count / len(all_data) * 100
    print(f"  {mode}: {count} ({pct:.1f}%)")

print(f"\nСредняя уверенность: {all_data['confidence'].mean():.3f}")
print(f"Доля ненулевых classifier_code: {all_data['classifier_code'].notna().mean()*100:.1f}%")

# Сохраняем отчёт
all_data.to_excel(OUTPUT_PATH, index=False)
print(f"\nПолный отчёт сохранён в {OUTPUT_PATH}")