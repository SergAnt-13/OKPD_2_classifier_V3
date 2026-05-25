# backend/tests/evaluate_prom_quality.py
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from tqdm import tqdm
from config.settings import RAW_DATA_DIR, REFERENCE_DIR, FAISS_DIR
from backend.preprocessing.cleaner import TextCleaner
from backend.models.retriever import Retriever

# Загружаем промышленную разметку
prom = pd.read_excel(RAW_DATA_DIR / "all_nomenclature.xlsx", dtype=str)
prom = prom[['nomenclature', 'okpd2_code']].dropna()
prom.columns = ["text", "true_code"]

cleaner = TextCleaner(abbreviations_path=REFERENCE_DIR / "сокращения.xlsx")
retriever = Retriever(model_name="artifacts/models/bge-m3-frozen-stratified-epoch2",
                      index_path=FAISS_DIR / "okpd_index.faiss",
                      id_map_path=FAISS_DIR / "id_map.csv")

weights = []
top1_count = 0
top5_count = 0
top10_count = 0
total = 0

for _, row in tqdm(prom.iterrows(), total=len(prom), desc="Анализ промки"):
    text = row["text"]
    true_code = row["true_code"].strip()
    if not text or not true_code:
        continue
    q = cleaner.clean(text, use_stemmer=True)
    result = retriever.search(q, top_k=10)
    candidates = result["candidates"]
    if not candidates:
        weights.append(0.1)  # совсем плохой пример
        continue
    total += 1
    found_at = None
    for i, c in enumerate(candidates, 1):
        if c["code"] == true_code:
            found_at = i
            break
    if found_at == 1:
        top1_count += 1
        weights.append(1.0)
    elif found_at is not None and found_at <= 5:
        top5_count += 1
        weights.append(0.5)
    elif found_at is not None and found_at <= 10:
        top10_count += 1
        weights.append(0.3)
    else:
        weights.append(0.1)  # не найден в топ-10 — вероятно, ошибка

print(f"\nВсего записей: {total}")
print(f"Top-1 совпадений: {top1_count} ({top1_count/total:.2%})")
print(f"Top-5 совпадений: {top5_count} ({top5_count/total:.2%})")
print(f"Top-10 совпадений: {top10_count} ({top10_count/total:.2%})")

# Сохраняем веса
prom["weight"] = weights[:len(prom)]
prom.to_csv(RAW_DATA_DIR / "prom_with_weights.csv", index=False)
print(f"Веса сохранены в {RAW_DATA_DIR / 'prom_with_weights.csv'}")