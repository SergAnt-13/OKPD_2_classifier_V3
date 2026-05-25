# calibrate_weights.py
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from tqdm import tqdm
from config.settings import TRAINING_DATA_DIR, RAW_DATA_DIR, REFERENCE_DIR, FAISS_DIR
from backend.preprocessing.cleaner import TextCleaner
from backend.models.retriever import Retriever

cleaner = TextCleaner(abbreviations_path=REFERENCE_DIR / "сокращения.xlsx")
retriever = Retriever(
    model_name="artifacts/models/bge-m3-frozen-stratified-epoch2",
    index_path=FAISS_DIR / "okpd_index.faiss",
    id_map_path=FAISS_DIR / "id_map.csv"
)

# === ШАГ 1: калибровка на золоте ===
gold = pd.read_excel(TRAINING_DATA_DIR / "train.xlsx", dtype=str)
gold = gold[["Номенклатура", "Код ОКПД2"]].dropna()
gold.columns = ["text", "true_code"]

correct_scores = []
incorrect_scores = []

for _, row in tqdm(gold.iterrows(), total=len(gold), desc="Калибровка на золоте"):
    q = cleaner.clean(row["text"], use_stemmer=True)
    result = retriever.search(q, top_k=1)
    if not result["candidates"]:
        continue
    top1 = result["candidates"][0]
    score = top1.get("rerank_score") or top1.get("score")
    if top1["code"] == row["true_code"].strip():
        correct_scores.append(score)
    else:
        incorrect_scores.append(score)

correct_scores = np.array(correct_scores)
incorrect_scores = np.array(incorrect_scores)

print(f"\n=== КАЛИБРОВКА НА ЗОЛОТЕ ===")
print(f"Правильных предсказаний: {len(correct_scores)}")
print(f"Неправильных предсказаний: {len(incorrect_scores)}")
print(f"\nScore когда ПРАВИЛЬНО: mean={correct_scores.mean():.3f}, "
      f"p25={np.percentile(correct_scores,25):.3f}, "
      f"p50={np.percentile(correct_scores,50):.3f}")
print(f"Score когда НЕПРАВИЛЬНО: mean={incorrect_scores.mean():.3f}, "
      f"p25={np.percentile(incorrect_scores,25):.3f}, "
      f"p50={np.percentile(incorrect_scores,50):.3f}")

# Находим порог: минимальный score где >50% предсказаний правильные
thresholds = np.arange(0.3, 1.0, 0.05)
print(f"\n{'Порог':>8} {'P(правильно|score>T)':>22} {'Покрытие':>10}")
for t in thresholds:
    n_correct = (correct_scores >= t).sum()
    n_incorrect = (incorrect_scores >= t).sum()
    total_above = n_correct + n_incorrect
    if total_above == 0:
        continue
    precision = n_correct / total_above
    coverage = total_above / (len(correct_scores) + len(incorrect_scores))
    print(f"{t:>8.2f} {precision:>22.3f} {coverage:>10.3f}")

# === ШАГ 2: взвешивание промки ===
print("\n=== ПРИМЕНЯЕМ К ПРОМКЕ ===")
prom = pd.read_csv(RAW_DATA_DIR / "prom_with_weights.csv", dtype=str)
food = prom[prom['true_code'].str.startswith(('01','02','03','10'))].copy()
print(f"Пищевых записей для взвешивания: {len(food)}")

new_weights = []
scores_out = []

for _, row in tqdm(food.iterrows(), total=len(food), desc="Взвешивание промки"):
    q = cleaner.clean(row["text"], use_stemmer=True)
    result = retriever.search(q, top_k=1)
    if not result["candidates"]:
        new_weights.append(0.0)
        scores_out.append(0.0)
        continue
    top1 = result["candidates"][0]
    s = top1.get("rerank_score") or top1.get("score")
    scores_out.append(s)
    if s >= 0.75:
        new_weights.append(round(s ** 2, 3))
    else:
        new_weights.append(0.0)

food["top1_score"] = scores_out
food["calibrated_weight"] = new_weights

usable = food[food["calibrated_weight"] > 0]
print(f"\nЗаписей с весом > 0: {len(usable)} ({len(usable)/len(food):.1%})")
print(f"Средний вес: {usable['calibrated_weight'].mean():.3f}")
print(f"\nРаспределение весов:")
bins = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for i in range(len(bins)-1):
    n = ((usable['calibrated_weight'] >= bins[i]) & 
         (usable['calibrated_weight'] < bins[i+1])).sum()
    print(f"  {bins[i]:.1f}-{bins[i+1]:.1f}: {n} записей")

food.to_csv(RAW_DATA_DIR / "prom_calibrated.csv", index=False)
print(f"\nСохранено в prom_calibrated.csv")