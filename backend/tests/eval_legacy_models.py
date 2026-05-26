# backend/tests/eval_legacy_models.py
import sys, os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import torch
import faiss
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from config.settings import TRAINING_DATA_DIR, REFERENCE_DIR, FAISS_DIR
from backend.preprocessing.cleaner import TextCleaner

# ---------- Настройки ----------
MODELS = [
    "artifacts/models/bge-m3-frozen-stratified-epoch2",
    "artifacts/models/bge-m3-frozen-3epoch",
]
GOLD_PATH = TRAINING_DATA_DIR / "train.xlsx"
OKPD_PATH = REFERENCE_DIR / "okpd_2.xlsx"
ABBR_PATH = REFERENCE_DIR / "сокращения.xlsx"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
# --------------------------------

cleaner = TextCleaner(abbreviations_path=ABBR_PATH)

# Загружаем золотую выборку
gold = pd.read_excel(GOLD_PATH, dtype=str)[["Номенклатура", "Код ОКПД2"]].dropna()
gold.columns = ["text", "code"]

# Справочник ОКПД-2 (стеммированные названия)
okpd = pd.read_excel(OKPD_PATH, dtype=str)
okpd = okpd.dropna(subset=["name"])
okpd["stemmed"] = okpd["name"].apply(lambda x: cleaner.clean(x, use_stemmer=True))
okpd_codes = okpd["code"].values

results = []

for model_name in MODELS:
    print(f"\n===== Оценка модели: {model_name} =====")
    # Загружаем модель
    model = SentenceTransformer(model_name, device=DEVICE)
    # Строим индекс
    print("Строим FAISS-индекс...")
    emb = model.encode(okpd["stemmed"].tolist(), batch_size=BATCH_SIZE, convert_to_numpy=True, show_progress_bar=True)
    faiss.normalize_L2(emb)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    # Оценка
    hits = {1:0, 5:0, 10:0}
    ndcg_sum = 0.0
    for _, row in tqdm(gold.iterrows(), total=len(gold), desc="Оценка"):
        q = cleaner.clean(row["text"], use_stemmer=True)
        q_emb = model.encode([q], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        scores, indices = index.search(q_emb, 10)
        for i, pos in enumerate(indices[0], 1):
            if okpd_codes[pos] == row["code"]:
                for k in hits:
                    if i <= k:
                        hits[k] += 1
                ndcg_sum += 1.0 / np.log2(i+1)
                break

    total = len(gold)
    r1 = hits[1] / total
    r5 = hits[5] / total
    r10 = hits[10] / total
    ndcg = ndcg_sum / total
    results.append((model_name, r1, r5, r10, ndcg))
    print(f"R@1={r1:.4f}, R@5={r5:.4f}, R@10={r10:.4f}, NDCG={ndcg:.4f}")

# Итоговая таблица
print("\n" + "="*70)
print("Сравнение старых моделей (честные метрики)")
print("="*70)
print(f"{'Модель':<45} {'R@1':<8} {'R@5':<8} {'R@10':<8} {'NDCG':<8}")
print("-"*70)
for name, r1, r5, r10, ndcg in results:
    print(f"{name:<45} {r1:<8.4f} {r5:<8.4f} {r10:<8.4f} {ndcg:<8.4f}")

best = max(results, key=lambda x: x[3])  # по R@10
print(f"\nЛучшая модель по R@10: {best[0]} (R@10={best[3]:.4f})")