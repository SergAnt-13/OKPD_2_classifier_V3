# backend/tests/test_final_ab.py
# Финальный A/B-тест: Dense only vs Dense + NEW Reranker
# Стемминг для Dense, оригинальные названия для реранкера
import sys, os, pickle
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import CrossEncoder

from config.settings import TRAINING_DATA_DIR, REFERENCE_DIR, FAISS_DIR
from backend.models.retriever import Retriever
from backend.preprocessing.cleaner import TextCleaner

# ---------- Настройки ----------
MODEL_NAME = "artifacts/models/bge-m3-frozen-3epoch"
RERANKER_PATH = "artifacts/models/reranker-final"
CACHE = Path("artifacts/dense_cache_stemmed_v2.pkl")
TOP_K_DENSE = 10       # для оценки Dense – стандартный top_k
TOP_K_CACHE = 15       # сохраняем 15 кандидатов для реранкера
RERANK_BATCH = 16
MAX_LENGTH = 256

# ---------- Данные ----------
gold = pd.read_excel(TRAINING_DATA_DIR / "train.xlsx", dtype=str)
gold = gold[["Номенклатура", "Код ОКПД2"]].dropna()
gold.columns = ["text", "true_code"]
print(f"Золотая выборка: {len(gold)} примеров")

cleaner = TextCleaner(abbreviations_path=REFERENCE_DIR / "сокращения.xlsx")

# ---------- Dense retrieval ----------
retriever = Retriever(
    model_name=MODEL_NAME,
    index_path=FAISS_DIR / "okpd_index.faiss",
    id_map_path=FAISS_DIR / "id_map.csv",
)

dense_hits = {1:0, 3:0, 5:0, 10:0}
dense_ndcg = 0.0
dense_cache = []

print("\n=== Dense only (стемминг) ===")
for _, row in tqdm(gold.iterrows(), total=len(gold), desc="Dense only"):
    q_stem = cleaner.clean(row["text"], use_stemmer=True)          # для Dense
    q_orig = cleaner.clean(row["text"], use_stemmer=False)         # для реранкера
    true_code = row["true_code"]

    # Поиск с возвратом 40 кандидатов, мы возьмём первые TOP_K_DENSE для метрик
    raw = retriever.search(q_stem, top_k=TOP_K_DENSE, use_reranker=False)
    all_cands = raw["candidates"]           # до 40 штук
    # метрики Dense only по первым 10
    for i, c in enumerate(all_cands[:TOP_K_DENSE], 1):
        if c["code"] == true_code:
            for k in dense_hits:
                if i <= k:
                    dense_hits[k] += 1
            dense_ndcg += 1.0 / np.log2(i + 1)
            break

    # Сохраняем топ-TOP_K_CACHE для реранкера с оригинальными названиями
    top_for_rerank = all_cands[:TOP_K_CACHE]
    dense_cache.append({
        "query_orig": q_orig,                # чистый запрос без стемминга
        "true_code": true_code,
        "candidates": [{"code": c["code"],
                        "original_name": c["name"],   # оригинальное название из справочника
                        "score": c["score"]}
                       for c in top_for_rerank]
    })

total = len(gold)
print("Dense only (стемминг):")
for k in [1, 3, 5, 10]:
    print(f"  Recall@{k}: {dense_hits[k]/total:.4f}")
print(f"  NDCG@10: {dense_ndcg/total:.4f}")

with open(CACHE, "wb") as f:
    pickle.dump(dense_cache, f)
print(f"Кандидаты сохранены в {CACHE}")

# Освобождаем память
del retriever
torch.cuda.empty_cache()

# ---------- Ререранкер (с оригинальными названиями) ----------
print("\n=== Dense + NEW Reranker (оригинальные названия) ===")

reranker = CrossEncoder(RERANKER_PATH, max_length=MAX_LENGTH, device="cuda")
reranker.model.to("cuda")
print("Ререранкер на:", next(reranker.model.parameters()).device)

with open(CACHE, "rb") as f:
    cache_data = pickle.load(f)

all_pairs = []
for item in cache_data:
    q_orig = item["query_orig"]
    for c in item["candidates"]:
        all_pairs.append((c["original_name"], q_orig))   # пара (документ, запрос) без стеммов

print(f"Переранжирование {len(all_pairs)} пар...")
scores = reranker.predict(all_pairs, batch_size=RERANK_BATCH, show_progress_bar=True)

idx = 0
for item in cache_data:
    for c in item["candidates"]:
        c["rerank_score"] = float(scores[idx])
        idx += 1
    item["candidates"].sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

rerank_hits = {1:0, 3:0, 5:0, 10:0}
rerank_ndcg = 0.0
for item in cache_data:
    true_code = item["true_code"]
    candidates = item["candidates"]
    for i, c in enumerate(candidates[:10], 1):
        if c["code"] == true_code:
            for k in rerank_hits:
                if i <= k:
                    rerank_hits[k] += 1
            rerank_ndcg += 1.0 / np.log2(i + 1)
            break

print("Dense + NEW Reranker (оригинальные названия):")
for k in [1, 3, 5, 10]:
    print(f"  Recall@{k}: {rerank_hits[k]/total:.4f}")
print(f"  NDCG@10: {rerank_ndcg/total:.4f}")

# ---------- Итог ----------
print("\n" + "="*60)
print("Сравнение (стемминг для Dense, оригинал для реранкера)")
print("="*60)
print(f"{'Метод':<25} {'R@1':<8} {'R@3':<8} {'R@5':<8} {'R@10':<8} {'NDCG@10':<8}")
print("-"*60)
print(f"{'Dense only':<25} {dense_hits[1]/total:<8.4f} {dense_hits[3]/total:<8.4f} {dense_hits[5]/total:<8.4f} {dense_hits[10]/total:<8.4f} {dense_ndcg/total:<8.4f}")
print(f"{'Dense + NEW Reranker':<25} {rerank_hits[1]/total:<8.4f} {rerank_hits[3]/total:<8.4f} {rerank_hits[5]/total:<8.4f} {rerank_hits[10]/total:<8.4f} {rerank_ndcg/total:<8.4f}")