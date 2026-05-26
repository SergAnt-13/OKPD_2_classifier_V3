# backend/tests/ab_test_reranker_gold.py
import sys, os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# GPU-флаги
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pandas as pd
import numpy as np
import torch
torch.cuda.set_device(0)
print(f"Используется устройство: {torch.cuda.get_device_name(0)}")
from tqdm import tqdm
from sentence_transformers import CrossEncoder
from config.settings import TRAINING_DATA_DIR, REFERENCE_DIR, FAISS_DIR
from backend.models.retriever import Retriever
from backend.preprocessing.cleaner import TextCleaner

# ---------- 1. Золотая выборка ----------
gold = pd.read_excel(TRAINING_DATA_DIR / "train.xlsx", dtype=str)
gold = gold[["Номенклатура", "Код ОКПД2"]].dropna()
gold.columns = ["text", "true_code"]
print(f"Золотая выборка: {len(gold)} примеров")

cleaner = TextCleaner(abbreviations_path=REFERENCE_DIR / "сокращения.xlsx")

# ---------- 2. Основной retriever ----------
ret = Retriever(
    model_name="artifacts/models/bge-m3-frozen-3epoch",
    index_path=FAISS_DIR / "okpd_index.faiss",
    id_map_path=FAISS_DIR / "id_map.csv",
)

# ---------- 3. Функция оценки с батчевым реранкером ----------
def evaluate_with_reranker(name, queries_df, retriever, reranker=None, use_reranker=False):
    hits = {1:0, 3:0, 5:0, 10:0}
    ndcg_sum = 0.0
    total = len(queries_df)

    # Шаг 1: получаем кандидатов для всех запросов (dense retrieval)
    cleaned_queries = []
    all_candidates = []
    true_codes = []
    for _, row in tqdm(queries_df.iterrows(), total=total, desc=f"{name} (dense)"):
        q = cleaner.clean(row["text"], use_lemmatizer=True)
        raw = retriever.search(q, top_k=10, use_reranker=use_reranker)
        cleaned_queries.append(q)
        all_candidates.append(raw["candidates"][:15])
        true_codes.append(row["true_code"])

    # # Шаг 2: если есть внешний реранкер, применяем его ко всем парам за один проход
    if reranker:
        all_pairs = []
        for q, candidates in zip(cleaned_queries, all_candidates):
            for c in candidates:
                all_pairs.append((c["name"], q))

        print(f"Переранжирование {len(all_pairs)} пар...")
        scores = reranker.predict(all_pairs, batch_size=16, show_progress_bar=True, max_length=256)
        idx = 0
        for candidates in all_candidates:
            for c in candidates:
                c["rerank_score"] = float(scores[idx])
                idx += 1
            candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

    # Шаг 3: считаем метрики
    for true_code, candidates in zip(true_codes, all_candidates):
        for i, c in enumerate(candidates[:10], 1):
            if c["code"] == true_code:
                for k in hits:
                    if i <= k:
                        hits[k] += 1
                ndcg_sum += 1.0 / np.log2(i + 1)
                break

    return {k: hits[k]/total for k in hits}, ndcg_sum/total

# ---------- 4. Запуск ----------
print("\n=== Dense only ===")
dense_metrics, dense_ndcg = evaluate_with_reranker("Dense only", gold, ret, use_reranker=False)

# print("\n=== Dense + OLD Reranker ===")
# reranker_old = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=256, device="cuda")
# reranker_old.model.to("cuda")
# old_metrics, old_ndcg = evaluate_with_reranker("OLD Reranker", gold, ret, reranker=reranker_old, use_reranker=False)
# del reranker_old
# torch.cuda.empty_cache()

print("\n=== Dense + NEW Reranker ===")
reranker_new = CrossEncoder("artifacts/models/reranker-final", max_length=256, device="cuda")
reranker_new.model.to("cuda")
new_metrics, new_ndcg = evaluate_with_reranker("NEW Reranker", gold, ret, reranker=reranker_new, use_reranker=False)
del reranker_new
torch.cuda.empty_cache()

# ---------- 5. Итоговая таблица ----------
print("\n" + "="*70)
print("A/B-ТЕСТ НА ВСЕЙ ЗОЛОТОЙ ВЫБОРКЕ (1475 примеров)")
print("="*70)
print(f"{'Конфигурация':<25} {'R@1':<8} {'R@3':<8} {'R@5':<8} {'R@10':<8} {'NDCG@10':<8}")
print("-"*70)
print(f"{'Dense only':<25} {dense_metrics[1]:<8.4f} {dense_metrics[3]:<8.4f} {dense_metrics[5]:<8.4f} {dense_metrics[10]:<8.4f} {dense_ndcg:<8.4f}")
print(f"{'Dense + OLD Reranker':<25} {old_metrics[1]:<8.4f} {old_metrics[3]:<8.4f} {old_metrics[5]:<8.4f} {old_metrics[10]:<8.4f} {old_ndcg:<8.4f}")
print(f"{'Dense + NEW Reranker':<25} {new_metrics[1]:<8.4f} {new_metrics[3]:<8.4f} {new_metrics[5]:<8.4f} {new_metrics[10]:<8.4f} {new_ndcg:<8.4f}")