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
torch.cuda.set_device(0)  # принудительно выбрать первую видеокарту
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

# ---------- 2. Модели ----------
# Dense retriever (чемпион)
ret = Retriever(
    model_name="artifacts/models/bge-m3-frozen-stratified-epoch2",
    index_path=FAISS_DIR / "okpd_index.faiss",
    id_map_path=FAISS_DIR / "id_map.csv",
)

reranker_old = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512, device="cuda")
reranker_old.model.to("cuda")
print("Старый реранкер на:", next(reranker_old.model.parameters()).device)

reranker_new = CrossEncoder("artifacts/models/reranker-final", max_length=512, device="cuda")
reranker_new.model.to("cuda")
print("Новый реранкер на:", next(reranker_new.model.parameters()).device)

# ---------- 3. Функция оценки ----------
def evaluate_with_reranker(name, queries_df, reranker=None):
    hits = {1:0, 3:0, 5:0, 10:0}
    ndcg_sum = 0.0
    for _, row in tqdm(queries_df.iterrows(), total=len(queries_df), desc=name):
        q = cleaner.clean(row["text"], use_stemmer=True)
        # Получаем топ-20 от dense
        raw = ret.search(q, top_k=20)
        candidates = raw["candidates"]

        if reranker:
            pairs = [(c["name"], q) for c in candidates]
            scores = reranker.predict(pairs, show_progress_bar=False)
            for c, s in zip(candidates, scores):
                c["rerank_score"] = float(s)
            candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

        for i, c in enumerate(candidates[:10], 1):
            if c["code"] == row["true_code"]:
                for k in hits:
                    if i <= k:
                        hits[k] += 1
                ndcg_sum += 1.0 / np.log2(i + 1)
                break
    total = len(queries_df)
    return {k: hits[k]/total for k in hits}, ndcg_sum/total

# ---------- 4. Запуск ----------
print("\n=== Dense only ===")
dense_metrics, dense_ndcg = evaluate_with_reranker("Dense only", gold)

print("\n=== Dense + OLD Reranker ===")
old_metrics, old_ndcg = evaluate_with_reranker("OLD Reranker", gold, reranker_old)

print("\n=== Dense + NEW Reranker ===")
new_metrics, new_ndcg = evaluate_with_reranker("NEW Reranker", gold, reranker_new)

# ---------- 5. Итоговая таблица ----------
print("\n" + "="*70)
print("A/B-ТЕСТ НА ВСЕЙ ЗОЛОТОЙ ВЫБОРКЕ (1475 примеров)")
print("="*70)
print(f"{'Конфигурация':<25} {'R@1':<8} {'R@3':<8} {'R@5':<8} {'R@10':<8} {'NDCG@10':<8}")
print("-"*70)
print(f"{'Dense only':<25} {dense_metrics[1]:<8.4f} {dense_metrics[3]:<8.4f} {dense_metrics[5]:<8.4f} {dense_metrics[10]:<8.4f} {dense_ndcg:<8.4f}")
print(f"{'Dense + OLD Reranker':<25} {old_metrics[1]:<8.4f} {old_metrics[3]:<8.4f} {old_metrics[5]:<8.4f} {old_metrics[10]:<8.4f} {old_ndcg:<8.4f}")
print(f"{'Dense + NEW Reranker':<25} {new_metrics[1]:<8.4f} {new_metrics[3]:<8.4f} {new_metrics[5]:<8.4f} {new_metrics[10]:<8.4f} {new_ndcg:<8.4f}")