# backend/tests/evaluate_reranker_dity.py
# Сравнивает новый реранкер DiTy с Dense-only режимом

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from collections import Counter
from sentence_transformers import CrossEncoder
from config.settings import TRAINING_DATA_DIR, REFERENCE_DIR, FAISS_DIR
from backend.models.retriever import Retriever
from backend.preprocessing.cleaner import TextCleaner

# Загружаем золотую выборку
gold = pd.read_excel(TRAINING_DATA_DIR / "train.xlsx", dtype=str)
gold = gold[["Номенклатура", "Код ОКПД2"]].dropna()
gold.columns = ["text", "true_code"]

# Стратифицированный сплит (80/20)
code_counts = Counter(gold["true_code"])
sorted_codes = [c for c, _ in code_counts.most_common()]
n_head = int(len(sorted_codes) * 0.2)
n_mid = int(len(sorted_codes) * 0.5)
head_codes = set(sorted_codes[:n_head])
mid_codes = set(sorted_codes[n_head:n_mid])
tail_codes = set(sorted_codes[n_mid:])

def get_group(code):
    if code in head_codes: return "head"
    if code in mid_codes: return "mid"
    return "tail"

gold["group"] = gold["true_code"].apply(get_group)
train_df, test_df = train_test_split(gold, test_size=0.2, random_state=42, stratify=gold["group"])
train_codes = set(train_df["true_code"])
test_df = test_df[test_df["true_code"].isin(train_codes)]

print(f"Test: {len(test_df)} примеров")

cleaner = TextCleaner(abbreviations_path=REFERENCE_DIR / "сокращения.xlsx")
retriever = Retriever(model_name="artifacts/models/bge-m3-frozen-stratified-epoch2",
                      index_path=FAISS_DIR / "okpd_index.faiss",
                      id_map_path=FAISS_DIR / "id_map.csv")

# 🆕 Новый специализированный реранкер
cross_encoder = CrossEncoder("DiTy/cross-encoder-russian-msmarco", max_length=512)

def evaluate_with_reranker(queries_df):
    hits = {1:0, 3:0, 5:0, 10:0}
    ndcg_sum = 0.0
    for _, row in tqdm(queries_df.iterrows(), total=len(queries_df), desc="DiTy Reranker"):
        q = cleaner.clean(row["text"], use_stemmer=True)
        raw = retriever.search(q, top_k=20)
        candidates = raw["candidates"]
        pairs = [(c["name"], q) for c in candidates]
        scores = cross_encoder.predict(pairs, show_progress_bar=False)
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

def evaluate_dense_only(queries_df):
    hits = {1:0, 3:0, 5:0, 10:0}
    ndcg_sum = 0.0
    for _, row in tqdm(queries_df.iterrows(), total=len(queries_df), desc="Dense only"):
        q = cleaner.clean(row["text"], use_stemmer=True)
        raw = retriever.search(q, top_k=10)
        candidates = raw["candidates"]
        for i, c in enumerate(candidates, 1):
            if c["code"] == row["true_code"]:
                for k in hits:
                    if i <= k:
                        hits[k] += 1
                ndcg_sum += 1.0 / np.log2(i + 1)
                break
    total = len(queries_df)
    return {k: hits[k]/total for k in hits}, ndcg_sum/total

print("\nЗамер Dense only...")
res_dense, ndcg_dense = evaluate_dense_only(test_df)
print("Замер Dense + DiTy Reranker...")
res_dity, ndcg_dity = evaluate_with_reranker(test_df)

print("\n" + "="*70)
print("СРАВНЕНИЕ: Dense vs Dense + DiTy Reranker")
print("="*70)
print(f"{'Конфигурация':<25} {'R@1':<8} {'R@3':<8} {'R@5':<8} {'R@10':<8} {'NDCG@10':<8}")
print("-"*70)
print(f"{'Dense only':<25} {res_dense[1]:<8.4f} {res_dense[3]:<8.4f} {res_dense[5]:<8.4f} {res_dense[10]:<8.4f} {ndcg_dense:<8.4f}")
print(f"{'Dense + DiTy Reranker':<25} {res_dity[1]:<8.4f} {res_dity[3]:<8.4f} {res_dity[5]:<8.4f} {res_dity[10]:<8.4f} {ndcg_dity:<8.4f}")