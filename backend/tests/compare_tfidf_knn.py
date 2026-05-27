# backend/tests/compare_tfidf_knn.py
import sys, os, json, math
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from collections import defaultdict
from config.settings import TRAINING_DATA_DIR, REFERENCE_DIR, FAISS_DIR
from backend.preprocessing.cleaner import TextCleaner
from backend.models.retriever import Retriever

# ---------- Загрузка данных ----------
gold = pd.read_excel(TRAINING_DATA_DIR / "train.xlsx", dtype=str)[["Номенклатура", "Код ОКПД2"]].dropna()
gold.columns = ["text", "code"]

okpd = pd.read_excel(REFERENCE_DIR / "okpd_2.xlsx", dtype=str)
okpd = okpd.dropna(subset=["name"])
okpd["name"] = okpd["name"].astype(str).str.strip()

cleaner = TextCleaner(abbreviations_path=REFERENCE_DIR / "сокращения.xlsx")
okpd["stemmed"] = okpd["name"].apply(lambda x: cleaner.clean(x, use_stemmer=True))

# Частоты классов в золоте для tail breakdown
code_counts = gold["code"].value_counts().to_dict()

# ---------- TF-IDF + kNN ----------
print("=== TF-IDF + kNN ===")
vectorizer = TfidfVectorizer(
    max_features=50000,
    sublinear_tf=True,
    analyzer='char_wb',
    ngram_range=(3, 5)
)
corpus_vectors = vectorizer.fit_transform(okpd["stemmed"].tolist())
knn = NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='brute')
knn.fit(corpus_vectors)

tfidf_results = []   # list of dict: {query, true_code, ranks, scores}
for _, row in tqdm(gold.iterrows(), total=len(gold), desc="TF-IDF"):
    q = cleaner.clean(row["text"], use_stemmer=True)
    q_vec = vectorizer.transform([q])
    distances, indices = knn.kneighbors(q_vec, n_neighbors=10)
    # расстояния - это косинусное расстояние, переведём в "сходство" для единообразия
    sim_scores = 1.0 - distances[0]
    ranks = []
    true_code = row["code"]
    found = False
    for i, idx in enumerate(indices[0]):
        if okpd.iloc[idx]["code"] == true_code:
            ranks.append(i+1)
            found = True
            break
    if not found:
        ranks.append(np.inf)   # не попал
    tfidf_results.append({
        "true_code": true_code,
        "ranks": ranks,
        "top_score": sim_scores[0] if len(sim_scores) > 0 else 0.0,
        "scores": sim_scores,
        "codes": [okpd.iloc[idx]["code"] for idx in indices[0]],
        "code_counts": code_counts.get(true_code, 0)
    })

# ---------- Наш чемпион ----------
print("\n=== BGE-M3 (frozen-3epoch) ===")
retriever = Retriever(
    model_name="artifacts/models/bge-m3-frozen-3epoch",
    index_path=FAISS_DIR / "okpd_index.faiss",
    id_map_path=FAISS_DIR / "id_map.csv",
)
champion_results = []
for _, row in tqdm(gold.iterrows(), total=len(gold), desc="Champion"):
    q = cleaner.clean(row["text"], use_stemmer=True)
    res = retriever.search(q, top_k=10, use_reranker=False)
    candidates = res["candidates"]
    true_code = row["code"]
    rank = np.inf
    top_score = candidates[0]["score"] if candidates else 0.0
    scores = [c["score"] for c in candidates]
    codes = [c["code"] for c in candidates]
    for i, c in enumerate(candidates):
        if c["code"] == true_code:
            rank = i+1
            break
    champion_results.append({
        "true_code": true_code,
        "rank": rank,
        "top_score": top_score,
        "scores": scores,
        "codes": codes,
        "code_counts": code_counts.get(true_code, 0)
    })

# ---------- Функции расчёта метрик ----------
def compute_recall_at_k(results, k):
    hits = sum(1 for r in results if r["rank"] <= k)
    return hits / len(results) if results else 0

def compute_precision_at_k(results, k):
    precisions = []
    for r in results:
        if r["rank"] <= k:
            precisions.append(1.0 / k)
        else:
            precisions.append(0.0)
    return np.mean(precisions) if precisions else 0

def compute_mrr(results):
    rr = [1.0/r["rank"] if r["rank"] != np.inf else 0.0 for r in results]
    return np.mean(rr) if rr else 0

def compute_ndcg(results, k=10):
    dcg_sum = 0.0
    for r in results:
        if r["rank"] <= k:
            dcg_sum += 1.0 / math.log2(r["rank"] + 1)
    return dcg_sum / len(results) if results else 0

def compute_margin(results):
    margins = []
    for r in results:
        if len(r["scores"]) >= 2:
            margins.append(r["scores"][0] - r["scores"][1])
        else:
            margins.append(0.0)
    return np.mean(margins), np.median(margins)

def tail_breakdown(results, groups=[(0,1), (1,5), (5, float('inf'))], k=10):
    breakdown = {}
    for low, high in groups:
        subset = [r for r in results if r["code_counts"] > low and r["code_counts"] <= high]
        recall = compute_recall_at_k(subset, k)
        breakdown[f"{low+1}-{int(high) if high != float('inf') else 'inf'}"] = {
            "recall": recall,
            "support": len(subset)
        }
    return breakdown

def calibration_curve(results, bins=5):
    scores = np.array([r["top_score"] for r in results])
    correct = np.array([1 if r["rank"] <= 1 else 0 for r in results])  # топ-1 корректен
    bin_edges = np.linspace(0, 1, bins+1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_correct = []
    bin_total = []
    for i in range(bins):
        mask = (scores >= bin_edges[i]) & (scores < bin_edges[i+1])
        bin_total.append(mask.sum())
        if bin_total[-1] > 0:
            bin_correct.append(correct[mask].mean())
        else:
            bin_correct.append(np.nan)
    return bin_centers, bin_correct, bin_total

# ---------- Вычисление метрик ----------
print("\nМетрики TF-IDF+kNN:")
tfidf_ranks = [r["ranks"][0] for r in tfidf_results]
# для удобства добавим rank в tfidf_results
for r in tfidf_results:
    r["rank"] = r["ranks"][0]
print(f"Recall@1: {compute_recall_at_k(tfidf_results, 1):.4f}")
print(f"Recall@5: {compute_recall_at_k(tfidf_results, 5):.4f}")
print(f"Recall@10: {compute_recall_at_k(tfidf_results, 10):.4f}")
print(f"MRR: {compute_mrr(tfidf_results):.4f}")
print(f"NDCG@10: {compute_ndcg(tfidf_results, 10):.4f}")

print("\nМетрики Champion (frozen-3epoch):")
print(f"Recall@1: {compute_recall_at_k(champion_results, 1):.4f}")
print(f"Recall@5: {compute_recall_at_k(champion_results, 5):.4f}")
print(f"Recall@10: {compute_recall_at_k(champion_results, 10):.4f}")
print(f"MRR: {compute_mrr(champion_results):.4f}")
print(f"NDCG@10: {compute_ndcg(champion_results, 10):.4f}")
mean_margin, median_margin = compute_margin(champion_results)
print(f"Mean Margin: {mean_margin:.4f}, Median Margin: {median_margin:.4f}")

print("\nTail breakdown (Recall@10):")
print("Champion:")
champion_tail = tail_breakdown(champion_results, groups=[(0,1), (1,5), (5, float('inf'))])
for group, vals in champion_tail.items():
    print(f"  {group}: recall={vals['recall']:.4f}, support={vals['support']}")
print("TF-IDF:")
tfidf_tail = tail_breakdown(tfidf_results, groups=[(0,1), (1,5), (5, float('inf'))])
for group, vals in tfidf_tail.items():
    print(f"  {group}: recall={vals['recall']:.4f}, support={vals['support']}")

print("\nCalibration curve (Champion топ-1 скор vs actual recall):")
bin_centers, bin_correct, bin_total = calibration_curve(champion_results)
for c, corr, tot in zip(bin_centers, bin_correct, bin_total):
    if tot > 0:
        print(f"  {c:.2f}: actual recall={corr:.3f}, total={tot}")
    else:
        print(f"  {c:.2f}: no data")

# Сохраним таблицы для отчёта
pd.DataFrame({
    "Metric": ["Recall@1","Recall@5","Recall@10","MRR","NDCG@10"],
    "TF-IDF+kNN": [compute_recall_at_k(tfidf_results,1), compute_recall_at_k(tfidf_results,5), compute_recall_at_k(tfidf_results,10), compute_mrr(tfidf_results), compute_ndcg(tfidf_results,10)],
    "BGE-M3 (frozen-3epoch)": [compute_recall_at_k(champion_results,1), compute_recall_at_k(champion_results,5), compute_recall_at_k(champion_results,10), compute_mrr(champion_results), compute_ndcg(champion_results,10)]
}).to_csv("metrics_comparison.csv", index=False)

# Если нужно, можно добавить график calibration curve