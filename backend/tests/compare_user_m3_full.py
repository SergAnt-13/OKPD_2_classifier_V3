# backend/tests/compare_user_m3_full.py
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from tqdm import tqdm
from config.settings import TRAINING_DATA_DIR, REFERENCE_DIR, FAISS_DIR
from backend.models.retriever import Retriever
from backend.preprocessing.cleaner import TextCleaner

# Загружаем ВСЮ золотую выборку
gold = pd.read_excel(TRAINING_DATA_DIR / "train.xlsx", dtype=str)
gold = gold[["Номенклатура", "Код ОКПД2"]].dropna()
gold.columns = ["text", "true_code"]
print(f"Золотая выборка: {len(gold)} примеров")

cleaner = TextCleaner(abbreviations_path=REFERENCE_DIR / "сокращения.xlsx")

models = {
    "BGE-M3 frozen epoch2 (чемпион)": "artifacts/models/bge-m3-frozen-stratified-epoch2",
    "USER-bge-m3 nofreeze": "artifacts/models/user-bge-m3-nofreeze",
}

results = {}
for name, model_path in models.items():
    ret = Retriever(model_name=model_path, index_path=FAISS_DIR/"okpd_index.faiss", id_map_path=FAISS_DIR/"id_map.csv")
    hits = {1:0, 3:0, 5:0, 10:0}
    ndcg_sum = 0.0
    for _, row in tqdm(gold.iterrows(), total=len(gold), desc=name):
        q = cleaner.clean(row["text"], use_stemmer=True)
        cands = ret.search(q, top_k=10)["candidates"]
        for i, c in enumerate(cands, 1):
            if c["code"] == row["true_code"]:
                for k in hits:
                    if i <= k:
                        hits[k] += 1
                ndcg_sum += 1.0 / np.log2(i + 1)
                break
    total = len(gold)
    results[name] = {k: hits[k]/total for k in hits}
    results[name]["NDCG@10"] = ndcg_sum / total

print("\n" + "="*70)
print("ФИНАЛЬНОЕ СРАВНЕНИЕ (вся золотая выборка, 1475 примеров)")
print("="*70)
print(f"{'Модель':<35} {'R@1':<8} {'R@3':<8} {'R@5':<8} {'R@10':<8} {'NDCG@10':<8}")
print("-"*70)
for name, metrics in results.items():
    print(f"{name:<35} {metrics[1]:<8.4f} {metrics[3]:<8.4f} {metrics[5]:<8.4f} {metrics[10]:<8.4f} {metrics['NDCG@10']:<8.4f}")