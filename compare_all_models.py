# compare_all_models.py
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from tqdm import tqdm
from config.settings import TRAINING_DATA_DIR, FAISS_DIR
from backend.models.retriever import Retriever
from backend.preprocessing.cleaner import TextCleaner
from backend.preprocessing.stemmer import get_stemmer

cleaner = TextCleaner(abbreviations_path="data/reference/сокращения.xlsx")
stemmer = get_stemmer()

gold = pd.read_excel(TRAINING_DATA_DIR / "train.xlsx", dtype=str)
gold = gold[["Номенклатура", "Код ОКПД2"]].dropna()
gold.columns = ["text", "true_code"]
sample = gold.sample(min(300, len(gold)), random_state=42)

models = {
    "Baseline": "BAAI/bge-m3",
    "Finetuned v1": "artifacts/models/bge-m3-finetuned",
    "Finetuned v2": "artifacts/models/bge-m3-finetuned-v2",
}

def evaluate(model_name, label):
    ret = Retriever(model_name=model_name, index_path=FAISS_DIR/"okpd_index.faiss", id_map_path=FAISS_DIR/"id_map.csv")
    hits = {1:0, 5:0, 10:0}
    for _, row in tqdm(sample.iterrows(), total=len(sample), desc=label):
        q = cleaner.clean(row["text"], stemmer=stemmer)
        cands = ret.search(q, top_k=10)["candidates"]
        for i, c in enumerate(cands, 1):
            if c["code"] == row["true_code"]:
                for k in hits:
                    if i <= k:
                        hits[k] += 1
                break
    total = len(sample)
    return {k: v/total for k, v in hits.items()}

print("=" * 70)
print(f"Сравнение на {len(sample)} примерах")
print("=" * 70)
print(f"{'Модель':<20} {'Recall@1':<10} {'Recall@5':<10} {'Recall@10':<10}")
print("-" * 70)

for name, path in models.items():
    metrics = evaluate(path, name)
    print(f"{name:<20} {metrics[1]:<10.4f} {metrics[5]:<10.4f} {metrics[10]:<10.4f}")