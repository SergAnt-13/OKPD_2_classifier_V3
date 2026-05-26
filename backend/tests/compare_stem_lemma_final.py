# backend/tests/compare_stem_lemma_final.py
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from tqdm import tqdm
from config.settings import TRAINING_DATA_DIR, FAISS_DIR, REFERENCE_DIR
from backend.models.retriever import Retriever
from backend.preprocessing.cleaner import TextCleaner

MODEL = "artifacts/models/bge-m3-frozen-3epoch"
gold = pd.read_excel(TRAINING_DATA_DIR / "train.xlsx", dtype=str)[["Номенклатура", "Код ОКПД2"]].dropna()
gold.columns = ["text", "code"]

retriever = Retriever(model_name=MODEL,
                      index_path=FAISS_DIR / "okpd_index.faiss",
                      id_map_path=FAISS_DIR / "id_map.csv")
cleaner = TextCleaner(abbreviations_path=REFERENCE_DIR / "сокращения.xlsx")

def evaluate(mode):
    hits = {1:0, 5:0, 10:0}
    ndcg = 0.0
    for _, row in tqdm(gold.iterrows(), total=len(gold), desc=mode):
        if mode == "stem":
            q = cleaner.clean(row["text"], use_stemmer=True)
        else:  # lemmatize
            q = cleaner.clean(row["text"], use_lemmatizer=True)
        cands = retriever.search(q, top_k=10, use_reranker=False)["candidates"]
        for i, c in enumerate(cands[:10], 1):
            if c["code"] == row["code"]:
                for k in hits:
                    if i <= k: hits[k] += 1
                ndcg += 1.0 / np.log2(i+1)
                break
    total = len(gold)
    return {f"R@{k}": hits[k]/total for k in hits}, ndcg/total

stem_res, stem_ndcg = evaluate("stem")
lemma_res, lemma_ndcg = evaluate("lemma")

print("\nСтемминг:", stem_res, f"NDCG={stem_ndcg:.4f}")
print("Лемматизация:", lemma_res, f"NDCG={lemma_ndcg:.4f}")