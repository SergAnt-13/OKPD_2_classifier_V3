# backend/tests/save_dense_candidates.py
import sys, os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import pickle
from tqdm import tqdm
from config.settings import TRAINING_DATA_DIR, REFERENCE_DIR, FAISS_DIR
from backend.models.retriever import Retriever
from backend.preprocessing.cleaner import TextCleaner

# Параметры
MODEL_NAME = "artifacts/models/bge-m3-frozen-3epoch"
OUTPUT = Path("artifacts/dense_candidates.pkl")
TOP_K = 15   # сколько кандидатов сохранять для реранкера

gold = pd.read_excel(TRAINING_DATA_DIR / "train.xlsx", dtype=str)
gold = gold[["Номенклатура", "Код ОКПД2"]].dropna()
gold.columns = ["text", "true_code"]

cleaner = TextCleaner(abbreviations_path=REFERENCE_DIR / "сокращения.xlsx")
retriever = Retriever(
    model_name=MODEL_NAME,
    index_path=FAISS_DIR / "okpd_index.faiss",
    id_map_path=FAISS_DIR / "id_map.csv",
)

data = []  # список словарей: query, candidates, true_code
for _, row in tqdm(gold.iterrows(), total=len(gold), desc="Сохранение кандидатов"):
    q = cleaner.clean(row["text"], use_lemmatizer=True)  # леммы
    raw = retriever.search(q, top_k=TOP_K, use_reranker=False)
    # оставляем только нужные поля
    cands = [{"code": c["code"], "name": c["name"], "score": c["score"]} for c in raw["candidates"]]
    data.append({"query": q, "candidates": cands, "true_code": row["true_code"]})

with open(OUTPUT, "wb") as f:
    pickle.dump(data, f)
print(f"Сохранено {len(data)} запросов в {OUTPUT}")