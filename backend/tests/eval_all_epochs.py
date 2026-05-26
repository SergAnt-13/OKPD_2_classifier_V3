# backend/tests/eval_all_epochs.py
import sys, os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from config.settings import TRAINING_DATA_DIR, REFERENCE_DIR, FAISS_DIR
from backend.preprocessing.cleaner import TextCleaner
from backend.models.retriever import Retriever
import faiss

MODEL_DIR = Path("artifacts/models/bge-m3-gold-6epoch")
GOLD_PATH = TRAINING_DATA_DIR / "train.xlsx"
OKPD_PATH = REFERENCE_DIR / "okpd_2.xlsx"
ABBR_PATH = REFERENCE_DIR / "сокращения.xlsx"

# Загружаем данные
gold = pd.read_excel(GOLD_PATH, dtype=str)
gold = gold[["Номенклатура", "Код ОКПД2"]].dropna()
gold.columns = ["text", "true_code"]

okpd = pd.read_excel(OKPD_PATH, dtype=str)
code_to_name = dict(zip(okpd["code"], okpd["name"]))
cleaner = TextCleaner(abbreviations_path=ABBR_PATH)

# Все подпапки с эпохами (отсортированные)
epoch_dirs = sorted([d for d in MODEL_DIR.iterdir() if d.is_dir() and d.name.startswith("epoch")])
if not epoch_dirs:
    # Может быть, чекпоинты в формате checkpoint-*
    epoch_dirs = sorted([d for d in MODEL_DIR.iterdir() if d.is_dir() and d.name.startswith("checkpoint")])

print(f"Найдены чекпоинты: {[d.name for d in epoch_dirs]}")
results = []

for epoch_dir in epoch_dirs:
    print(f"\nОценка {epoch_dir.name} ...")
    # Для каждой эпохи строим отдельный индекс (быстрее один раз перестроить для каждой)
    model = SentenceTransformer(str(epoch_dir), device="cuda" if torch.cuda.is_available() else "cpu")
    # Строим индекс
    okpd_df = pd.read_excel(OKPD_PATH, dtype=str)
    okpd_df = okpd_df.dropna(subset=["name"])
    okpd_df["name"] = okpd_df["name"].astype(str).str.strip()
    okpd_df["stemmed"] = okpd_df["name"].apply(lambda x: cleaner.clean(x, use_stemmer=True))
    emb = model.encode(okpd_df["stemmed"].tolist(), batch_size=32, convert_to_numpy=True, show_progress_bar=True)
    faiss.normalize_L2(emb)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    # Оценка
    hits = {1:0, 5:0, 10:0}
    for _, row in tqdm(gold.iterrows(), total=len(gold), desc=f"Eval {epoch_dir.name}"):
        q = cleaner.clean(row["text"], use_stemmer=True)
        emb_q = model.encode([q], convert_to_numpy=True)
        faiss.normalize_L2(emb_q)
        scores, indices = index.search(emb_q, 10)
        for i, idx in enumerate(indices[0], 1):
            if okpd_df.iloc[idx]["code"] == row["true_code"]:
                for k in hits:
                    if i <= k:
                        hits[k] += 1
                break
    total = len(gold)
    r1 = hits[1]/total
    r10 = hits[10]/total
    results.append((epoch_dir.name, r1, r10))
    print(f"R@1={r1:.4f}, R@10={r10:.4f}")

# Вывод лучшей
best = max(results, key=lambda x: x[2])  # по R@10
print(f"\nЛучшая эпоха: {best[0]} (R@1={best[1]:.4f}, R@10={best[2]:.4f})")