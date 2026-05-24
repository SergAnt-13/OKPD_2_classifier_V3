# eval_epoch2_full.py
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pandas as pd
import torch
torch.set_default_device("cpu")
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from config.settings import TRAINING_DATA_DIR, REFERENCE_DIR, FAISS_DIR
from backend.preprocessing.cleaner import TextCleaner

# Загружаем золотую выборку
gold = pd.read_excel(TRAINING_DATA_DIR / "train.xlsx", dtype=str)
gold = gold[["Номенклатура", "Код ОКПД2"]].dropna()
gold.columns = ["text", "true_code"]
print(f"Золотая выборка: {len(gold)} примеров")

# Загружаем модель после 2 эпох
model = SentenceTransformer("artifacts/models/bge-m3-food-only-epoch5", device="cpu")

# Строим стеммированный FAISS-индекс
cleaner = TextCleaner(abbreviations_path=REFERENCE_DIR / "сокращения.xlsx")
okpd = pd.read_excel(REFERENCE_DIR / "okpd_2.xlsx", dtype=str)
okpd = okpd.dropna(subset=["name"])
okpd["name"] = okpd["name"].astype(str).str.strip()
okpd["name_stemmed"] = okpd["name"].apply(lambda x: cleaner.clean(x, use_stemmer=True))

print("Строим стеммированный индекс...")
emb = model.encode(okpd["name_stemmed"].tolist(), batch_size=8, convert_to_numpy=True, show_progress_bar=True)
faiss.normalize_L2(emb)
idx = faiss.IndexFlatIP(emb.shape[1])
idx.add(emb)

# Оценка
hits = {1:0, 3:0, 5:0, 10:0}
ndcg_sum = 0.0
total = 0
for _, row in tqdm(gold.iterrows(), total=len(gold), desc="Оценка epoch5"):
    q = cleaner.clean(row["text"], use_stemmer=True)
    emb_q = model.encode([q], convert_to_numpy=True, show_progress_bar=False)
    faiss.normalize_L2(emb_q)
    _, indices = idx.search(emb_q, 10)
    retrieved = [okpd.iloc[i]["code"] for i in indices[0] if 0 <= i < len(okpd)]
    true_code = row["true_code"]
    for k in hits:
        if true_code in retrieved[:k]:
            hits[k] += 1
    for i, code in enumerate(retrieved[:10]):
        if code == true_code:
            ndcg_sum += 1.0 / np.log2(i + 2)
            break
    total += 1

print(f"\nEpoch 5 (вся золотая выборка):")
print(f"Recall@1:  {hits[1]/total:.4f}")
print(f"Recall@3:  {hits[3]/total:.4f}")
print(f"Recall@5:  {hits[5]/total:.4f}")
print(f"Recall@10: {hits[10]/total:.4f}")
print(f"NDCG@10:   {ndcg_sum/total:.4f}")