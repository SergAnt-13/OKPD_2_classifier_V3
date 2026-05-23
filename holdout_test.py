# holdout_test.py — честный hold‑out тест для OKPD‑2 Classifier V3
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
from config.settings import TRAINING_DATA_DIR, REFERENCE_DIR, FAISS_DIR
from backend.preprocessing.cleaner import TextCleaner
from backend.models.retriever import Retriever, build_faiss_index

# --- 1. Загружаем золотую выборку ---
gold = pd.read_excel(TRAINING_DATA_DIR / "train.xlsx", dtype=str)
gold = gold[["Номенклатура", "Код ОКПД2"]].dropna()
gold.columns = ["text", "true_code"]

# --- 2. Hold‑out разбиение (300 тест, остальное train) ---
train_df, test_df = train_test_split(gold, test_size=300, random_state=42)
print(f"Train: {len(train_df)}, Test: {len(test_df)}")

# --- 3. Загружаем справочник ОКПД‑2 ---
okpd = pd.read_excel(REFERENCE_DIR / "okpd_2.xlsx", dtype=str)
code_to_name = dict(zip(okpd["code"], okpd["name"]))

cleaner = TextCleaner(abbreviations_path=REFERENCE_DIR / "сокращения.xlsx")
stemmer = get_stemmer()

# --- 4. Строим обучающие пары ---
train_pairs = []
for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Train pairs"):
    text = cleaner.clean(row["text"], use_stemmer=True)
    target = code_to_name.get(row["true_code"].strip())
    if text and target:
        train_pairs.append((text, target))

# --- 5. Дообучаем модель на train (на CPU, с офлайн‑флагами) ---
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

model = SentenceTransformer("BAAI/bge-m3", device="cpu")
model.to("cpu")

train_examples = [InputExample(texts=[p[0], p[1]]) for p in train_pairs]
train_dataloader = torch.utils.data.DataLoader(train_examples, shuffle=True, batch_size=4, pin_memory=False)
train_loss = MultipleNegativesRankingLoss(model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    warmup_steps=100,
    output_path="artifacts/models/holdout_model",
    show_progress_bar=True,
)

# --- 6. Перестраиваем FAISS‑индекс под новую модель ---
build_faiss_index(model_name="artifacts/models/holdout_model")

# --- 7. Замеряем Recall@1, @5, @10 на hold‑out ---
ret = Retriever(model_name="artifacts/models/holdout_model",
                index_path=FAISS_DIR/"okpd_index.faiss",
                id_map_path=FAISS_DIR/"id_map.csv")

hits = {1:0, 5:0, 10:0}
for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Hold‑out eval"):
    q = cleaner.clean(row["text"], use_stemmer=True)
    cands = ret.search(q, top_k=10)["candidates"]
    for i, c in enumerate(cands, 1):
        if c["code"] == row["true_code"]:
            for k in hits:
                if i <= k:
                    hits[k] += 1
            break

total = len(test_df)
print("\n=== Hold‑out results ===")
for k in [1, 5, 10]:
    print(f"Recall@{k}: {hits[k]/total:.4f}")