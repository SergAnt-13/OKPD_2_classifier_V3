# backend/training/train_bgem3_frozen.py
"""
Дообучение BGE‑M3 с заморозкой нижних 6 слоёв (3 эпохи).
После обучения — сравнение с текущим чемпионом (epoch2) на всей золотой выборке.
"""
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
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
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
from config.settings import TRAINING_DATA_DIR, RAW_DATA_DIR, REFERENCE_DIR, FAISS_DIR
from backend.preprocessing.cleaner import TextCleaner

# ============================================================
# 1. ДАННЫЕ И ПАРЫ (как в V2)
# ============================================================
gold = pd.read_excel(TRAINING_DATA_DIR / "train.xlsx", dtype=str)
gold = gold[["Номенклатура", "Код ОКПД2"]].dropna()
gold.columns = ["text", "code"]

prom = pd.read_excel(RAW_DATA_DIR / "all_nomenclature.xlsx", dtype=str)
prom = prom[['nomenclature', 'okpd2_code']].dropna()
prom.columns = ["text", "code"]

all_data = pd.concat([gold, prom], ignore_index=True)
food_prefixes = ('01', '02', '03', '10')
all_data = all_data[all_data['code'].str.startswith(food_prefixes, na=False)]
print(f"Пищевых товаров: {len(all_data)}")

ref_path = REFERENCE_DIR / "okpd_2_normalized_stemmed.csv"
if ref_path.exists():
    ref = pd.read_csv(ref_path, dtype=str)
else:
    okpd = pd.read_excel(REFERENCE_DIR / "okpd_2.xlsx", dtype=str)
    cleaner_tmp = TextCleaner(abbreviations_path=REFERENCE_DIR / "сокращения.xlsx")
    okpd["name_normalized"] = okpd["name"].apply(lambda x: cleaner_tmp.clean(x, use_stemmer=True))
    ref = okpd[["code", "parent_code", "name", "name_normalized"]]
    ref.to_csv(ref_path, index=False)
code_to_name = dict(zip(ref["code"], ref["name_normalized"]))

cleaner = TextCleaner(abbreviations_path=REFERENCE_DIR / "сокращения.xlsx")
pairs = []
for _, row in tqdm(all_data.iterrows(), total=len(all_data), desc="Пары"):
    text = cleaner.clean(row['text'], use_stemmer=True)
    target = code_to_name.get(row['code'].strip())
    if text and target:
        pairs.append((text, target))
print(f"Обучающих пар: {len(pairs)}")

train_pairs, _ = train_test_split(pairs, test_size=0.1, random_state=42)
train_examples = [InputExample(texts=[p[0], p[1]]) for p in train_pairs]

# ============================================================
# 2. МОДЕЛЬ С ЗАМОРОЗКОЙ НИЖНИХ 6 СЛОЁВ
# ============================================================
model = SentenceTransformer("BAAI/bge-m3", device="cpu")
model.to("cpu")

# Замораживаем слои 0-5 (первые 6 из 12)
for name, param in model.named_parameters():
    if name.startswith("0.encoder.layer.") or name.startswith("1.encoder.layer."):
        parts = name.split(".")
        layer_num = int(parts[2]) if len(parts) > 2 else -1
        if layer_num < 6:
            param.requires_grad = False

print("Заморожены нижние 6 слоёв.")

train_dataloader = torch.utils.data.DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = MultipleNegativesRankingLoss(model)

print("Обучение 3 эпохи с заморозкой...")
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100,
    output_path="artifacts/models/bge-m3-frozen-3epoch",
    save_best_model=False,
    show_progress_bar=True,
)

# ============================================================
# 3. СРАВНЕНИЕ С ТЕКУЩИМ ЧЕМПИОНОМ (epoch2) НА ВСЕЙ ЗОЛОТОЙ ВЫБОРКЕ
# ============================================================
def evaluate_model(model_path, label):
    # Загружаем золотую выборку с правильными колонками
    gold_eval = pd.read_excel(TRAINING_DATA_DIR / "train.xlsx", dtype=str)
    gold_eval = gold_eval[["Номенклатура", "Код ОКПД2"]].dropna()
    gold_eval.columns = ["text", "true_code"]

    model = SentenceTransformer(model_path, device="cpu")
    okpd = pd.read_excel(REFERENCE_DIR / "okpd_2.xlsx", dtype=str)
    okpd = okpd.dropna(subset=["name"])
    okpd["name"] = okpd["name"].astype(str).str.strip()
    okpd["name_stemmed"] = okpd["name"].apply(lambda x: cleaner.clean(x, use_stemmer=True))
    emb = model.encode(okpd["name_stemmed"].tolist(), batch_size=8,
                       convert_to_numpy=True, show_progress_bar=True)
    faiss.normalize_L2(emb)
    idx = faiss.IndexFlatIP(emb.shape[1])
    idx.add(emb)

    hits = {1:0, 3:0, 5:0, 10:0}
    ndcg_sum = 0.0
    total = 0
    for _, row in tqdm(gold_eval.iterrows(), total=len(gold_eval), desc=f"Оценка {label}"):
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
    return {
        "Recall@1": hits[1]/total, "Recall@3": hits[3]/total,
        "Recall@5": hits[5]/total, "Recall@10": hits[10]/total,
        "NDCG@10": ndcg_sum/total,
    }

print("\nОценка frozen модели...")
res_frozen = evaluate_model("artifacts/models/bge-m3-frozen-3epoch", "Frozen 3 epoch")
print("Оценка epoch2...")
res_epoch2 = evaluate_model("artifacts/models/bge-m3-food-only-epoch2", "Epoch 2")

print("\n" + "="*70)
print("СРАВНЕНИЕ НА ВСЕЙ ЗОЛОТОЙ ВЫБОРКЕ (1475 примеров)")
print("="*70)
print(f"{'Модель':<25} {'R@1':<8} {'R@3':<8} {'R@5':<8} {'R@10':<8} {'NDCG@10':<8}")
print("-"*70)
for name, m in [("Epoch 2 (чемпион)", res_epoch2), ("Frozen 3 epoch", res_frozen)]:
    print(f"{name:<25} {m['Recall@1']:<8.4f} {m['Recall@3']:<8.4f} {m['Recall@5']:<8.4f} {m['Recall@10']:<8.4f} {m['NDCG@10']:<8.4f}")