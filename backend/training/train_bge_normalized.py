# backend/training/train_bge_normalized.py
import sys, os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import torch
import faiss
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
from sentence_transformers.sentence_transformer.evaluation import EmbeddingSimilarityEvaluator

from config.settings import TRAINING_DATA_DIR, REFERENCE_DIR, FAISS_DIR
from backend.preprocessing.cleaner import TextCleaner

# ---------- Настройки ----------
BGE_BASE = "deepvk/USER-bge-m3"
OUTPUT_DIR = Path("artifacts/models/bge-m3-user-rus-gold")
FREEZE_LAYERS = 3
MAX_EPOCHS = 15
PATIENCE = 3
BATCH_SIZE = 16
LR = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NORMALIZED_GOLD = "gold_normalized.csv"
OKPD_PATH = REFERENCE_DIR / "okpd_2.xlsx"
ABBR_PATH = REFERENCE_DIR / "сокращения.xlsx"
# --------------------------------

cleaner = TextCleaner(abbreviations_path=ABBR_PATH)

# 1. Загружаем нормализованные названия и коды
gold = pd.read_csv(NORMALIZED_GOLD, dtype=str)


original_gold = pd.read_excel(TRAINING_DATA_DIR / "train.xlsx", dtype=str)[["Номенклатура", "Код ОКПД2"]].dropna()
original_gold.columns = ["text_original", "code"]

# Объединяем по исходному названию (может быть не идеально, но для первого эксперимента сойдёт)
gold = gold.merge(original_gold, on="text_original", how="inner")
print(f"Записей после слияния с кодами: {len(gold)}")

# 2. Подготовка стеммированных пар
okpd = pd.read_excel(OKPD_PATH, dtype=str)
code_to_name = {}
for _, r in okpd.iterrows():
    code_to_name[r["code"]] = cleaner.clean(r["name"], use_stemmer=True)

pairs = []
for _, row in tqdm(gold.iterrows(), total=len(gold), desc="Пары"):
    q = cleaner.clean(row["text_normalized"], use_stemmer=True)   # стеммим нормализованный запрос
    target = code_to_name.get(row["code"].strip())
    if q and target:
        pairs.append((q, target))

# Разделение на train/eval
train_pairs, eval_pairs = train_test_split(pairs, test_size=0.2, random_state=42)
eval_s1 = [p[0] for p in eval_pairs]
eval_s2 = [p[1] for p in eval_pairs]
print(f"Train: {len(train_pairs)}, Eval: {len(eval_pairs)}")

# 3. Модель с заморозкой
model = SentenceTransformer(BGE_BASE, device=DEVICE)
encoder_layers = model._first_module().auto_model.encoder.layer
for i, layer in enumerate(encoder_layers):
    if i < FREEZE_LAYERS:
        for param in layer.parameters():
            param.requires_grad = False
for param in model._first_module().auto_model.embeddings.parameters():
    param.requires_grad = False
print(f"Заморожены первые {FREEZE_LAYERS} слоёв и эмбеддинги")

train_loss = MultipleNegativesRankingLoss(model)
train_examples = [InputExample(texts=[p[0], p[1]]) for p in train_pairs]
train_dataloader = torch.utils.data.DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)

# 4. Обучение с early stopping
evaluator = EmbeddingSimilarityEvaluator(
    sentences1=eval_s1,
    sentences2=eval_s2,
    scores=[1.0] * len(eval_s1),
    name="eval",
    show_progress_bar=False,
)

def evaluate_r10(model):
    model.eval()
    with torch.no_grad():
        emb_q = model.encode(eval_s1, batch_size=BATCH_SIZE, show_progress_bar=False, convert_to_numpy=True)
        emb_d = model.encode(eval_s2, batch_size=BATCH_SIZE, show_progress_bar=False, convert_to_numpy=True)
        faiss.normalize_L2(emb_q)
        faiss.normalize_L2(emb_d)
        index = faiss.IndexFlatIP(emb_d.shape[1])
        index.add(emb_d)
        scores, indices = index.search(emb_q, 10)
        hits = sum(1 for i in range(len(eval_s1)) if i in indices[i])
        r10 = hits / len(eval_s1)
    model.train()
    return r10

best_r10 = 0.0
best_epoch = 0
no_improve = 0

for epoch in range(1, MAX_EPOCHS + 1):
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1,
        warmup_steps=50,
        optimizer_params={'lr': LR},
        output_path=None,
        show_progress_bar=True,
        use_amp=True,
    )
    r10 = evaluate_r10(model)
    print(f"Epoch {epoch}: R@10={r10:.4f}")
    if r10 > best_r10:
        best_r10 = r10
        best_epoch = epoch
        no_improve = 0
        model.save(str(OUTPUT_DIR))
        print(f"  -> новый лучший чекпоинт")
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"Ранний останов после {epoch} эпох")
            break

print(f"Лучшая эпоха: {best_epoch} (R@10={best_r10:.4f})")

# 5. Строим стеммированный FAISS-индекс
best_model = SentenceTransformer(str(OUTPUT_DIR), device=DEVICE)
okpd_df = pd.read_excel(OKPD_PATH, dtype=str)
okpd_df = okpd_df.dropna(subset=["name"])
okpd_df["stemmed"] = okpd_df["name"].apply(lambda x: cleaner.clean(x, use_stemmer=True))
emb = best_model.encode(okpd_df["stemmed"].tolist(), batch_size=32, convert_to_numpy=True, show_progress_bar=True)
faiss.normalize_L2(emb)
index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb)
faiss.write_index(index, str(FAISS_DIR / "okpd_index_normalized.faiss"))
okpd_df[["code", "parent_code", "name"]].to_csv(FAISS_DIR / "id_map_normalized.csv", index=False)

# 6. Оценка на всей золотой выборке (нормализованные запросы)
print("\nОценка Dense only на всей золотой выборке (нормализованные запросы)...")
hits = {1:0, 5:0, 10:0}
ndcg_sum = 0.0
total = len(gold)
for _, row in tqdm(gold.iterrows(), total=total, desc="Оценка"):
    q = cleaner.clean(row["text_normalized"], use_stemmer=True)
    q_emb = best_model.encode([q], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, 10)
    for i, pos in enumerate(indices[0], 1):
        if okpd_df.iloc[pos]["code"] == row["code"]:
            for k in hits:
                if i <= k:
                    hits[k] += 1
            ndcg_sum += 1.0 / np.log2(i+1)
            break
print(f"R@1: {hits[1]/total:.4f}, R@5: {hits[5]/total:.4f}, R@10: {hits[10]/total:.4f}, NDCG@10: {ndcg_sum/total:.4f}")

# 7. Сравнение с оригиналом (если нужно)
print("\nСравнение с исходными названиями:")
original_model = SentenceTransformer("artifacts/models/bge-m3-frozen-3epoch", device=DEVICE)
# перестроим индекс для оригинальной модели с тем же id_map
original_index = faiss.read_index(str(FAISS_DIR / "okpd_index.faiss"))  # текущий стеммированный индекс
id_map_orig = pd.read_csv(FAISS_DIR / "id_map.csv", dtype=str)
codes_orig = id_map_orig["code"].values
hits_orig = {1:0, 5:0, 10:0}
ndcg_orig = 0.0
for _, row in tqdm(gold.iterrows(), total=total, desc="Оригинал"):
    q = cleaner.clean(row["text_original"], use_stemmer=True)  # исходный запрос
    q_emb = original_model.encode([q], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    scores, indices = original_index.search(q_emb, 10)
    for i, pos in enumerate(indices[0], 1):
        if codes_orig[pos] == row["code"]:
            for k in hits_orig:
                if i <= k:
                    hits_orig[k] += 1
            ndcg_orig += 1.0 / np.log2(i+1)
            break
print(f"Оригинал R@1: {hits_orig[1]/total:.4f}, R@10: {hits_orig[10]/total:.4f}, NDCG@10: {ndcg_orig/total:.4f}")