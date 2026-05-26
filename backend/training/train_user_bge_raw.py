# backend/training/train_user_bge_raw.py
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

from config.settings import TRAINING_DATA_DIR, REFERENCE_DIR, FAISS_DIR
from backend.preprocessing.cleaner import TextCleaner

# ---------- Настройки ----------
BGE_BASE = "deepvk/USER-bge-m3"
OUTPUT_DIR = Path("artifacts/models/user-bge-m3-raw")
FREEZE_LAYERS = 3
MAX_EPOCHS = 15
PATIENCE = 3
BATCH_SIZE = 16
LR = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GOLD_PATH = TRAINING_DATA_DIR / "train.xlsx"   # исходные названия, без нормализации
OKPD_PATH = REFERENCE_DIR / "okpd_2.xlsx"
ABBR_PATH = REFERENCE_DIR / "сокращения.xlsx"
# --------------------------------

cleaner = TextCleaner(abbreviations_path=ABBR_PATH)

# 1. Загружаем золотую выборку
gold = pd.read_excel(GOLD_PATH, dtype=str)[["Номенклатура", "Код ОКПД2"]].dropna()
gold.columns = ["text", "code"]

# 2. Строим стеммированные пары (запрос → описание кода)
okpd = pd.read_excel(OKPD_PATH, dtype=str)
code_to_name = {}
for _, r in okpd.iterrows():
    code_to_name[r["code"]] = cleaner.clean(r["name"], use_stemmer=True)

pairs = []
for _, row in tqdm(gold.iterrows(), total=len(gold), desc="Пары"):
    q = cleaner.clean(row["text"], use_stemmer=True)      # стемминг запроса
    target = code_to_name.get(row["code"].strip())
    if q and target:
        pairs.append((q, target))

# 3. Разделение train/eval
train_pairs, eval_pairs = train_test_split(pairs, test_size=0.2, random_state=42)
eval_s1 = [p[0] for p in eval_pairs]
eval_s2 = [p[1] for p in eval_pairs]
print(f"Train: {len(train_pairs)}, Eval: {len(eval_pairs)}")

# 4. Модель и заморозка
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

# 5. Функция для расчёта R@10 на eval
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

# 6. Обучение с early stopping
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

# 7. Строим стеммированный FAISS-индекс для USER
best_model = SentenceTransformer(str(OUTPUT_DIR), device=DEVICE)
okpd_df = pd.read_excel(OKPD_PATH, dtype=str)
okpd_df = okpd_df.dropna(subset=["name"])
okpd_df["stemmed"] = okpd_df["name"].apply(lambda x: cleaner.clean(x, use_stemmer=True))
emb = best_model.encode(okpd_df["stemmed"].tolist(), batch_size=32, convert_to_numpy=True, show_progress_bar=True)
faiss.normalize_L2(emb)
index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb)
faiss.write_index(index, str(FAISS_DIR / "okpd_index_user_raw.faiss"))
okpd_df[["code", "parent_code", "name"]].to_csv(FAISS_DIR / "id_map_user_raw.csv", index=False)

# 8. Оценка USER на всей золотой выборке (исходные запросы, стемминг)
print("\nОценка USER на всей золотой выборке...")
hits_user = {1:0, 5:0, 10:0}
ndcg_user = 0.0
total = len(gold)
for _, row in tqdm(gold.iterrows(), total=total, desc="USER Dense only"):
    q = cleaner.clean(row["text"], use_stemmer=True)
    q_emb = best_model.encode([q], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, 10)
    for i, pos in enumerate(indices[0], 1):
        if okpd_df.iloc[pos]["code"] == row["code"]:
            for k in hits_user:
                if i <= k:
                    hits_user[k] += 1
            ndcg_user += 1.0 / np.log2(i+1)
            break

print(f"USER: R@1: {hits_user[1]/total:.4f}, R@5: {hits_user[5]/total:.4f}, R@10: {hits_user[10]/total:.4f}, NDCG@10: {ndcg_user/total:.4f}")

# 9. Сравнение с чемпионом (загружаем его и его индекс)
print("\nСравнение с чемпионом (frozen-3epoch)...")
champion_model = SentenceTransformer("artifacts/models/bge-m3-frozen-3epoch", device=DEVICE)
champion_index = faiss.read_index(str(FAISS_DIR / "okpd_index.faiss"))
id_map_champion = pd.read_csv(FAISS_DIR / "id_map.csv", dtype=str)
codes_champion = id_map_champion["code"].values

hits_champ = {1:0, 5:0, 10:0}
ndcg_champ = 0.0
for _, row in tqdm(gold.iterrows(), total=total, desc="Champion Dense only"):
    q = cleaner.clean(row["text"], use_stemmer=True)
    q_emb = champion_model.encode([q], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    scores, indices = champion_index.search(q_emb, 10)
    for i, pos in enumerate(indices[0], 1):
        if codes_champion[pos] == row["code"]:
            for k in hits_champ:
                if i <= k:
                    hits_champ[k] += 1
            ndcg_champ += 1.0 / np.log2(i+1)
            break

print(f"Champion: R@1: {hits_champ[1]/total:.4f}, R@5: {hits_champ[5]/total:.4f}, R@10: {hits_champ[10]/total:.4f}, NDCG@10: {ndcg_champ/total:.4f}")

# 10. Итоговая таблица
print("\n" + "="*50)
print("СРАВНЕНИЕ МОДЕЛЕЙ (исходные запросы, стемминг)")
print("="*50)
print(f"{'Модель':<35} {'R@1':<8} {'R@5':<8} {'R@10':<8} {'NDCG':<8}")
print(f"{'USER-bge-m3 (raw, 3 слоя)':<35} {hits_user[1]/total:<8.4f} {hits_user[5]/total:<8.4f} {hits_user[10]/total:<8.4f} {ndcg_user/total:<8.4f}")
print(f"{'bge-m3-frozen-3epoch (чемпион)':<35} {hits_champ[1]/total:<8.4f} {hits_champ[5]/total:<8.4f} {hits_champ[10]/total:<8.4f} {ndcg_champ/total:<8.4f}")