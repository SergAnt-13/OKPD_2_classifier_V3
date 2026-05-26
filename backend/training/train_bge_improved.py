# backend/training/train_bge_improved.py
"""
Дообучение BGE‑M3: эксперты + 30% промки, заморозка 6 слоёв,
evaluator для сохранения лучшей модели, финальные метрики.
"""
import sys, os, random
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

from config.settings import TRAINING_DATA_DIR, RAW_DATA_DIR, REFERENCE_DIR, FAISS_DIR
from backend.preprocessing.cleaner import TextCleaner

# ---------- Настройки ----------
BGE_BASE = "BAAI/bge-m3"
OUTPUT_DIR = Path("artifacts/models/bge-m3-final-mix")
FREEZE_LAYERS = 6
EPOCHS = 10
BATCH_SIZE = 16
LR = 2e-5
PROM_WEIGHT = 0.3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GOLD_PATH = TRAINING_DATA_DIR / "train.xlsx"
PROM_PATH = RAW_DATA_DIR / "all_nomenclature.xlsx"
OKPD_PATH = REFERENCE_DIR / "okpd_2.xlsx"
ABBR_PATH = REFERENCE_DIR / "сокращения.xlsx"
# --------------------------------

cleaner = TextCleaner(abbreviations_path=ABBR_PATH)

# 1. Загружаем экспертные пары и промку (стеммированные)
gold = pd.read_excel(GOLD_PATH, dtype=str)[["Номенклатура", "Код ОКПД2"]].dropna()
gold.columns = ["text", "code"]
prom = pd.read_excel(PROM_PATH, dtype=str)[['nomenclature', 'okpd2_code']].dropna()
prom.columns = ["text", "code"]

okpd = pd.read_excel(OKPD_PATH, dtype=str)
code_to_name = {}
for _, r in okpd.iterrows():
    code_to_name[r["code"]] = cleaner.clean(r["name"], use_stemmer=True)

def make_pairs(df, desc="пары"):
    pairs = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        q = cleaner.clean(row["text"], use_stemmer=True)
        target = code_to_name.get(row["code"].strip())
        if q and target:
            pairs.append((q, target))
    return pairs

expert_pairs = make_pairs(gold, "Экспертные пары")
prom_pairs = make_pairs(prom, "Промка")
print(f"Экспертных пар: {len(expert_pairs)}, промки: {len(prom_pairs)}")

# 2. Разделяем экспертов на train/eval (80/20)
train_expert, eval_expert = train_test_split(expert_pairs, test_size=0.2, random_state=42)
eval_s1 = [p[0] for p in eval_expert]
eval_s2 = [p[1] for p in eval_expert]
print(f"Train (эксперты): {len(train_expert)}, Eval: {len(eval_expert)}")

# 3. Модель с заморозкой 6 слоёв
model = SentenceTransformer(BGE_BASE, device=DEVICE)
encoder_layers = model._first_module().auto_model.encoder.layer
for i, layer in enumerate(encoder_layers):
    if i < FREEZE_LAYERS:
        for param in layer.parameters():
            param.requires_grad = False
for param in model._first_module().auto_model.embeddings.parameters():
    param.requires_grad = False
print(f"Заморожены первые {FREEZE_LAYERS} слоёв и эмбеддинги")

# 4. Добавляем промку с весом 0.3
random.seed(42)
num_expert = len(train_expert)
num_prom_to_add = int(num_expert * PROM_WEIGHT / (1 - PROM_WEIGHT))
if num_prom_to_add > len(prom_pairs):
    prom_sample = random.choices(prom_pairs, k=num_prom_to_add)
else:
    prom_sample = random.sample(prom_pairs, num_prom_to_add)

all_train_pairs = train_expert + prom_sample
random.shuffle(all_train_pairs)
print(f"Всего обучающих пар (с учётом промки): {len(all_train_pairs)} (промки: {len(prom_sample)})")

train_examples = [InputExample(texts=[p[0], p[1]]) for p in all_train_pairs]
train_dataloader = torch.utils.data.DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
train_loss = MultipleNegativesRankingLoss(model)

# 5. Evaluator для сохранения лучшей модели
evaluator = EmbeddingSimilarityEvaluator(
    sentences1=eval_s1,
    sentences2=eval_s2,
    scores=[1.0] * len(eval_s1),
    name="eval",
    show_progress_bar=False,
)

# 6. Обучение (лучшая модель сохранится автоматически)
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator,
    epochs=EPOCHS,
    warmup_steps=50,
    optimizer_params={'lr': LR},
    output_path=str(OUTPUT_DIR),
    save_best_model=True,
    show_progress_bar=True,
    use_amp=True,
)
print(f"Лучшая модель сохранена в {OUTPUT_DIR}")

# 7. Построение FAISS-индекса для лучшей модели
best_model = SentenceTransformer(str(OUTPUT_DIR), device=DEVICE)
okpd_df = pd.read_excel(OKPD_PATH, dtype=str)
okpd_df = okpd_df.dropna(subset=["name"])
okpd_df["stemmed"] = okpd_df["name"].apply(lambda x: cleaner.clean(x, use_stemmer=True))
emb = best_model.encode(okpd_df["stemmed"].tolist(), batch_size=32, convert_to_numpy=True, show_progress_bar=True)
faiss.normalize_L2(emb)
index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb)
faiss.write_index(index, str(FAISS_DIR / "okpd_index_final_mix.faiss"))
okpd_df[["code", "parent_code", "name"]].to_csv(FAISS_DIR / "id_map_final_mix.csv", index=False)

# 8. Финальная оценка на всей золотой выборке
gold_all = pd.read_excel(GOLD_PATH, dtype=str)[["Номенклатура", "Код ОКПД2"]].dropna()
gold_all.columns = ["text", "code"]
hits = {1:0, 5:0, 10:0}
ndcg_sum = 0.0
for _, row in tqdm(gold_all.iterrows(), total=len(gold_all), desc="Финальная оценка"):
    q = cleaner.clean(row["text"], use_stemmer=True)
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
total = len(gold_all)
print(f"\nФинальные метрики на всей золотой выборке:")
print(f"R@1: {hits[1]/total:.4f}")
print(f"R@5: {hits[5]/total:.4f}")
print(f"R@10: {hits[10]/total:.4f}")
print(f"NDCG@10: {ndcg_sum/total:.4f}")