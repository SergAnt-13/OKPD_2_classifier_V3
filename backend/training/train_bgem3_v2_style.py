# backend/training/train_bgem3_v2_style.py
"""
Дообучение BGE-M3 в стиле V2:
- только пищевые классы (01, 02, 03, 10)
- стемминг
- MultipleNegativesRankingLoss
- 2 эпохи с промежуточной оценкой
- автоматическое построение стеммированного FAISS-индекса
- финальный замер Recall@10 на hold‑out (300 примеров)
"""
import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent   # backend/training -> корень
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
from sentence_transformers.sentence_transformer.evaluation import EmbeddingSimilarityEvaluator

from config.settings import TRAINING_DATA_DIR, RAW_DATA_DIR, REFERENCE_DIR, FAISS_DIR
from backend.preprocessing.cleaner import TextCleaner

# ============================================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================================
gold = pd.read_excel(TRAINING_DATA_DIR / "train.xlsx", dtype=str)
gold = gold[["Номенклатура", "Код ОКПД2"]].dropna()
gold.columns = ["text", "code"]

prom = pd.read_excel(RAW_DATA_DIR / "all_nomenclature.xlsx", dtype=str)
prom = prom[['nomenclature', 'okpd2_code']].dropna()
prom.columns = ["text", "code"]

all_data = pd.concat([gold, prom], ignore_index=True)
print(f"Всего записей: {len(all_data)} (экспертных: {len(gold)}, промышленных: {len(prom)})")

# Фильтруем только пищевые классы
food_prefixes = ('01', '02', '03', '10')
all_data = all_data[all_data['code'].str.startswith(food_prefixes, na=False)]
print(f"Пищевых товаров: {len(all_data)}")

# ============================================================
# 2. СПРАВОЧНИК (стеммированный)
# ============================================================
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

# ============================================================
# 3. ПОСТРОЕНИЕ ОБУЧАЮЩИХ ПАР
# ============================================================
cleaner = TextCleaner(abbreviations_path=REFERENCE_DIR / "сокращения.xlsx")
pairs = []
for _, row in tqdm(all_data.iterrows(), total=len(all_data), desc="Пары"):
    text = cleaner.clean(row['text'], use_stemmer=True)
    target = code_to_name.get(row['code'].strip())
    if text and target:
        pairs.append((text, target))
print(f"Всего пар: {len(pairs)}")

# Разделяем на train/eval
train_pairs, eval_pairs = train_test_split(pairs, test_size=0.2, random_state=42)
train_examples = [InputExample(texts=[p[0], p[1]]) for p in train_pairs]
eval_s1 = [p[0] for p in eval_pairs]
eval_s2 = [p[1] for p in eval_pairs]
eval_scores = [1.0] * len(eval_pairs)

# ============================================================
# 4. ЗАГРУЗКА МОДЕЛИ
# ============================================================
model = SentenceTransformer("BAAI/bge-m3", device="cpu")
model.to("cpu")

train_dataloader = torch.utils.data.DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = MultipleNegativesRankingLoss(model)

evaluator_epoch1 = EmbeddingSimilarityEvaluator(
    sentences1=eval_s1,
    sentences2=eval_s2,
    scores=eval_scores,
    name="eval_epoch1",
    show_progress_bar=True,
)

print("\n=== ЭПОХА 1 ===")
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator_epoch1,
    epochs=1,
    warmup_steps=100,
    output_path="artifacts/models/bge-m3-food-only-epoch1",
    save_best_model=True,
    show_progress_bar=True,
)

# ============================================================
# 5. ВТОРАЯ ЭПОХА (дообучение на основе первой)
# ============================================================
model_epoch2 = SentenceTransformer("artifacts/models/bge-m3-food-only-epoch1", device="cpu")
model_epoch2.to("cpu")

evaluator_epoch2 = EmbeddingSimilarityEvaluator(
    sentences1=eval_s1,
    sentences2=eval_s2,
    scores=eval_scores,
    name="eval_epoch2",
    show_progress_bar=True,
)

print("\n=== ЭПОХА 2 ===")
model_epoch2.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator_epoch2,
    epochs=1,
    warmup_steps=50,
    output_path="artifacts/models/bge-m3-food-only-epoch2",
    save_best_model=True,
    show_progress_bar=True,
)

# ============================================================
# 6. ПОСТРОЕНИЕ СТЕММИРОВАННОГО FAISS-ИНДЕКСА (для лучшей модели)
# ============================================================
print("\nСтроим стеммированный FAISS-индекс...")
best_model_path = "artifacts/models/bge-m3-food-only-epoch2"
best_model = SentenceTransformer(best_model_path, device="cpu")

okpd = pd.read_excel(REFERENCE_DIR / "okpd_2.xlsx", dtype=str)
okpd = okpd.dropna(subset=["name"])
okpd["name"] = okpd["name"].astype(str).str.strip()
okpd["name_stemmed"] = okpd["name"].apply(lambda x: cleaner.clean(x, use_stemmer=True))

emb = best_model.encode(okpd["name_stemmed"].tolist(), batch_size=8,
                        convert_to_numpy=True, show_progress_bar=True)
faiss.normalize_L2(emb)
idx = faiss.IndexFlatIP(emb.shape[1])
idx.add(emb)
faiss.write_index(idx, str(FAISS_DIR / "okpd_index_food_only.faiss"))
okpd[["code", "parent_code", "name"]].to_csv(FAISS_DIR / "id_map_food_only.csv", index=False)

# ============================================================
# 7. ОЦЕНКА НА HOLD‑OUT (300 примеров)
# ============================================================
print("\nОценка на hold‑out...")
gold = pd.read_excel(TRAINING_DATA_DIR / "train.xlsx", dtype=str)
gold = gold[["Номенклатура", "Код ОКПД2"]].dropna()
gold.columns = ["text", "true_code"]
_, test_df = train_test_split(gold, test_size=300, random_state=42)

idx = faiss.read_index(str(FAISS_DIR / "okpd_index_food_only.faiss"))
id_map = pd.read_csv(FAISS_DIR / "id_map_food_only.csv", dtype=str)
codes = id_map["code"].values

hits = {1:0, 5:0, 10:0}
for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Тест"):
    q = cleaner.clean(row["text"], use_stemmer=True)
    emb = best_model.encode([q], convert_to_numpy=True, show_progress_bar=False)
    faiss.normalize_L2(emb)
    scores, indices = idx.search(emb, 10)
    for i, pos in enumerate(indices[0], 1):
        if 0 <= pos < len(codes) and codes[pos] == row["true_code"]:
            for k in hits:
                if i <= k:
                    hits[k] += 1
            break

total = len(test_df)
print(f"\nRecall@1:  {hits[1]/total:.4f}")
print(f"Recall@5:  {hits[5]/total:.4f}")
print(f"Recall@10: {hits[10]/total:.4f}")
print(f"\nМодель сохранена в {best_model_path}")