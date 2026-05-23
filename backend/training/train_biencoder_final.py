# backend/training/train_biencoder_final.py
import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
from config.settings import REFERENCE_DIR, TRAINING_DATA_DIR, RAW_DATA_DIR
from backend.preprocessing.cleaner import TextCleaner

torch.set_default_device("cpu")

# 1. ЗАГРУЗКА ДАННЫХ
gold = pd.read_excel(TRAINING_DATA_DIR / "train.xlsx", dtype=str)
gold = gold[["Номенклатура", "Код ОКПД2"]].dropna()
gold.columns = ["text", "code"]

prom = pd.read_excel(RAW_DATA_DIR / "all_nomenclature.xlsx", dtype=str)
prom = prom[['nomenclature', 'okpd2_code']].dropna()
prom.columns = ["text", "code"]

all_data = pd.concat([gold, prom], ignore_index=True)
print(f"Экспертных: {len(gold)}, Промышленных: {len(prom)}, Всего: {len(all_data)}")

# 2. СПРАВОЧНИК ОКПД-2
okpd = pd.read_excel(REFERENCE_DIR / "okpd_2.xlsx", dtype=str)
code_to_name = dict(zip(okpd["code"], okpd["name"]))

# 3. ПОСТРОЕНИЕ ПАР
cleaner = TextCleaner(abbreviations_path=REFERENCE_DIR / "сокращения.xlsx")
stemmer = get_stemmer()

pairs = []
for _, row in tqdm(all_data.iterrows(), total=len(all_data), desc="Подготовка пар"):
    text = cleaner.clean(row["text"], use_stemmer=True)
    target = code_to_name.get(row["code"].strip())
    if text and target:
        pairs.append((text, target))

print(f"Построено пар: {len(pairs)}")

# 4. МОДЕЛЬ
model = SentenceTransformer("artifacts/models/bge-m3-finetuned", device="cpu")
model.to("cpu")

# 5. TRAIN / EVAL SPLIT
train_pairs, eval_pairs = train_test_split(pairs, test_size=0.1, random_state=42)
train_examples = [InputExample(texts=[p[0], p[1]]) for p in train_pairs]

train_dataloader = torch.utils.data.DataLoader(
    train_examples, shuffle=True, batch_size=4, pin_memory=False
)
train_loss = MultipleNegativesRankingLoss(model)

# 6. ОБУЧЕНИЕ
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    warmup_steps=100,
    output_path="artifacts/models/bge-m3-finetuned-v2",
    show_progress_bar=True,
)

print("Дообучение завершено. Модель сохранена в artifacts/models/bge-m3-finetuned-v2")