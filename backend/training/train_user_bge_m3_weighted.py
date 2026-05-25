# backend/training/train_user_bge_m3_weighted.py
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pandas as pd
import torch
torch.set_default_device("cpu")
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
from config.settings import TRAINING_DATA_DIR, RAW_DATA_DIR, REFERENCE_DIR
from backend.preprocessing.cleaner import TextCleaner

# 1. Золото (вес 1.0) + промка (вес 0.5)
gold = pd.read_excel(TRAINING_DATA_DIR / "train.xlsx", dtype=str)
gold = gold[["Номенклатура", "Код ОКПД2"]].dropna()
gold.columns = ["text", "code"]
gold["weight"] = 1.0

prom = pd.read_excel(RAW_DATA_DIR / "all_nomenclature.xlsx", dtype=str)
prom = prom[['nomenclature', 'okpd2_code']].dropna()
prom.columns = ["text", "code"]
prom["weight"] = 0.5   # <-- усреднённый вес для всех

all_data = pd.concat([gold, prom], ignore_index=True)
print(f"Золото: {len(gold)}, Промка: {len(prom)}")

# 2. Стеммированный справочник
ref = pd.read_csv(REFERENCE_DIR / "okpd_2_normalized_stemmed.csv", dtype=str)
code_to_name = dict(zip(ref["code"], ref["name_normalized"]))

cleaner = TextCleaner(abbreviations_path=REFERENCE_DIR / "сокращения.xlsx")
pairs = []
for _, row in tqdm(all_data.iterrows(), total=len(all_data), desc="Пары"):
    text = cleaner.clean(row["text"], use_stemmer=True)
    target = code_to_name.get(row["code"].strip())
    if text and target:
        pairs.append((text, target))

print(f"Пар для обучения: {len(pairs)}")

# 3. Модель с заморозкой
model = SentenceTransformer("deepvk/USER-bge-m3", device="cpu")
model.to("cpu")

for name, param in model.named_parameters():
    if name.startswith("0.encoder.layer.") or name.startswith("1.encoder.layer."):
        layer_num = int(name.split(".")[2]) if len(name.split(".")) > 2 else -1
        if layer_num < 6:
            param.requires_grad = False
print("Заморожены нижние 6 слоёв.")

train_examples = [InputExample(texts=[p[0], p[1]]) for p in pairs]
train_dataloader = torch.utils.data.DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = MultipleNegativesRankingLoss(model)

print("Дообучение USER-bge-m3 (2 эпохи)...")
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=2,
    warmup_steps=100,
    output_path="artifacts/models/user-bge-m3-extended",
    save_best_model=True,
    show_progress_bar=True,
)
print("Готово: artifacts/models/user-bge-m3-extended")