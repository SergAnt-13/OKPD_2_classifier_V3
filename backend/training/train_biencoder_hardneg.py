# train_biencoder_hardneg.py
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
from sentence_transformers import SentenceTransformer, InputExample, losses
from config.settings import REFERENCE_DIR, TRAINING_DATA_DIR

# 1. Загружаем hard negatives
hard_negatives = pd.read_pickle(TRAINING_DATA_DIR / "hard_negatives.pkl")
print(f"Загружено {len(hard_negatives)} примеров с hard negatives")

# 2. Загружаем справочник ОКПД-2
okpd = pd.read_excel(REFERENCE_DIR / "okpd_2.xlsx", dtype=str)
code_to_name = dict(zip(okpd["code"], okpd["name"]))

# 3. Формируем тройки (anchor, positive, negative)
triplets = []
for item in hard_negatives:
    anchor_text = item["text"]
    positive_code = item["positive_code"]
    positive_name = code_to_name.get(positive_code)
    if not positive_name:
        continue
    for neg_code in item["negatives"]:
        neg_name = code_to_name.get(neg_code)
        if neg_name:
            triplets.append((anchor_text, positive_name, neg_name))

print(f"Сформировано {len(triplets)} троек")

# 4. Превращаем в InputExample
train_examples = [InputExample(texts=[t[0], t[1], t[2]]) for t in triplets]

# 5. Загружаем модель (дообучаем finetuned v2)
model = SentenceTransformer("artifacts/models/bge-m3-finetuned-v2", device="cpu")
model.to("cpu")

train_dataloader = torch.utils.data.DataLoader(
    train_examples, shuffle=True, batch_size=4, pin_memory=False
)
train_loss = losses.TripletLoss(model)

# 6. Дообучение
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    warmup_steps=50,
    output_path="artifacts/models/bge-m3-hardneg",
    show_progress_bar=True,
)

print("Дообучение завершено. Модель сохранена в artifacts/models/bge-m3-hardneg")