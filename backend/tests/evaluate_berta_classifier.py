# backend/tests/evaluate_berta_classifier.py
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from config.settings import TRAINING_DATA_DIR

MODEL_PATH = "artifacts/models/berta_classifier_improved"
GOLD_PATH = TRAINING_DATA_DIR / "train.xlsx"

# Загружаем модель и лейблы
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
with open(Path(MODEL_PATH) / "label_encoder.json", "r", encoding="utf-8") as f:
    class_names = json.load(f)
label2id = {name: i for i, name in enumerate(class_names)}

if torch.cuda.is_available():
    model.to("cuda")
model.eval()

# Загружаем золотую выборку
gold = pd.read_excel(GOLD_PATH, dtype=str)[["Номенклатура", "Код ОКПД2"]].dropna()
gold.columns = ["text", "code"]

# Оставляем только классы, известные модели
known_codes = set(class_names)
gold_filtered = gold[gold["code"].isin(known_codes)]
print(f"Примеров с известными классами: {len(gold_filtered)} из {len(gold)}")

y_true = []
y_pred = []

for _, row in tqdm(gold_filtered.iterrows(), total=len(gold_filtered), desc="BERTA Eval"):
    text = row["text"]
    true_code = row["code"]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        pred_id = logits.argmax(dim=1).item()
    pred_code = class_names[pred_id] if pred_id < len(class_names) else None
    if pred_code is None:
        continue
    y_true.append(true_code)
    y_pred.append(pred_code)

# Метрики
acc = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

print("\nBERTA-классификатор на золотой выборке (известные классы):")
print(f"  Accuracy:   {acc:.4f}")
print(f"  Macro F1:   {macro_f1:.4f}")
print(f"  Weighted F1: {weighted_f1:.4f}")
print(f"  Число примеров: {len(y_true)}")
print(f"  Число уникальных предсказанных классов: {len(set(y_pred))}")