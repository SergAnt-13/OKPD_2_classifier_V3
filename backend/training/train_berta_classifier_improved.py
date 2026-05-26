# backend/training/train_berta_classifier_improved.py
# Дообучение BERTA-классификатора: золото + промка с весом 0.3, заморозка 6 слоёв, отбор лучшей.
import sys, os, json, random
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from config.settings import TRAINING_DATA_DIR, RAW_DATA_DIR, REFERENCE_DIR
from backend.preprocessing.cleaner import TextCleaner

# ---------- Настройки ----------
BASE_MODEL = "sergeyzh/BERTA"
OUTPUT_DIR = Path("artifacts/models/berta_classifier_improved")
EPOCHS = 13
BATCH_SIZE = 16
FREEZE_LAYERS = 6
MAX_LENGTH = 256
LR = 2e-5
PROM_WEIGHT = 0.3
USE_PROM = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GOLD_PATH = TRAINING_DATA_DIR / "train.xlsx"
PROM_PATH = RAW_DATA_DIR / "all_nomenclature.xlsx"
ABBR_PATH = REFERENCE_DIR / "сокращения.xlsx"
# -----------------------------

# 1. Загружаем золотую выборку
gold = pd.read_excel(GOLD_PATH, dtype=str)
gold = gold[["Номенклатура", "Код ОКПД2"]].dropna()
gold.columns = ["text", "code"]

# Оставляем классы, встречающиеся более 1 раза
class_counts = gold["code"].value_counts()
valid_classes = class_counts[class_counts > 1].index
gold = gold[gold["code"].isin(valid_classes)]

cleaner = TextCleaner(abbreviations_path=ABBR_PATH)

# 2. Загружаем и фильтруем промку (только те классы, что есть в золоте)
if USE_PROM:
    prom = pd.read_excel(PROM_PATH, dtype=str)[['nomenclature', 'okpd2_code']].dropna()
    prom.columns = ["text", "code"]
    prom = prom[prom["code"].isin(valid_classes)]  # фильтрация
    print(f"Промка после фильтрации: {len(prom)} примеров (только классы из золота)")

# 3. Кодируем метки (LabelEncoder обучаем только на золоте, чтобы не было новых классов)
le = LabelEncoder()
le.fit(gold["code"])
gold["label"] = le.transform(gold["code"])
num_classes = len(le.classes_)
print(f"Число классов: {num_classes}")

# 4. Разделяем золото на train/val (стратифицированно)
train_gold, val_gold = train_test_split(gold, test_size=0.2, random_state=42, stratify=gold["label"])
print(f"Train (золото): {len(train_gold)}, Val (золото): {len(val_gold)}")

# 5. Добавляем промку в train с нужным весом
if USE_PROM:
    prom["label"] = le.transform(prom["code"])
    num_prom = int(len(train_gold) * PROM_WEIGHT / (1 - PROM_WEIGHT))  # сколько нужно промки для веса 0.3
    if num_prom > len(prom):
        prom_sample = prom.sample(n=num_prom, replace=True, random_state=42)
    else:
        prom_sample = prom.sample(n=num_prom, random_state=42)
    train_all = pd.concat([train_gold, prom_sample], ignore_index=True).sample(frac=1, random_state=42)
    print(f"Train с промкой: {len(train_all)} (промки: {len(prom_sample)})")
else:
    train_all = train_gold

# 6. Датасет
class TextDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.texts = [cleaner.clean(t, use_stemmer=False) for t in df["text"]]
        self.labels = df["label"].values
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=num_classes)
model.to(DEVICE)

# Заморозка нижних слоёв BERT
for i, layer in enumerate(model.bert.encoder.layer):
    if i < FREEZE_LAYERS:
        for param in layer.parameters():
            param.requires_grad = False
for param in model.bert.embeddings.parameters():
    param.requires_grad = False

train_dataset = TextDataset(train_all, tokenizer, MAX_LENGTH)
val_dataset = TextDataset(val_gold, tokenizer, MAX_LENGTH)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

optimizer = optim.AdamW(model.parameters(), lr=LR)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

best_val_acc = 0.0
best_epoch = None
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Train epoch {epoch}/{EPOCHS}"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        train_loss += loss.item()
    avg_train_loss = train_loss / len(train_loader)

    model.eval()
    val_acc = 0
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            val_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            val_acc += (preds == labels).sum().item()
    avg_val_loss = val_loss / len(val_loader)
    val_acc /= len(val_dataset)
    print(f"Epoch {epoch}: train loss {avg_train_loss:.4f}, val loss {avg_val_loss:.4f}, val acc {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        with open(OUTPUT_DIR / "label_encoder.json", "w", encoding="utf-8") as f:
            json.dump(list(le.classes_), f, ensure_ascii=False)
        print(f"  -> новый лучший чекпоинт (val_acc={best_val_acc:.4f})")

print(f"\nОбучение завершено. Лучшая эпоха: {best_epoch} (val_acc={best_val_acc:.4f})")
print(f"Модель и лейбл-энкодер сохранены в {OUTPUT_DIR}")