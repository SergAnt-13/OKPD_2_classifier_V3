# backend/training/train_berta_classifier.py
"""
Обучение BERTA как классификатора на золотой выборке (аналог train_bert.py из V2).
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
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from datasets import Dataset

from config.settings import TRAINING_DATA_DIR, REFERENCE_DIR
from backend.preprocessing.cleaner import TextCleaner

# ----- 1. Загрузка и подготовка данных -----
gold = pd.read_excel(TRAINING_DATA_DIR / "train.xlsx", dtype=str)
gold = gold[["Номенклатура", "Код ОКПД2"]].dropna()
gold.columns = ["text", "code"]

cleaner = TextCleaner(abbreviations_path=REFERENCE_DIR / "сокращения.xlsx")
gold["text_clean"] = gold["text"].apply(lambda x: cleaner.clean(x, use_stemmer=False))

le = LabelEncoder()
gold["label"] = le.fit_transform(gold["code"])
num_classes = len(le.classes_)
print(f"Число классов: {num_classes}")

# Разбиение (без стратификации из‑за длинного хвоста)
train_df, test_df = train_test_split(gold, test_size=0.2, random_state=42)
print(f"Train: {len(train_df)}, Test: {len(test_df)}")

# ----- 2. Токенизация и модель -----
model_name = "sergeyzh/BERTA"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=num_classes
)

def tokenize_function(examples):
    return tokenizer(
        examples["text_clean"], padding="max_length", truncation=True, max_length=128
    )

train_dataset = Dataset.from_pandas(train_df[["text_clean", "label"]])
test_dataset = Dataset.from_pandas(test_df[["text_clean", "label"]])
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# ----- 3. Параметры обучения -----
training_args = TrainingArguments(
    output_dir="artifacts/models/berta-classifier-checkpoints",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="artifacts/logs/berta-classifier",
    logging_steps=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    learning_rate=2e-5,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_total_limit=1,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# ----- 4. Обучение -----
print("\nОбучение BERTA-классификатора...")
trainer.train()

# ----- 5. Сохранение модели и энкодера меток -----
output_dir = "artifacts/models/berta-classifier"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
import joblib
joblib.dump(le, f"{output_dir}/label_encoder.joblib")
print(f"Модель сохранена в {output_dir}")

# ----- 6. Оценка на тестовой выборке -----
preds = trainer.predict(test_dataset)
y_pred = np.argmax(preds.predictions, axis=1)
y_true = test_df["label"].values

from sklearn.metrics import classification_report, accuracy_score

# Анализ уникальных кодов (то, что ты хотел видеть)
print(f"\n=== АНАЛИЗ ДАТАСЕТА ===")
print(f"Уникальных кодов в train: {train_df['code'].nunique()}")
print(f"Уникальных кодов в test: {test_df['code'].nunique()}")
print(f"Всего уникальных кодов в золотой выборке: {gold['code'].nunique()}")
print(f"\nТоп-10 кодов по частоте:")
print(gold['code'].value_counts().head(10))
print(f"\nТоп-10 самых редких кодов (по 1 примеру):")
rare_codes = gold['code'].value_counts()
print(rare_codes[rare_codes == 1].head(10))

# Accuracy
acc = accuracy_score(y_true, y_pred)
print(f"\nAccuracy (BERTA-классификатор): {acc:.4f}")

# Исправленный classification report (без ошибки)
present_labels = sorted(set(y_true) | set(y_pred))
present_names = le.inverse_transform(present_labels)
print(f"\nОтчёт по классификации (все {len(present_labels)} встреченных классов):")
print(classification_report(y_true, y_pred, labels=present_labels, target_names=present_names, zero_division=0))