# backend/training/train_berta_weighted.py
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
prom["weight"] = 0.5

all_data = pd.concat([gold, prom], ignore_index=True)
food_prefixes = ('01', '02', '03', '10')
all_data = all_data[all_data['code'].str.startswith(food_prefixes, na=False)]
print(f"Пищевых товаров: {len(all_data)}")

# 2. Предобработка
cleaner = TextCleaner(abbreviations_path=REFERENCE_DIR / "сокращения.xlsx")
all_data["text_clean"] = all_data["text"].apply(lambda x: cleaner.clean(x, use_stemmer=False))

le = LabelEncoder()
all_data["label"] = le.fit_transform(all_data["code"])
num_classes = len(le.classes_)
print(f"Число классов: {num_classes}")

# 3. Разделение
train_df, test_df = train_test_split(all_data, test_size=0.2, random_state=42)
print(f"Train: {len(train_df)}, Test: {len(test_df)}")

# 4. Токенизация и модель
model_name = "sergeyzh/BERTA"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)

def tokenize(examples):
    return tokenizer(examples["text_clean"], padding="max_length", truncation=True, max_length=128)

train_dataset = Dataset.from_pandas(train_df[["text_clean", "label", "weight"]])
test_dataset = Dataset.from_pandas(test_df[["text_clean", "label"]])
train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label", "weight"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        weights = inputs.pop("weight", None)
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(logits, labels)
        if weights is not None:
            loss = loss * weights
        loss = loss.mean()
        return (loss, outputs) if return_outputs else loss

training_args = TrainingArguments(
    output_dir="artifacts/models/berta-extended-checkpoints",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="artifacts/logs/berta-extended",
    logging_steps=50,
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

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

print("\nДообучение BERTA...")
trainer.train()

output_dir = "artifacts/models/berta-extended"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
import joblib
joblib.dump(le, f"{output_dir}/label_encoder.joblib")
print(f"Модель сохранена в {output_dir}")

# Оценка
preds = trainer.predict(test_dataset)
y_pred = np.argmax(preds.predictions, axis=1)
y_true = test_df["label"].values
from sklearn.metrics import accuracy_score
print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")