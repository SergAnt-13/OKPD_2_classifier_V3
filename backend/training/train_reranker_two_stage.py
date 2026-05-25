# backend/training/train_reranker_two_stage.py
"""
Двухэтапное дообучение реранкера (BAAI/bge-reranker-v2-m3)
с использованием всей доступной выборки и Hard Negative Mining.
"""
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
torch.set_default_device("cpu")
torch.backends.mps.is_available = lambda: False   # ← отключаем MPS полностью

import pandas as pd
from tqdm import tqdm
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder import CrossEncoderTrainer, CrossEncoderTrainingArguments
from datasets import Dataset
from config.settings import TRAINING_DATA_DIR, RAW_DATA_DIR, REFERENCE_DIR
from backend.preprocessing.cleaner import TextCleaner
from backend.models.retriever import Retriever

# ============================================================
# 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# ============================================================
# Золотая выборка (вес 1.0)
gold = pd.read_excel(TRAINING_DATA_DIR / "train.xlsx", dtype=str)
gold = gold[["Номенклатура", "Код ОКПД2"]].dropna()
gold.columns = ["text", "code"]

# Вся промка
prom = pd.read_excel(RAW_DATA_DIR / "all_nomenclature.xlsx", dtype=str)
prom = prom[['nomenclature', 'okpd2_code']].dropna()
prom.columns = ["text", "code"]

# Справочник для получения эталонных названий
ref = pd.read_csv(REFERENCE_DIR / "okpd_2_normalized_stemmed.csv", dtype=str)
code_to_name = dict(zip(ref["code"], ref["name"]))

cleaner = TextCleaner(abbreviations_path=REFERENCE_DIR / "сокращения.xlsx")

# Инициализируем retriever для поиска сложных негативов
retriever = Retriever(model_name="artifacts/models/bge-m3-frozen-stratified-epoch2")

# ============================================================
# 2. ФОРМИРОВАНИЕ ОБУЧАЮЩИХ ПАР
# ============================================================
train_data = []

# --- Позитивные пары из золотой выборки ---
print("Формирование позитивных пар из золотой выборки...")
for _, row in tqdm(gold.iterrows(), total=len(gold), desc="Золото"):
    text = cleaner.clean(row["text"], use_stemmer=False)
    target_name = code_to_name.get(row["code"].strip())
    if not target_name:
        continue

    # Позитивная пара
    train_data.append({
        "query": text,
        "document": target_name,
        "label": 1,
    })

    # Hard Negative Mining
    try:
        candidates = retriever.search(text, top_k=10)["candidates"]
        for cand in candidates[:3]:
            if cand["code"] != row["code"]:
                train_data.append({
                    "query": text,
                    "document": cand["name"],
                    "label": 0,
                })
    except Exception:
        pass

# --- Пары из промки ---
print("Формирование пар из промки...")
for _, row in tqdm(prom.iterrows(), total=len(prom), desc="Промка"):
    text = cleaner.clean(row["text"], use_stemmer=False)
    target_name = code_to_name.get(row["code"].strip())
    if not target_name:
        continue
    train_data.append({
        "query": text,
        "document": target_name,
        "label": 1,
    })

print(f"Всего обучающих пар: {len(train_data)}")
train_df = pd.DataFrame(train_data)

# Оставляем только нужные колонки для датасета
train_df = train_df[["query", "document", "label"]]

# ============================================================
# 3. ПОДГОТОВКА ДАТАСЕТА ДЛЯ ОБУЧЕНИЯ
# ============================================================
dataset = Dataset.from_pandas(train_df)

# ============================================================
# 4. ДООБУЧЕНИЕ РЕРАНКЕРА (ДВА ЭТАПА)
# ============================================================
model_name = "BAAI/bge-reranker-v2-m3"
model = CrossEncoder(model_name, max_length=512)

# --- Этап 1: Базовое дообучение ---
print("\n=== ЭТАП 1: Базовое дообучение ===")
args_stage1 = CrossEncoderTrainingArguments(
    output_dir="artifacts/models/reranker-st1",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    warmup_steps=50,
    logging_steps=50,
    save_strategy="epoch",
    report_to="none",
)

trainer_stage1 = CrossEncoderTrainer(
    model=model,
    args=args_stage1,
    train_dataset=dataset,
)

trainer_stage1.train()
model.save_pretrained("artifacts/models/reranker-st1")

# --- Этап 2: Тонкая настройка ---
print("\n=== ЭТАП 2: Тонкая настройка ===")
model = CrossEncoder("artifacts/models/reranker-st1", max_length=512)
args_stage2 = CrossEncoderTrainingArguments(
    output_dir="artifacts/models/reranker-final",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=1e-5,
    warmup_steps=50,
    logging_steps=50,
    save_strategy="epoch",
    report_to="none",
)

trainer_stage2 = CrossEncoderTrainer(
    model=model,
    args=args_stage2,
    train_dataset=dataset,
)

trainer_stage2.train()
model.save_pretrained("artifacts/models/reranker-final")

print("\nГотово: artifacts/models/reranker-final")