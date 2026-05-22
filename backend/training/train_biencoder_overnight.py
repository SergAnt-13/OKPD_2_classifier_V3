# train_biencoder_overnight.py (исправленный)
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

# Загружаем данные
gold = pd.read_excel(TRAINING_DATA_DIR / "train.xlsx", dtype=str)
gold = gold[["Номенклатура", "Код ОКПД2"]].dropna()
gold.columns = ["text", "code"]
gold["source"] = "expert"

prom = pd.read_excel(RAW_DATA_DIR / "all_nomenclature.xlsx", dtype=str)
prom = prom[['nomenclature', 'okpd2_code']].dropna()
prom.columns = ["text", "code"]
prom["source"] = "industrial"

all_data = pd.concat([gold, prom], ignore_index=True)
print(f"Экспертных: {len(gold)}, Промышленных: {len(prom)}")

okpd = pd.read_excel(REFERENCE_DIR / "okpd_2.xlsx", dtype=str)
code_to_name = dict(zip(okpd["code"], okpd["name"]))

from backend.preprocessing.cleaner import TextCleaner
from backend.preprocessing.stemmer import get_stemmer

cleaner = TextCleaner(abbreviations_path=REFERENCE_DIR / "сокращения.xlsx")
stemmer = get_stemmer()

pairs = []
for _, row in tqdm(all_data.iterrows(), total=len(all_data), desc="Подготовка пар"):
    text = cleaner.clean(row["text"], stemmer=stemmer)
    target = code_to_name.get(row["code"].strip())
    if text and target:
        pairs.append((text, target))

print(f"Построено пар: {len(pairs)}")

# Загружаем модель на CPU с маленьким батчем
model = SentenceTransformer("BAAI/bge-m3", device="cpu")
train_pairs, eval_pairs = train_test_split(pairs, test_size=0.1, random_state=42)

train_examples = [InputExample(texts=[p[0], p[1]]) for p in train_pairs]

train_dataloader = torch.utils.data.DataLoader(
    train_examples, shuffle=True, batch_size=4  # <-- уменьшили
)
train_loss = MultipleNegativesRankingLoss(model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    warmup_steps=100,
    output_path="artifacts/models/bge-m3-finetuned",
    show_progress_bar=True,
)
print("Дообучение завершено. Модель сохранена в artifacts/models/bge-m3-finetuned.")