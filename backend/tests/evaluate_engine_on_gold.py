import sys, os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from tqdm import tqdm
from config.settings import TRAINING_DATA_DIR, REFERENCE_DIR, FAISS_DIR
from backend.models.retriever import Retriever
from backend.models.engine import DecisionEngine

MODEL = "artifacts/models/bge-m3-frozen-3epoch"
CLASSIFIER_PATH = Path("artifacts/models/berta_classifier_improved")

gold = pd.read_excel(TRAINING_DATA_DIR / "train.xlsx", dtype=str)
gold = gold[["Номенклатура", "Код ОКПД2"]].dropna()
gold.columns = ["text", "true_code"]

retriever = Retriever(
    model_name=MODEL,
    index_path=FAISS_DIR / "okpd_index.faiss",
    id_map_path=FAISS_DIR / "id_map.csv",
)
engine = DecisionEngine(retriever, classifier_path=CLASSIFIER_PATH)

results = []
for _, row in tqdm(gold.iterrows(), total=len(gold)):
    res = engine.predict(row["text"])
    results.append({
        "true_code": row["true_code"],
        "predicted_code": res["predicted_code"],
        "mode": res["mode"],
        "confidence": res["confidence"],
        "correct": res["predicted_code"] == row["true_code"] if res["predicted_code"] else False,
        "classifier_code": res.get("classifier_code"),
        "classifier_prob": res.get("classifier_prob"),
    })

df = pd.DataFrame(results)
total = len(df)
print(f"\nВсего примеров: {total}")
print(f"AUTO:     {(df['mode']=='AUTO').sum():4d} ({(df['mode']=='AUTO').mean()*100:.1f}%)")
print(f"REVIEW:   {(df['mode']=='REVIEW').sum():4d} ({(df['mode']=='REVIEW').mean()*100:.1f}%)")
print(f"MANUAL:   {(df['mode']=='MANUAL').sum():4d} ({(df['mode']=='MANUAL').mean()*100:.1f}%)")

# Точность среди AUTO
auto_correct = df[df['mode']=='AUTO']['correct']
if len(auto_correct) > 0:
    print(f"\nТочность в AUTO: {auto_correct.mean()*100:.1f}%")
else:
    print("Нет AUTO предсказаний")