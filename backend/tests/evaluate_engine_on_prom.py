# backend/tests/evaluate_engine_on_prom.py
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from tqdm import tqdm
from config.settings import RAW_DATA_DIR, REFERENCE_DIR, FAISS_DIR
from backend.models.retriever import Retriever
from backend.models.engine import DecisionEngine

MODEL = "artifacts/models/bge-m3-frozen-3epoch"
CLASSIFIER_PATH = Path("artifacts/models/berta_classifier_improved")

prom = pd.read_excel(RAW_DATA_DIR / "all_nomenclature.xlsx", dtype=str)
prom = prom.rename(columns={'nomenclature': 'text'})

retriever = Retriever(
    model_name=MODEL,
    index_path=FAISS_DIR / "okpd_index.faiss",
    id_map_path=FAISS_DIR / "id_map.csv",
)

engine = DecisionEngine(
    retriever,
    classifier_path=CLASSIFIER_PATH,
    auto_threshold=0.45,
    review_threshold=0.3,
    margin_threshold=0.05,
    ood_threshold=0.2,
)

modes = {"AUTO": 0, "REVIEW": 0, "MANUAL": 0}
confidences = []
total = len(prom)

for _, row in tqdm(prom.iterrows(), total=total, desc="Processing prom"):
    query = str(row['text']).strip() if pd.notna(row['text']) else ""
    if not query:
        modes["MANUAL"] += 1
        continue
    res = engine.predict(query)
    modes[res["mode"]] += 1
    confidences.append(res["confidence"])

print("\nРаспределение режимов на промке (порог AUTO=0.75):")
for mode, count in modes.items():
    print(f"  {mode}: {count} ({count/total*100:.1f}%)")
print(f"\nСредняя уверенность: {np.mean(confidences):.3f}")
print(f"Медианная уверенность: {np.median(confidences):.3f}")