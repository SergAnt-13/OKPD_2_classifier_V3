# backend/tests/evaluate_engine_thresholds.py
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from tqdm import tqdm
from config.settings import TRAINING_DATA_DIR, REFERENCE_DIR, FAISS_DIR
from backend.models.retriever import Retriever
from backend.models.engine import DecisionEngine

MODEL = "artifacts/models/bge-m3-frozen-3epoch"
CLASSIFIER_PATH = Path("artifacts/models/berta_classifier_improved")

gold = pd.read_excel(TRAINING_DATA_DIR / "train.xlsx", dtype=str)[["Номенклатура", "Код ОКПД2"]].dropna()
gold.columns = ["text", "code"]

retriever = Retriever(
    model_name=MODEL,
    index_path=FAISS_DIR / "okpd_index.faiss",
    id_map_path=FAISS_DIR / "id_map.csv",
)

results = []
thresholds = np.arange(0.35, 0.85, 0.05)

for thresh in tqdm(thresholds, desc="Testing thresholds"):
    engine = DecisionEngine(
        retriever,
        classifier_path=CLASSIFIER_PATH,
        auto_threshold=thresh,
        review_threshold=0.3,
        margin_threshold=0.05,
        ood_threshold=0.2,
    )
    total = len(gold)
    auto_count = 0
    auto_correct = 0
    for _, row in gold.iterrows():
        res = engine.predict(row["text"])
        if res["mode"] == "AUTO":
            auto_count += 1
            if res["predicted_code"] == row["code"]:
                auto_correct += 1
    precision = auto_correct / auto_count if auto_count > 0 else 0.0
    results.append({
        "threshold": thresh,
        "auto_share": auto_count / total,
        "precision": precision,
        "auto_count": auto_count,
    })

df_res = pd.DataFrame(results)
print("\nРезультаты варьирования auto_threshold на золотой выборке:")
print(df_res.to_string(index=False))
df_res.to_csv("auto_threshold_analysis.csv", index=False)

# Найдём порог, где precision >= 0.95 и максимальная доля AUTO
high_precision = df_res[df_res["precision"] >= 0.95]
if not high_precision.empty:
    best = high_precision.loc[high_precision["auto_share"].idxmax()]
    print(f"\nРекомендуемый порог для точности ≥ 95%: {best['threshold']:.2f}")
    print(f"Доля AUTO: {best['auto_share']:.2%}, точность: {best['precision']:.2%}")