# compare_augmented.py
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from tqdm import tqdm
from config.settings import TRAINING_DATA_DIR, FAISS_DIR, REFERENCE_DIR
from backend.models.retriever import Retriever
from backend.preprocessing.cleaner import TextCleaner

# Hold‑out выборка (те же 300 примеров)
gold = pd.read_excel(TRAINING_DATA_DIR / "train.xlsx", dtype=str)
gold = gold[["Номенклатура", "Код ОКПД2"]].dropna()
gold.columns = ["text", "true_code"]
test_df = gold.sample(300, random_state=42)

cleaner = TextCleaner(abbreviations_path=REFERENCE_DIR / "сокращения.xlsx")
stemmer = get_stemmer()

# Два индекса
indices = {
    "Обычный": (FAISS_DIR / "okpd_index.faiss", FAISS_DIR / "id_map.csv"),
    "Обогащённый (Label Augmentation)": (FAISS_DIR / "okpd_index_augmented.faiss", FAISS_DIR / "id_map_augmented.csv"),
}

results = {}
for name, (idx_path, map_path) in indices.items():
    if not idx_path.exists():
        print(f"{name}: индекс не найден, пропускаем")
        continue
    ret = Retriever(
        model_name="artifacts/models/bge-m3-finetuned-v2",
        index_path=idx_path,
        id_map_path=map_path,
    )
    hits = {1:0, 5:0, 10:0}
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=name):
        q = cleaner.clean(row["text"], use_stemmer=True)
        cands = ret.search(q, top_k=10)["candidates"]
        for i, c in enumerate(cands, 1):
            if c["code"] == row["true_code"]:
                for k in hits:
                    if i <= k:
                        hits[k] += 1
                break
    total = len(test_df)
    results[name] = {k: hits[k]/total for k in hits}

print("\n=== Честное сравнение (одна модель, разные индексы) ===")
print(f"{'Индекс':<30} {'Recall@1':<10} {'Recall@5':<10} {'Recall@10':<10}")
print("-" * 70)
for name, metrics in results.items():
    print(f"{name:<30} {metrics[1]:<10.4f} {metrics[5]:<10.4f} {metrics[10]:<10.4f}")
