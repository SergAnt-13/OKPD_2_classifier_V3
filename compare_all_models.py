# compare_all_models.py
# Независимое сравнение нескольких моделей на одних и тех же данных.
# Каждая модель получает свой FAISS-индекс, который строится автоматически при первом запуске.
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from sentence_transformers import SentenceTransformer
from config.settings import TRAINING_DATA_DIR, REFERENCE_DIR, FAISS_DIR
from backend.preprocessing.cleaner import TextCleaner
from backend.models.retriever import Retriever, build_faiss_index as _build_faiss

# --- Имена моделей и соответствующие суффиксы для файлов индексов ---
MODELS = {
    "MiniLM (base)": "paraphrase-multilingual-MiniLM-L12-v2",
    "BGE-M3 (base)": "BAAI/bge-m3",
    "BGE-M3 (finetuned v2)": "artifacts/models/bge-m3-finetuned-v2",
}

# --- Hold-out выборка (300 примеров, фиксированный random_state) ---
gold = pd.read_excel(TRAINING_DATA_DIR / "train.xlsx", dtype=str)
gold = gold[["Номенклатура", "Код ОКПД2"]].dropna()
gold.columns = ["text", "true_code"]
_, test_df = train_test_split(gold, test_size=300, random_state=42)

cleaner = TextCleaner(abbreviations_path=REFERENCE_DIR / "сокращения.xlsx")

results = {}

for display_name, model_path in MODELS.items():
    print(f"\n{'='*60}")
    print(f"  {display_name}  ({model_path})")
    print('='*60)

    # Уникальное имя для индекса и маппинга
    safe_name = model_path.replace("/", "_").replace("\\", "_").replace(":", "_")
    index_path = FAISS_DIR / f"okpd_index_{safe_name}.faiss"
    id_map_path = FAISS_DIR / f"id_map_{safe_name}.csv"

    # Строим индекс, если его ещё нет
    if not index_path.exists():
        print(f"  Строим FAISS-индекс...")
        _build_faiss(
            model_name=model_path,
            index_path=index_path,
            id_map_path=id_map_path,
            reference_path=REFERENCE_DIR / "okpd_2.xlsx",
            batch_size=8,   # для Mac CPU
        )
    else:
        print(f"  Индекс уже существует, пропускаем построение.")

    # Создаём Retriever
    ret = Retriever(
        model_name=model_path,
        index_path=index_path,
        id_map_path=id_map_path,
    )

    # Замеряем Recall@1, @5, @10
    hits = {1:0, 5:0, 10:0}
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"  Оценка {display_name}"):
        q = cleaner.clean(row["text"], use_stemmer=True)
        try:
            cands = ret.search(q, top_k=10)["candidates"]
        except Exception as e:
            print(f"  Ошибка поиска: {e}")
            continue
        for i, c in enumerate(cands, 1):
            if c["code"] == row["true_code"]:
                for k in hits:
                    if i <= k:
                        hits[k] += 1
                break
    total = len(test_df)
    results[display_name] = {k: hits[k]/total for k in hits}
    print(f"  Результат: Recall@1={results[display_name][1]:.4f}, Recall@5={results[display_name][5]:.4f}, Recall@10={results[display_name][10]:.4f}")

# --- Итоговая таблица ---
print("\n\n" + "=" * 80)
print("ИТОГОВОЕ СРАВНЕНИЕ (Hold‑out 300 примеров, Mac CPU)")
print("=" * 80)
print(f"{'Модель':<25} {'Recall@1':<10} {'Recall@5':<10} {'Recall@10':<10}")
print("-" * 80)
for name, metrics in results.items():
    print(f"{name:<25} {metrics[1]:<10.4f} {metrics[5]:<10.4f} {metrics[10]:<10.4f}")