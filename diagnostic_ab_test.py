# diagnostic_ab_test.py
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from tqdm import tqdm
from config.settings import TRAINING_DATA_DIR, REFERENCE_DIR, FAISS_DIR
from backend.preprocessing.cleaner import TextCleaner
from backend.models.retriever import Retriever, build_faiss_index

# Загружаем 200 случайных примеров
gold = pd.read_excel(TRAINING_DATA_DIR / "train.xlsx", dtype=str)
gold = gold[["Номенклатура", "Код ОКПД2"]].dropna()
gold.columns = ["text", "true_code"]
test_sample = gold.sample(min(200, len(gold)), random_state=42)

cleaner = TextCleaner(abbreviations_path=REFERENCE_DIR / "сокращения.xlsx")
stemmer = get_stemmer()

models_to_test = {
    "MiniLM (V2)": "paraphrase-multilingual-MiniLM-L12-v2",
    "BGE-M3 (base)": "BAAI/bge-m3",
    "BGE-M3 (finetuned v2)": "artifacts/models/bge-m3-finetuned-v2",
}

for name, model_path in models_to_test.items():
    try:
        ret = Retriever(model_name=model_path, index_path=FAISS_DIR/"okpd_index.faiss", id_map_path=FAISS_DIR/"id_map.csv")
        hits = {1:0, 5:0, 10:0}
        for _, row in tqdm(test_sample.iterrows(), total=len(test_sample), desc=name):
            q = cleaner.clean(row["text"], use_stemmer=True)
            cands = ret.search(q, top_k=10)["candidates"]
            for i, c in enumerate(cands, 1):
                if c["code"] == row["true_code"]:
                    for k in hits:
                        if i <= k:
                            hits[k] += 1
                    break
        total = len(test_sample)
        print(f"{name}: R@1={hits[1]/total:.4f}, R@5={hits[5]/total:.4f}, R@10={hits[10]/total:.4f}")
    except Exception as e:
        print(f"{name}: ERROR — {e}")