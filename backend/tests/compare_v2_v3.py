# compare_v2_v3.py
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from config.settings import REFERENCE_DIR, FAISS_DIR, TRAINING_DATA_DIR
from backend.preprocessing.cleaner import TextCleaner

# ---------- 1. Hold‑out набор (300 примеров) ----------
gold = pd.read_excel(TRAINING_DATA_DIR / "train.xlsx", dtype=str)
gold = gold[["Номенклатура", "Код ОКПД2"]].dropna()
gold.columns = ["text", "true_code"]
test_df = gold.sample(300, random_state=42)

cleaner = TextCleaner(abbreviations_path=REFERENCE_DIR / "сокращения.xlsx")

# ---------- 2. Строим стеммированный индекс для модели ----------
def build_stemmed_index(model_name: str, index_path: Path, id_map_path: Path):
    okpd = pd.read_excel(REFERENCE_DIR / "okpd_2.xlsx", dtype=str)
    okpd = okpd.dropna(subset=["name"])
    okpd["name"] = okpd["name"].astype(str).str.strip()
    okpd = okpd[okpd["name"] != ""]
    okpd["name_stemmed"] = okpd["name"].apply(
        lambda x: cleaner.clean(x, use_stemmer=True)
    )

    model = SentenceTransformer(model_name, device="cpu")
    emb = model.encode(
        okpd["name_stemmed"].tolist(), batch_size=8,
        convert_to_numpy=True, show_progress_bar=True
    )
    faiss.normalize_L2(emb)
    idx = faiss.IndexFlatIP(emb.shape[1])
    idx.add(emb)
    faiss.write_index(idx, str(index_path))
    okpd[["code", "parent_code", "name"]].to_csv(id_map_path, index=False)
    return model

# ---------- 3. Тестирование ----------
def evaluate(model, index_path: Path, id_map_path: Path):
    from backend.models.retriever import Retriever
    ret = Retriever.__new__(Retriever)
    ret.model = model
    ret.index_path = index_path
    ret.id_map_path = id_map_path
    ret._loaded = False
    ret.index = None
    ret.codes = None
    ret.names = None
    ret.parent_codes = None

    def lazy_load():
        ret.index = faiss.read_index(str(index_path))
        id_map = pd.read_csv(id_map_path, dtype=str)
        ret.codes = id_map["code"].values
        ret.names = id_map["name"].values
        ret.parent_codes = id_map.get("parent_code", id_map["code"]).values
        ret._loaded = True

    hits = {1:0, 5:0, 10:0}
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Тест"):
        lazy_load()
        q = cleaner.clean(row["text"], use_stemmer=True)
        emb = model.encode([q], convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(emb)
        scores, indices = ret.index.search(emb, 10)
        for i, idx in enumerate(indices[0], 1):
            if 0 <= idx < len(ret.codes) and ret.codes[idx] == row["true_code"]:
                for k in hits:
                    if i <= k:
                        hits[k] += 1
                break
    total = len(test_df)
    return {k: hits[k]/total for k in hits}

# ---------- 4. Запуск ----------
print("Строим индекс для MiniLM (V2)...")
mini_idx = FAISS_DIR / "okpd_index_minilm_stemmed.faiss"
mini_map = FAISS_DIR / "id_map_minilm_stemmed.csv"
model_mini = build_stemmed_index("paraphrase-multilingual-MiniLM-L12-v2", mini_idx, mini_map)
res_mini = evaluate(model_mini, mini_idx, mini_map)

print("\nСтроим индекс для BGE-M3 finetuned v2 (V3)...")
bgem3_idx = FAISS_DIR / "okpd_index_bgem3_stemmed.faiss"
bgem3_map = FAISS_DIR / "id_map_bgem3_stemmed.csv"
model_bgem3 = build_stemmed_index("artifacts/models/bge-m3-finetuned-v2", bgem3_idx, bgem3_map)
res_bgem3 = evaluate(model_bgem3, bgem3_idx, bgem3_map)

print("\n=== Итоговое сравнение ===")
print(f"{'Модель':<15} {'Recall@1':<10} {'Recall@5':<10} {'Recall@10':<10}")
print("-"*50)
for name, res in [("MiniLM (V2)", res_mini), ("BGE-M3 (V3)", res_bgem3)]:
    print(f"{name:<15} {res[1]:<10.4f} {res[5]:<10.4f} {res[10]:<10.4f}")