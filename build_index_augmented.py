# build_index_augmented.py
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from config.settings import REFERENCE_DIR, FAISS_DIR

def build_hierarchy_map(okpd_df: pd.DataFrame) -> dict:
    # Очистка: убираем NaN, приводим к str
    okpd_df = okpd_df.dropna(subset=["code", "name"])
    okpd_df["code"] = okpd_df["code"].astype(str).str.strip()
    okpd_df["name"] = okpd_df["name"].astype(str).str.strip()
    okpd_df["parent_code"] = okpd_df["parent_code"].astype(str).str.strip()
    okpd_df = okpd_df[okpd_df["code"] != ""]

    code_to_parent = dict(zip(okpd_df["code"], okpd_df["parent_code"]))
    code_to_name = dict(zip(okpd_df["code"], okpd_df["name"]))

    def get_full_path(code: str) -> str:
        parts = []
        current = code
        visited = set()
        while current and current in code_to_name and current not in visited:
            visited.add(current)
            parts.append(code_to_name[current])
            parent = code_to_parent.get(current)
            if parent == current or parent in visited:
                break
            current = parent
        parts.reverse()
        return " → ".join(parts) if parts else code_to_name.get(code, str(code))

    return {code: get_full_path(code) for code in okpd_df["code"]}

# Загружаем справочник
okpd = pd.read_excel(REFERENCE_DIR / "okpd_2.xlsx", dtype=str)
hierarchy_map = build_hierarchy_map(okpd)

# Обогащаем названия
okpd = okpd.dropna(subset=["code"])
okpd["code"] = okpd["code"].astype(str).str.strip()
okpd["augmented_name"] = okpd["code"].map(hierarchy_map)
okpd = okpd.dropna(subset=["augmented_name"])
okpd = okpd[okpd["augmented_name"] != ""]

print("Пример обогащённого названия:")
sample = okpd["augmented_name"].iloc[0] if len(okpd) > 0 else "Нет данных"
print(f"  {sample}\n")

# Строим FAISS-индекс
model = SentenceTransformer("artifacts/models/bge-m3-finetuned-v2", device="cpu")
texts = okpd["augmented_name"].tolist()
print(f"Кодирование {len(texts)} текстов...")
embeddings = model.encode(texts, batch_size=8, convert_to_numpy=True, show_progress_bar=True)
faiss.normalize_L2(embeddings)

dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)

index_path = FAISS_DIR / "okpd_index_augmented.faiss"
faiss.write_index(index, str(index_path))
print(f"FAISS index saved to {index_path}")

id_map = okpd[["code", "parent_code", "name"]].copy()
id_map_path = FAISS_DIR / "id_map_augmented.csv"
id_map.to_csv(id_map_path, index=False)
print(f"ID map saved to {id_map_path}")
print("Готово!")