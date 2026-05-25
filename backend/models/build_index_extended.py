# backend/models/build_index_extended.py
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from config.settings import REFERENCE_DIR, RAW_DATA_DIR, FAISS_DIR

# Загружаем существующий id_map (основной индекс)
id_map = pd.read_csv(FAISS_DIR / "id_map.csv", dtype=str)
existing_codes = set(id_map["code"])

# Добавляем разделы 17, 22, 20 из справочника
okpd = pd.read_excel(REFERENCE_DIR / "okpd_2.xlsx", dtype=str)
okpd = okpd.dropna(subset=["name"])
okpd["name"] = okpd["name"].astype(str).str.strip()
new_sections = okpd[okpd["code"].str.startswith(('17', '22', '20')) & ~okpd["code"].isin(existing_codes)]
print(f"Новых кодов из разделов 17,22,20: {len(new_sections)}")

# Добавляем реальные названия из промки для существующих кодов
prom = pd.read_excel(RAW_DATA_DIR / "all_nomenclature.xlsx", dtype=str)
prom = prom[['nomenclature', 'okpd2_code']].dropna()
prom.columns = ["name", "code"]
prom = prom[prom["code"].isin(existing_codes)]  # только коды, уже есть в индексе
prom = prom.drop_duplicates(subset=["name"])
print(f"Реальных названий из промки: {len(prom)}")

# Объединяем
extended = pd.concat([new_sections[["code", "name"]], prom], ignore_index=True)
extended = extended.dropna(subset=["name"])
extended["name"] = extended["name"].astype(str).str.strip()

# Стемминг
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("russian")
extended["name_stemmed"] = extended["name"].apply(
    lambda x: " ".join(stemmer.stem(w) for w in x.split())
)

# Модель
model = SentenceTransformer("artifacts/models/bge-m3-frozen-stratified-epoch2", device="cpu")
texts = extended["name_stemmed"].tolist()
embeddings = model.encode(texts, batch_size=8, convert_to_numpy=True, show_progress_bar=True)
faiss.normalize_L2(embeddings)

# Загружаем существующий индекс
index = faiss.read_index(str(FAISS_DIR / "okpd_index.faiss"))
index.add(embeddings)

# Сохраняем расширенный индекс
faiss.write_index(index, str(FAISS_DIR / "okpd_index_extended.faiss"))

# Обновлённый id_map
extended["code"] = extended["code"].astype(str)
extended["parent_code"] = ""  # опционально
full_id_map = pd.concat([id_map, extended[["code", "parent_code", "name"]]], ignore_index=True)
full_id_map.to_csv(FAISS_DIR / "id_map_extended.csv", index=False)
print(f"Расширенный индекс сохранён: {FAISS_DIR / 'okpd_index_extended.faiss'}")