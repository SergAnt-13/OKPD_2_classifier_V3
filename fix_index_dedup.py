# fix_index_dedup.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from backend.preprocessing.cleaner import TextCleaner
from config.settings import REFERENCE_DIR, FAISS_DIR

# Загружаем справочник
okpd = pd.read_excel(REFERENCE_DIR / "okpd_2.xlsx", dtype=str)
okpd = okpd.dropna(subset=["name"])
okpd["name"] = okpd["name"].astype(str).str.strip()
okpd["code"] = okpd["code"].astype(str).str.strip()

cleaner = TextCleaner(abbreviations_path=REFERENCE_DIR / "сокращения.xlsx")
# Создаём стеммированное название для группировки
okpd["stemmed"] = okpd["name"].apply(lambda x: cleaner.clean(x, use_stemmer=True))

# Группируем по стеммированному названию, в каждой группе оставляем код с макс. длиной
idx_max = okpd.groupby("stemmed")["code"].apply(lambda x: x.str.len().idxmax())
okpd_dedup = okpd.loc[idx_max].copy()
print(f"Удалено дубликатов: {len(okpd) - len(okpd_dedup)} (оставлено {len(okpd_dedup)} уникальных стеммированных названий)")

# Строим индекс по стеммированным названиям
model = SentenceTransformer("artifacts/models/bge-m3-frozen-3epoch", device="cuda")
emb = model.encode(okpd_dedup["stemmed"].tolist(), batch_size=32, convert_to_numpy=True, show_progress_bar=True)
faiss.normalize_L2(emb)
index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb)
faiss.write_index(index, str(FAISS_DIR / "okpd_index.faiss"))

# Сохраняем id_map (оригинальные названия)
id_map = okpd_dedup[["code", "parent_code", "name"]]
id_map.to_csv(FAISS_DIR / "id_map.csv", index=False)
print("Индекс и id_map обновлены.")