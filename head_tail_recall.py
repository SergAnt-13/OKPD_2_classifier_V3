# head_tail_recall.py
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from collections import Counter
from config.settings import TRAINING_DATA_DIR, FAISS_DIR, REFERENCE_DIR
from backend.models.retriever import Retriever
from backend.preprocessing.cleaner import TextCleaner
from tqdm import tqdm

# 1. Загружаем золотую выборку и определяем head/mid/tail
gold = pd.read_excel(TRAINING_DATA_DIR / "train.xlsx", dtype=str)
gold = gold[["Номенклатура", "Код ОКПД2"]].dropna()
gold.columns = ["text", "true_code"]

code_counts = Counter(gold["true_code"])
total_codes = len(code_counts)
head_codes = set([code for code, _ in code_counts.most_common(int(total_codes * 0.2))])
mid_codes = set([code for code, _ in code_counts.most_common(int(total_codes * 0.5))]) - head_codes
tail_codes = set(code_counts.keys()) - head_codes - mid_codes

# 2. Те же 300 тестовых примеров, что и в holdout_test.py
test_df = gold.sample(300, random_state=42)

# 3. Инициализация
ret = Retriever(
    model_name="artifacts/models/holdout_model",
    index_path=FAISS_DIR / "okpd_index.faiss",
    id_map_path=FAISS_DIR / "id_map.csv"
)
cleaner = TextCleaner(abbreviations_path=REFERENCE_DIR / "сокращения.xlsx")
stemmer = get_stemmer()

# 4. Считаем Recall@10 для каждого сегмента
for segment_name, segment_codes in [("Head", head_codes), ("Mid", mid_codes), ("Tail", tail_codes)]:
    segment_df = test_df[test_df["true_code"].isin(segment_codes)]
    if len(segment_df) == 0:
        print(f"{segment_name}: нет примеров")
        continue
    hits = 0
    for _, row in tqdm(segment_df.iterrows(), total=len(segment_df), desc=segment_name):
        q = cleaner.clean(row["text"], use_stemmer=True)
        cands = ret.search(q, top_k=10)["candidates"]
        for i, c in enumerate(cands, 1):
            if c["code"] == row["true_code"] and i <= 10:
                hits += 1
                break
    recall = hits / len(segment_df)
    print(f"{segment_name}: Recall@10 = {recall:.4f} (n={len(segment_df)})")