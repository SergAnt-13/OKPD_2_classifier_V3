# prepare_hard_negatives.py
import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from tqdm import tqdm

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from sentence_transformers import CrossEncoder
from config.settings import TRAINING_DATA_DIR, REFERENCE_DIR, FAISS_DIR
from backend.preprocessing.cleaner import TextCleaner
from backend.models.retriever import Retriever

# ---------------------------- 1. Загрузка данных ----------------------------
gold = pd.read_excel(TRAINING_DATA_DIR / "train.xlsx", dtype=str)
gold = gold[["Номенклатура", "Код ОКПД2"]].dropna()
gold.columns = ["text", "true_code"]

cleaner = TextCleaner(abbreviations_path=REFERENCE_DIR / "сокращения.xlsx")
stemmer = get_stemmer()

# ---------------------------- 2. Инициализация retriever и реранкера ----------------------------
ret = Retriever(
    model_name="artifacts/models/bge-m3-finetuned-v2",
    index_path=FAISS_DIR / "okpd_index.faiss",
    id_map_path=FAISS_DIR / "id_map.csv"
)
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)

# ---------------------------- 3. Генерация hard negatives ----------------------------
hard_negatives = []  # список словарей {text, positive_code, negatives: [code1, code2, ...]}

for _, row in tqdm(gold.iterrows(), total=len(gold), desc="Hard negatives"):
    text = row["text"]
    true_code = row["true_code"].strip()

    # Очистка запроса (стемминг, как при инференсе)
    query_stemmed = cleaner.clean(text, stemmer=stemmer)

    # Получаем топ-50 кандидатов от dense retrieval
    try:
        candidates = ret.search(query_stemmed, top_k=50)["candidates"]
    except Exception as e:
        print(f"Ошибка поиска для '{text}': {e}")
        continue

    # Отбираем только неверные коды (исключаем истинный)
    wrong_candidates = [c for c in candidates if c["code"] != true_code]
    if len(wrong_candidates) < 5:
        continue  # недостаточно кандидатов, пропускаем пример

    # Используем реранкер, чтобы найти самые сложные негативы
    pairs = [(text, c["name"]) for c in wrong_candidates]
    scores = reranker.predict(pairs, show_progress_bar=False)

    # Сортируем по убыванию скора (наиболее похожие на запрос = самые трудные)
    wrong_candidates_sorted = [c for _, c in sorted(zip(scores, wrong_candidates), key=lambda x: x[0], reverse=True)]

    # Берём топ-5
    top_negatives = wrong_candidates_sorted[:5]

    hard_negatives.append({
        "text": text,
        "positive_code": true_code,
        "negatives": [c["code"] for c in top_negatives]
    })

# ---------------------------- 4. Сохранение результата ----------------------------
output_path = TRAINING_DATA_DIR / "hard_negatives.pkl"
pd.to_pickle(hard_negatives, output_path)
print(f"Сохранено {len(hard_negatives)} примеров с hard negatives в {output_path}")