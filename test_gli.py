# test_gli.py (финальная версия)
# Purpose: A/B-тест GLiClass vs bge-reranker на 200 примерах без визуального шума.
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import faiss
from tqdm import tqdm
from config.settings import REFERENCE_DIR, TRAINING_DATA_DIR
from backend.preprocessing.cleaner import TextCleaner
from backend.preprocessing.stemmer import get_stemmer
from backend.models.retriever import Retriever
from backend.models.gli_scorer import GLiScorer

# Инициализация
cleaner = TextCleaner(abbreviations_path=REFERENCE_DIR / "сокращения.xlsx")
stemmer = get_stemmer()
retriever = Retriever(model_name="BAAI/bge-m3")
gli = GLiScorer()

# Загружаем 200 случайных примеров из золотой выборки
gold = pd.read_excel(TRAINING_DATA_DIR / "train.xlsx", dtype=str)
gold = gold[["Номенклатура", "Код ОКПД2"]].dropna()
gold.columns = ["text", "true_code"]
sample = gold.sample(min(700, len(gold)), random_state=42)

print(f"Сравнение GLiClass и bge-reranker на {len(sample)} примерах...\n")

rerank_hits = 0  # сколько раз реранкер попал в топ-5
gli_hits = 0  # сколько раз GLiClass попал в топ-5
rerank_better = 0  # ранг реранкера лучше (меньше), чем у GLiClass
gli_better = 0  # ранг GLiClass лучше
equal = 0  # одинаковый ранг
total = 0

for _, row in tqdm(sample.iterrows(), total=len(sample), desc="A/B тест", disable=False):
    text = row["text"]
    true_code = row["true_code"].strip()

    # Стеммированный запрос для retrieval и реранкера
    query_stemmed = cleaner.clean(text, stemmer=stemmer)

    # Получаем кандидатов от Dense + реранкер (top-5)
    result = retriever.search(query_stemmed, top_k=5)
    candidates_rerank = result["candidates"]

    rank_rerank = None
    for i, cand in enumerate(candidates_rerank, 1):
        if cand["code"] == true_code:
            rank_rerank = i
            break

    # Нестеммированный запрос для GLiClass
    query_clean = cleaner.clean(text)  # без стеммера

    # Получаем 20 кандидатов от Dense (без реранка) для GLi
    embedding = retriever.model.encode([query_stemmed], convert_to_numpy=True, show_progress_bar=False)
    faiss.normalize_L2(embedding)
    scores, indices = retriever.index.search(embedding, 20)
    gli_candidates = []
    for score, idx in zip(scores[0], indices[0]):
        if 0 <= idx < len(retriever.codes):
            gli_candidates.append({
                "code": retriever.codes[idx],
                "name": retriever.names[idx],
            })
    gli_candidates = gli.score(query_clean, gli_candidates)

    rank_gli = None
    for i, cand in enumerate(gli_candidates[:5], 1):
        if cand["code"] == true_code:
            rank_gli = i
            break

    # Сбор статистики
    total += 1
    if rank_rerank is not None:
        rerank_hits += 1
    if rank_gli is not None:
        gli_hits += 1

    if rank_rerank is not None and rank_gli is not None:
        if rank_rerank < rank_gli:
            rerank_better += 1
        elif rank_gli < rank_rerank:
            gli_better += 1
        else:
            equal += 1
    elif rank_rerank is not None:
        rerank_better += 1
    elif rank_gli is not None:
        gli_better += 1

print("\n" + "=" * 50)
print("РЕЗУЛЬТАТЫ A/B-ТЕСТА")
print("=" * 50)
print(f"Всего примеров:        {total}")
print(f"Rerank попал в топ-5:   {rerank_hits} ({rerank_hits / total:.1%})")
print(f"GLiClass попал в топ-5: {gli_hits} ({gli_hits / total:.1%})")
print(f"Rerank лучше:           {rerank_better} ({rerank_better / total:.1%})")
print(f"GLiClass лучше:         {gli_better} ({gli_better / total:.1%})")
print(f"Одинаково:              {equal} ({equal / total:.1%})")

# Решение
if gli_better > rerank_better * 1.2:  # GLiClass значимо лучше
    print("\n✅ GLiClass ПОЛЕЗЕН — включаем в engine как дополнительный сигнал.")
elif rerank_better > gli_better * 1.2:  # Ререранкер значимо лучше
    print("\n❌ GLiClass НЕ ДАЁТ ПРИРОСТА — отключаем, фокусируемся на мета-модели.")
else:  # Примерно равны
    print("\n🟡 Результаты сопоставимы. GLiClass можно включить как опциональный сигнал, но без гарантий прироста.")