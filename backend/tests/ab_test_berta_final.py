# backend/tests/ab_test_berta_final.py
import sys, os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from tqdm import tqdm

from config.settings import TRAINING_DATA_DIR, REFERENCE_DIR, FAISS_DIR
from backend.preprocessing.cleaner import TextCleaner
from backend.models.retriever import Retriever
from backend.models.reranker import Reranker
from backend.models.engine import DecisionEngine

# ---------- Настройки ----------
MODEL_NAME = "artifacts/models/bge-m3-frozen-3epoch"
RERANKER_PATH = "artifacts/models/berta_reranker_gold_6ep"   # BERTA реранкер
NORMALIZED_GOLD = "gold_normalized.csv"                      # выход LLM
USE_RERANKER = True                                          # поставим True для сравнения
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
# -----------------------------

# 1. Загружаем нормализованное золото + коды
gold_norm = pd.read_csv(NORMALIZED_GOLD, dtype=str)
original_gold = pd.read_excel(TRAINING_DATA_DIR / "train.xlsx", dtype=str)[["Номенклатура", "Код ОКПД2"]].dropna()
original_gold.columns = ["text_original", "code"]

# Объединяем по исходному названию
gold = gold_norm.merge(original_gold, on="text_original", how="inner")
print(f"Записей после слияния: {len(gold)}")

# 2. Инициализация retriever и engine (без реранкера)
cleaner = TextCleaner(abbreviations_path=REFERENCE_DIR / "сокращения.xlsx")
retriever = Retriever(
    model_name=MODEL_NAME,
    index_path=FAISS_DIR / "okpd_index.faiss",        # текущий стеммированный индекс
    id_map_path=FAISS_DIR / "id_map.csv",
)

engine_dense = DecisionEngine(retriever)

# 3. Инициализация engine с BERTA-реранкером (если нужно)
if USE_RERANKER and Path(RERANKER_PATH).exists():
    reranker = Reranker(RERANKER_PATH)
    engine_berta = DecisionEngine(retriever, reranker=reranker)
    print("BERTA реранкер загружен")
else:
    engine_berta = None
    print("BERTA реранкер не найден — сравнение только Dense")

# 4. Функция оценки
def evaluate(engine, df, label):
    hits = {1:0, 3:0, 5:0, 10:0}
    ndcg_sum = 0.0
    modes = {"AUTO":0, "REVIEW":0, "MANUAL":0}
    total = len(df)

    for _, row in tqdm(df.iterrows(), total=total, desc=label):
        query = cleaner.clean(row["text_normalized"], use_stemmer=True)  # стемминг запроса
        true_code = row["code"]
        res = engine.predict(query, use_reranker=(engine is engine_berta))
        pred_code = res["predicted_code"]
        modes[res["mode"]] += 1

        # Считаем метрики по top-10 кандидатам (уже после engine, но для чистоты возьмём из top_candidates)
        # engine возвращает top_candidates уже отсортированными
        candidates = res["top_candidates"]
        for i, c in enumerate(candidates, 1):
            if c["code"] == true_code:
                for k in hits:
                    if i <= k:
                        hits[k] += 1
                ndcg_sum += 1.0 / np.log2(i + 1)
                break

    for k in hits:
        hits[k] /= total
    ndcg = ndcg_sum / total
    return hits, ndcg, modes

# 5. Запуск
print("\n=== Dense only ===")
dense_hits, dense_ndcg, dense_modes = evaluate(engine_dense, gold, "Dense only")

if engine_berta:
    print("\n=== Dense + BERTA ===")
    berta_hits, berta_ndcg, berta_modes = evaluate(engine_berta, gold, "Dense + BERTA")
else:
    berta_hits = {1:0, 3:0, 5:0, 10:0}
    berta_ndcg = 0.0
    berta_modes = {"AUTO":0, "REVIEW":0, "MANUAL":0}

# 6. Вывод результатов
print("\n" + "="*60)
print("A/B-ТЕСТ: DENSE vs DENSE+BERTA (нормализованные запросы)")
print("="*60)
print(f"{'Конфигурация':<20} {'R@1':<8} {'R@3':<8} {'R@5':<8} {'R@10':<8} {'NDCG@10':<8}")
print("-"*60)
print(f"{'Dense only':<20} {dense_hits[1]:<8.4f} {dense_hits[3]:<8.4f} {dense_hits[5]:<8.4f} {dense_hits[10]:<8.4f} {dense_ndcg:<8.4f}")
if engine_berta:
    print(f"{'Dense + BERTA':<20} {berta_hits[1]:<8.4f} {berta_hits[3]:<8.4f} {berta_hits[5]:<8.4f} {berta_hits[10]:<8.4f} {berta_ndcg:<8.4f}")

print("\nРаспределение режимов:")
print(f"{'Режим':<10} {'Dense only':<15} {'Dense + BERTA':<15}")
print(f"{'AUTO':<10} {dense_modes['AUTO']:<15} {berta_modes['AUTO']:<15}")
print(f"{'REVIEW':<10} {dense_modes['REVIEW']:<15} {berta_modes['REVIEW']:<15}")
print(f"{'MANUAL':<10} {dense_modes['MANUAL']:<15} {berta_modes['MANUAL']:<15}")

# Дополнительно: точность AUTO
if dense_modes['AUTO'] > 0:
    # Подсчитаем точность среди AUTO (но это потребует повторного прогона, здесь просто показываем кол-во)
    pass  # можно добавить, но не критично