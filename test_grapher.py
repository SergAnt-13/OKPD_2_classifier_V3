# test_grapher.py (исправленный)
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from tqdm import tqdm
from sentence_transformers import CrossEncoder

from config.settings import TRAINING_DATA_DIR, FAISS_DIR, REFERENCE_DIR
from backend.models.retriever import Retriever
from backend.models.graph_reranker import GraphReranker
from backend.preprocessing.cleaner import TextCleaner

# Hold‑out выборка (те же 300 примеров)
gold = pd.read_excel(TRAINING_DATA_DIR / "train.xlsx", dtype=str)
gold = gold[["Номенклатура", "Код ОКПД2"]].dropna()
gold.columns = ["text", "true_code"]
test_df = gold.sample(300, random_state=42)

cleaner = TextCleaner(abbreviations_path=REFERENCE_DIR / "сокращения.xlsx")
stemmer = get_stemmer()

# Загружаем модели
ret = Retriever(
    model_name="artifacts/models/bge-m3-finetuned-v2",
    index_path=FAISS_DIR / "okpd_index.faiss",
    id_map_path=FAISS_DIR / "id_map.csv",
)
# Исправленный порядок аргументов: [doc, query]
cross_encoder = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)
graph_reranker = GraphReranker(ret.model)

def evaluate(config_name: str, retriever, rerank_mode: str = "none", top_k: int = 5) -> dict:
    hits = {1:0, 5:0, 10:0}
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=config_name):
        q = cleaner.clean(row["text"], use_stemmer=True)
        # Получаем кандидатов от Dense retrieval (исправлено: используем метод search)
        raw_result = retriever.search(q, top_k=20 if rerank_mode in ("graph_only", "graph_then_ce") else 100)
        candidates = raw_result["candidates"]

        if rerank_mode == "cross_encoder":
            # Исправленный порядок аргументов: [c["name"], q]
            pairs = [(c["name"], q) for c in candidates]
            ce_scores = cross_encoder.predict(pairs, show_progress_bar=False)
            for c, s in zip(candidates, ce_scores):
                c["score"] = float(s)
            candidates.sort(key=lambda x: x["score"], reverse=True)
            candidates = candidates[:top_k]
        elif rerank_mode == "graph_only":
            candidates = graph_reranker.rerank(q, candidates, top_k=top_k)
        elif rerank_mode == "graph_then_ce":
            candidates = graph_reranker.rerank(q, candidates, top_k=5)
            pairs = [(c["name"], q) for c in candidates]
            ce_scores = cross_encoder.predict(pairs, show_progress_bar=False)
            for c, s in zip(candidates, ce_scores):
                c["score"] = float(s)
            candidates.sort(key=lambda x: x["score"], reverse=True)

        for i, c in enumerate(candidates[:10], 1):
            if c["code"] == row["true_code"]:
                for k in hits:
                    if i <= k:
                        hits[k] += 1
                break
    total = len(test_df)
    return {k: hits[k]/total for k in hits}

# Конфигурации
configs = {
    "Dense only (без реранка)": "none",
    "Dense + Cross-Encoder (исправленный)": "cross_encoder",
    "Dense + GraphER": "graph_only",
    "Dense + GraphER + Cross-Encoder": "graph_then_ce",
}

print("\n=== A/B‑тест (исправленный порядок аргументов) ===\n")
results = {}
for name, mode in configs.items():
    results[name] = evaluate(name, ret, rerank_mode=mode)

print("\nРезультаты:")
print(f"{'Конфигурация':<45} {'Recall@1':<10} {'Recall@5':<10} {'Recall@10':<10}")
print("-" * 80)
for name, metrics in results.items():
    print(f"{name:<45} {metrics[1]:<10.4f} {metrics[5]:<10.4f} {metrics[10]:<10.4f}")