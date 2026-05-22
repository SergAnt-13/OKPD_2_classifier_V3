# backend/models/reranker.py
# Purpose: Cross-Encoder reranker using BAAI/bge-reranker-v2-m3.

from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model = CrossEncoder(model_name, max_length=512)

    def rerank(self, query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
        pairs = [(query, cand["name"]) for cand in candidates]
        scores = self.model.predict(pairs, show_progress_bar=False)

        for cand, score in zip(candidates, scores):
            cand["rerank_score"] = float(score)

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[:top_k]