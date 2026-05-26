# backend/models/ood.py
"""
Простая OOD‑детекция для retrieval‑first системы.
Использует максимальный скор из FAISS‑поиска: если он ниже порога,
запрос считается вне‑распределения (Out‑of‑Distribution).
"""

from backend.models.retriever import Retriever

def is_ood(
    query: str,
    retriever: Retriever,
    threshold: float = 0.3,   # порог подбирается эмпирически
    top_k: int = 5,
) -> bool:
    """
    Возвращает True, если запрос признан OOD.
    Проверяем максимальный скор среди top‑k кандидатов.
    """
    result = retriever.search(query, top_k=top_k, use_reranker=False)
    candidates = result.get("candidates", [])
    if not candidates:
        return True          # совсем нет кандидатов → точно OOD
    max_score = max(c["score"] for c in candidates)
    return max_score < threshold