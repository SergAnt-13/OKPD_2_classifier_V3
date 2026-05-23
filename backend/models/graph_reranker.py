# backend/models/graph_reranker.py
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

class GraphReranker:
    """
    Легковесный графовый реранкер.
    Строит граф на кандидатах, используя эмбеддинги модели, и переранжирует их
    с помощью персонализированного PageRank.
    """

    def __init__(self, model, alpha: float = 0.85, tol: float = 1e-6):
        self.model = model
        self.alpha = alpha
        self.tol = tol

    def _build_graph(self, query_embedding: np.ndarray, candidate_embeddings: np.ndarray):
        """Строит матрицу смежности графа кандидатов."""
        # Нормализуем векторы для cosine similarity
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        cands_norm = candidate_embeddings / (np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-10)

        # Сходство между кандидатами
        sim_matrix = np.dot(cands_norm, cands_norm.T)
        # Сходство запроса с каждым кандидатом (для персонализации)
        query_sim = np.dot(cands_norm, query_norm.T).flatten()

        # Оставляем только сильные связи (top-k соседей), k = 5
        k = min(5, len(candidate_embeddings) - 1)
        for i in range(len(candidate_embeddings)):
            row = sim_matrix[i]
            # Обнуляем все, кроме k+1 наибольших (сам с собой + k соседей)
            threshold = np.sort(row)[- (k + 1)]
            sim_matrix[i, row < threshold] = 0.0

        # Нормализуем столбцы (матрица переходов)
        col_sums = sim_matrix.sum(axis=0)
        col_sums[col_sums == 0] = 1.0  # избегаем деления на 0
        transition_matrix = sim_matrix / col_sums

        return transition_matrix, query_sim

    def _pagerank(self, transition_matrix, personalization):
        """Вычисляет персонализированный PageRank."""
        n = transition_matrix.shape[0]
        # Равномерное начальное распределение
        v = np.ones(n) / n
        # Персонализация: чем ближе кандидат к запросу, тем выше его вес
        p = personalization / (personalization.sum() + 1e-10)

        for _ in range(100):
            v_new = self.alpha * transition_matrix @ v + (1 - self.alpha) * p
            if np.linalg.norm(v_new - v, 1) < self.tol:
                break
            v = v_new
        return v

    def rerank(self, query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
        """
        Переранжирует кандидатов с помощью графового PageRank.
        Возвращает top_k кандидатов, отсортированных по новому скору.
        """
        if len(candidates) <= top_k:
            return candidates

        # Получаем эмбеддинги запроса и кандидатов
        texts = [query] + [c["name"] for c in candidates]
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        query_emb = embeddings[0:1]
        cand_embs = embeddings[1:]

        # Строим граф и вычисляем PageRank
        trans_mat, query_sim = self._build_graph(query_emb, cand_embs)
        scores = self._pagerank(trans_mat, query_sim)

        # Присваиваем новые скоры
        for cand, score in zip(candidates, scores):
            cand["graph_score"] = float(score)

        # Сортируем и возвращаем top_k
        candidates.sort(key=lambda x: x.get("graph_score", 0.0), reverse=True)
        return candidates[:top_k]