# backend/models/retriever.py
# Purpose: Dense retrieval + Cross-Encoder reranking.
# Uses BAAI/bge-m3 for retrieval and BAAI/bge-reranker-v2-m3 for reranking.

import faiss
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
from backend.models.gli_scorer import GLiScorer
from backend.preprocessing.cleaner import TextCleaner

import torch
from sentence_transformers import SentenceTransformer
from backend.models.reranker import Reranker
from config.settings import FAISS_DIR, REFERENCE_DIR

# Глобальный кэш для моделей (ленивая загрузка)
_MODEL_CACHE = {}
_RERANKER_CACHE = {}

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

class Retriever:
    def __init__(
            self,
            model_name: str = "artifacts/models/bge-m3-finetuned",
            index_path: Optional[Path] = None,
            id_map_path: Optional[Path] = None,
            reranker_model: str = "BAAI/bge-reranker-v2-m3",
            use_gli: bool = False,
    ):
        # Ленивая загрузка энкодера
        if model_name not in _MODEL_CACHE:
            _MODEL_CACHE[model_name] = SentenceTransformer(model_name, device=get_device())
        self.model = _MODEL_CACHE[model_name]

        # Ленивая загрузка реранкера
        if reranker_model not in _RERANKER_CACHE:
            _RERANKER_CACHE[reranker_model] = Reranker(reranker_model)
        self.reranker = _RERANKER_CACHE[reranker_model]

        self.index_path = index_path or FAISS_DIR / "okpd_index.faiss"
        self.id_map_path = id_map_path or FAISS_DIR / "id_map.csv"
        self._loaded = False
        self.index = None
        self.codes = None
        self.parent_codes = None
        self.names = None
        self.gli = GLiScorer() if use_gli else None
        self.cleaner = TextCleaner(abbreviations_path=REFERENCE_DIR / "сокращения.xlsx")
        self.gli = GLiScorer() if use_gli else None

    def _lazy_load(self):
        if self._loaded:
            return
        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
        if not self.id_map_path.exists():
            raise FileNotFoundError(f"ID map not found: {self.id_map_path}")
        self.index = faiss.read_index(str(self.index_path))
        id_map = pd.read_csv(self.id_map_path, dtype=str)
        self.codes = id_map["code"].values
        self.parent_codes = id_map.get("parent_code", id_map["code"]).values
        self.names = id_map["name"].values
        self._loaded = True

    def search(self, query: str, top_k: int = 5) -> Dict:
        self._lazy_load()
        # Стемминг теперь внутри cleaner.clean(…, use_stemmer=True)
        query_norm = self.cleaner.clean(query, use_stemmer=True)
        if not query_norm.strip():
            return {"candidates": []}

        embedding = self.model.encode([query_norm], convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(embedding)
        scores, indices = self.index.search(embedding, top_k * 4)

        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self.codes):
                candidates.append({
                    "code": self.codes[idx],
                    "parent_code": self.parent_codes[idx],
                    "score": float(score),
                    "name": self.names[idx],
                })

        candidates = self.reranker.rerank(query, candidates, top_k=top_k)
        return {"candidates": candidates}

def build_faiss_index(
    reference_path: Optional[Path] = None,
    model_name: str = "BAAI/bge-m3",
    batch_size: int = 32,
    index_path: Optional[Path] = None,
    id_map_path: Optional[Path] = None,
):
    reference_path = reference_path or REFERENCE_DIR / "okpd_2.xlsx"
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference file not found: {reference_path}")

    # Если пути не переданы, используем значения по умолчанию
    if index_path is None:
        index_path = FAISS_DIR / "okpd_index.faiss"
    if id_map_path is None:
        id_map_path = FAISS_DIR / "id_map.csv"

    df = pd.read_excel(reference_path, dtype=str)
    df = df.dropna(subset=["name"])
    df["name"] = df["name"].astype(str).str.strip()
    df = df[df["name"] != ""]
    texts = df["name"].tolist()
    print(f"Encoding {len(texts)} texts for Dense index...")

    model = SentenceTransformer(model_name, device=get_device())
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(index_path))
    print(f"FAISS index saved to {index_path}")

    id_map = df[["code", "parent_code", "name"]].copy()
    id_map.to_csv(id_map_path, index=False)
    print(f"ID map saved to {id_map_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="BAAI/bge-m3")
    args = parser.parse_args()
    build_faiss_index(model_name=args.model)