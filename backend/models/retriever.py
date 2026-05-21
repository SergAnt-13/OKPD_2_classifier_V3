# backend/models/retriever.py
# Purpose: Hybrid retrieval (Dense + Sparse) with RRF fusion.

import faiss
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import torch
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

from config.settings import FAISS_DIR, REFERENCE_DIR

def get_device() -> str:
    """Auto-detect best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

class BM25Retriever:
    """Sparse retrieval using BM25."""
    def __init__(self, corpus: List[str]):
        tokenized_corpus = [word_tokenize(doc.lower()) for doc in corpus]
        self.model = BM25Okapi(tokenized_corpus)
        self.corpus = corpus

    def search(self, query: str, top_k: int = 100) -> List[tuple]:
        """Return list of (index, score) sorted by score descending."""
        tokenized_query = word_tokenize(query.lower())
        scores = self.model.get_scores(tokenized_query)
        # Get indices of top_k scores
        idx = np.argsort(scores)[::-1][:top_k]
        return [(i, scores[i]) for i in idx]

    @classmethod
    def from_pickle(cls, path: Path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls.__new__(cls)
        obj.model = data["model"]
        obj.corpus = data["corpus"]
        return obj

    def to_pickle(self, path: Path):
        data = {"model": self.model, "corpus": self.corpus}
        with open(path, "wb") as f:
            pickle.dump(data, f)

class Retriever:
    """Hybrid retriever combining Dense (FAISS) and Sparse (BM25)."""

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large",
        index_path: Optional[Path] = None,
        id_map_path: Optional[Path] = None,
        bm25_path: Optional[Path] = None,
    ):
        self.model = SentenceTransformer(model_name, device=get_device())
        self.index_path = index_path or FAISS_DIR / "okpd_index.faiss"
        self.id_map_path = id_map_path or FAISS_DIR / "id_map.csv"
        self.bm25_path = bm25_path or FAISS_DIR / "bm25_index.pkl"
        self._loaded = False
        self.index = None
        self.codes = None
        self.parent_codes = None
        self.names = None
        self.bm25 = None

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
        if self.bm25_path.exists():
            self.bm25 = BM25Retriever.from_pickle(self.bm25_path)
        self._loaded = True

    def _dense_search(self, query: str, top_k: int = 100) -> Dict:
        embedding = self.model.encode([query], convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(embedding)
        scores, indices = self.index.search(embedding, top_k)
        candidates = {}
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self.codes):
                candidates[str(idx)] = {
                    "code": self.codes[idx],
                    "parent_code": self.parent_codes[idx],
                    "score": float(score),
                    "name": self.names[idx],
                    "dense_score": float(score),
                }
        return candidates

    def _sparse_search(self, query: str, top_k: int = 100) -> Dict:
        if self.bm25 is None:
            return {}
        results = self.bm25.search(query, top_k)
        candidates = {}
        for idx, score in results:
            candidates[str(idx)] = {
                "code": self.codes[idx],
                "parent_code": self.parent_codes[idx],
                "score": float(score),
                "name": self.names[idx],
                "sparse_score": float(score),
            }
        return candidates

    def _rrf_fusion(self, dense_candidates: Dict, sparse_candidates: Dict, k: int = 60, top_k: int = 5) -> List[Dict]:
        """Reciprocal Rank Fusion."""
        rrf_scores = {}
        # Assign rank to dense candidates (1-indexed)
        dense_sorted = sorted(dense_candidates.values(), key=lambda x: x["dense_score"], reverse=True)
        for rank, cand in enumerate(dense_sorted, 1):
            idx = cand["code"]  # use code as unique key? better to use original index.
            # We'll use code as identifier
            code = cand["code"]
            rrf_scores[code] = rrf_scores.get(code, 0) + 1 / (k + rank)

        sparse_sorted = sorted(sparse_candidates.values(), key=lambda x: x["sparse_score"], reverse=True)
        for rank, cand in enumerate(sparse_sorted, 1):
            code = cand["code"]
            rrf_scores[code] = rrf_scores.get(code, 0) + 1 / (k + rank)

        # Sort by RRF score
        sorted_codes = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        # Build candidate dict with combined info
        combined = []
        for code, rrf_score in sorted_codes[:top_k]:
            # Find the candidate from dense or sparse
            cand = None
            for c in dense_candidates.values():
                if c["code"] == code:
                    cand = c
                    break
            if cand is None:
                for c in sparse_candidates.values():
                    if c["code"] == code:
                        cand = c
                        break
            if cand:
                cand["rrf_score"] = rrf_score
                combined.append(cand)
        return combined

    def search(self, query: str, top_k: int = 5) -> Dict:
        self._lazy_load()
        dense = self._dense_search(query, top_k=100)
        sparse = self._sparse_search(query, top_k=100)
        candidates = self._rrf_fusion(dense, sparse, top_k=top_k)
        return {"candidates": candidates}

def build_faiss_index(
    reference_path: Optional[Path] = None,
    model_name: str = "intfloat/multilingual-e5-large",
    batch_size: int = 32,
):
    reference_path = reference_path or REFERENCE_DIR / "okpd_2.xlsx"
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference file not found: {reference_path}")

    df = pd.read_excel(reference_path, dtype=str)
    df = df.dropna(subset=["name"])
    df["name"] = df["name"].astype(str).str.strip()
    df = df[df["name"] != ""]
    texts = df["name"].tolist()
    print(f"Encoding {len(texts)} texts for Dense index...")

    model = SentenceTransformer(model_name, device=get_device())
    embeddings = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True)
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    index_path = FAISS_DIR / "okpd_index.faiss"
    faiss.write_index(index, str(index_path))
    print(f"FAISS index saved to {index_path}")

    # Build and save BM25 index
    print("Building BM25 index...")
    bm25 = BM25Retriever(texts)
    bm25_path = FAISS_DIR / "bm25_index.pkl"
    bm25.to_pickle(bm25_path)
    print(f"BM25 index saved to {bm25_path}")

    id_map = df[["code", "parent_code", "name"]].copy()
    id_map_path = FAISS_DIR / "id_map.csv"
    id_map.to_csv(id_map_path, index=False)
    print(f"ID map saved to {id_map_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="intfloat/multilingual-e5-large")
    args = parser.parse_args()
    build_faiss_index(model_name=args.model)