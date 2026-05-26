# backend/models/engine.py
from typing import Dict, List, Optional
import re, json
import numpy as np
import torch
from pathlib import Path

from backend.models.retriever import Retriever
from backend.models.reranker import Reranker

PACKAGING_PATTERNS = [
    re.compile(r'\bэтикетк[а-яё]*\.?\b', re.IGNORECASE),
    re.compile(r'\bэтик\.?\b', re.IGNORECASE),
    re.compile(r'\bярлык[а-яё]*\.?\b', re.IGNORECASE),
    re.compile(r'\bупаковк[а-яё]*\.?\b', re.IGNORECASE),
    re.compile(r'\bупак\.?\b', re.IGNORECASE),
    re.compile(r'\bуп\.?\b', re.IGNORECASE),
    re.compile(r'\bтар[а-яё]*\.?\b', re.IGNORECASE),
    re.compile(r'\bпакет[а-яё]*\.?\b', re.IGNORECASE),
    re.compile(r'\bмеш[а-яё]*\.?\b', re.IGNORECASE),
    re.compile(r'\bкороб[а-яё]*\.?\b', re.IGNORECASE),
    re.compile(r'\bгофрокороб[а-яё]*\.?\b', re.IGNORECASE),
    re.compile(r'\bлот[а-яё]*\.?\b', re.IGNORECASE),
    re.compile(r'\bстикер[а-яё]*\.?\b', re.IGNORECASE),
    re.compile(r'\bпленк[а-яё]*\.?\b', re.IGNORECASE),
    re.compile(r'\bплёнк[а-яё]*\.?\b', re.IGNORECASE),
    re.compile(r'\bэт\.?\b', re.IGNORECASE),
]
PACKAGING_CODES = {
    '17.21.12.110', '17.21.13.000', '17.21.14.110', '17.21.14.120',
    '17.21.15.000', '17.29.11.110', '17.29.19.160',
    '24.42.25.130', '17.29.19.190',
    '22.22.13.000', '22.22.19.000',
}

class DecisionEngine:
    def __init__(self, retriever: Retriever, reranker: Optional[Reranker] = None,
                 classifier_path: Optional[Path] = None,
                 auto_threshold: float = 0.7, review_threshold: float = 0.4,
                 margin_threshold: float = 0.2, ood_threshold: float = 0.3,
                 classifier_conf_threshold: float = 0.6,
                 classifier_max_entropy: float = 3.0):
        self.retriever = retriever
        self.reranker = reranker
        self.auto_threshold = auto_threshold
        self.review_threshold = review_threshold
        self.margin_threshold = margin_threshold
        self.ood_threshold = ood_threshold
        self.classifier_conf_threshold = classifier_conf_threshold
        self.classifier_max_entropy = classifier_max_entropy
        self.classifier = None
        self.classifier_tokenizer = None
        self.classifier_labels = None
        if classifier_path and Path(classifier_path).exists():
            self.classifier_path = Path(classifier_path)
        else:
            self.classifier_path = None

    def _load_classifier(self):
        if self.classifier is not None or self.classifier_path is None:
            return
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.classifier_tokenizer = AutoTokenizer.from_pretrained(self.classifier_path)
        self.classifier = AutoModelForSequenceClassification.from_pretrained(self.classifier_path)
        self.classifier.eval()
        with open(self.classifier_path / "label_encoder.json", "r") as f:
            self.classifier_labels = json.load(f)
        if torch.cuda.is_available():
            self.classifier.to("cuda")

    def _has_packaging_marker(self, query: str) -> bool:
        return any(p.search(query) for p in PACKAGING_PATTERNS)

    def _detect_packaging_risk(self, query: str, predicted_code: str) -> bool:
        if not self._has_packaging_marker(query):
            return False
        if predicted_code in PACKAGING_CODES:
            return False
        return True

    def predict(self, query: str, top_k: int = 10, use_reranker: bool = False) -> Dict:
        # 1. retrieval
        if use_reranker and self.reranker is not None:
            raw = self.retriever.search(query, top_k=top_k, use_reranker=False)
            candidates = raw["candidates"]
            pairs = [(c["name"], query) for c in candidates]
            scores = self.reranker.predict(pairs, show_progress_bar=False)
            for c, s in zip(candidates, scores):
                c["rerank_score"] = float(s)
            candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        else:
            raw = self.retriever.search(query, top_k=top_k, use_reranker=False)
            candidates = raw["candidates"]
            candidates.sort(key=lambda x: x["score"], reverse=True)

        if not candidates:
            return {"predicted_code": None, "confidence": 0.0, "mode": "MANUAL",
                    "top_candidates": [], "explanation": "Нет кандидатов",
                    "is_ood": True, "packaging_risk": False}

        max_score = max(c.get("rerank_score", c["score"]) for c in candidates)
        if max_score < self.ood_threshold:
            return {"predicted_code": None, "confidence": 0.0, "mode": "MANUAL",
                    "top_candidates": candidates[:5],
                    "explanation": "Запрос вне распределения (OOD).",
                    "is_ood": True, "packaging_risk": False}

        top1 = candidates[0]
        top1_score = top1.get("rerank_score", top1["score"])
        top2_score = candidates[1].get("rerank_score", candidates[1]["score"]) if len(candidates) > 1 else 0.0
        margin = top1_score - top2_score

        # 2. Классификатор (только если код в его списке)
        classifier_prob = None
        classifier_code = None
        if self.classifier_path:
            self._load_classifier()
            if self.classifier and self.classifier_labels and top1["code"] in self.classifier_labels:
                inputs = self.classifier_tokenizer(query, return_tensors="pt", truncation=True, max_length=256)
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                with torch.no_grad():
                    logits = self.classifier(**inputs).logits
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                entropy = -np.sum(probs * np.log(probs + 1e-9))
                if entropy <= self.classifier_max_entropy:
                    idx = np.argmax(probs)
                    classifier_prob = probs[idx]
                    classifier_code = self.classifier_labels[idx]

        # 3. Режим
        mode = "MANUAL"
        confidence = top1_score
        packaging_risk = self._detect_packaging_risk(query, top1["code"])

        if top1_score >= self.auto_threshold and margin >= self.margin_threshold:
            mode = "AUTO"
            if packaging_risk:
                mode = "REVIEW"
                confidence *= 0.9
        elif top1_score >= self.review_threshold:
            mode = "REVIEW"
            if packaging_risk:
                confidence *= 0.85
        else:
            if packaging_risk:
                confidence *= 0.8

        # 4. Корректировка классификатором
        classifier_comment = ""
        if classifier_prob is not None and classifier_code is not None:
            if classifier_code == top1["code"]:
                confidence = max(confidence, classifier_prob)
                if classifier_prob >= self.classifier_conf_threshold and mode == "REVIEW":
                    mode = "AUTO"
                    classifier_comment = f"BERTA подтверждает (p={classifier_prob:.3f})"
                else:
                    classifier_comment = f"BERTA согласна (p={classifier_prob:.3f})"
            else:
                if classifier_prob >= self.classifier_conf_threshold:
                    mode = "REVIEW"
                    confidence = max(0.0, confidence - 0.2)
                    classifier_comment = f"BERTA предлагает {classifier_code} (p={classifier_prob:.3f})"
                else:
                    classifier_comment = f"BERTA неуверена (p={classifier_prob:.3f})"

        explanation = self._build_explanation(mode, top1_score, margin, packaging_risk, classifier_comment)

        return {
            "predicted_code": top1["code"],
            "confidence": round(confidence, 4),
            "mode": mode,
            "top_candidates": [{"code": c["code"], "score": c.get("rerank_score", c["score"]), "name": c["name"]} for c in candidates[:5]],
            "explanation": explanation,
            "is_ood": False,
            "packaging_risk": packaging_risk,
            "classifier_code": classifier_code,
            "classifier_prob": classifier_prob,
        }

    def _build_explanation(self, mode, score, margin, packaging, classifier_comment):
        parts = []
        if mode == "AUTO": parts.append(f"высокий скор ({score:.3f})")
        elif mode == "REVIEW": parts.append(f"умеренный скор ({score:.3f})")
        else: parts.append(f"низкий скор ({score:.3f}) – экспертиза")
        if margin < self.margin_threshold and mode != "AUTO":
            parts.append("малый отрыв от второго")
        if packaging:
            parts.append("признаки упаковки/этикетки")
        if classifier_comment:
            parts.append(classifier_comment)
        return "; ".join(parts) + "."