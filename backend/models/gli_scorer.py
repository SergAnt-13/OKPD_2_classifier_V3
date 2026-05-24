backend/models/gli_scorer.py
# Purpose: GLiClass zero-shot scorer for independent semantic evaluation.
# Uses knowledgator/gliclass-base-v3.0 (DeBERTa-v3-base, 187M params).

from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from transformers import AutoTokenizer
import torch


class GLiScorer:
    def __init__(
            self,
            model_name: str = "knowledgator/gliclass-base-v3.0",
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
            max_length: int = 512,
    ):
        self.model = GLiClassModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pipeline = ZeroShotClassificationPipeline(
            self.model,
            self.tokenizer,
            classification_type='multi-label',
            device=device,
            max_length=max_length,
        )

    def score(self, text: str, candidates: list[dict], threshold: float = 0.0) -> list[dict]:
        """
        Args:
            text: cleaned product name (without stemming).
            candidates: list of {'code': str, 'name': str, ...}.
            threshold: minimum score to include.
        Returns:
            candidates with added 'gli_score' field, sorted by gli_score desc.
        """
        labels = [cand["name"] for cand in candidates]
        results = self.pipeline(text, labels, threshold=threshold)[0]

        # Build score map
        score_map = {r["label"]: r["score"] for r in results}
        for cand in candidates:
            cand["gli_score"] = score_map.get(cand["name"], 0.0)

        candidates.sort(key=lambda x: x.get("gli_score", 0), reverse=True)
        return candidates