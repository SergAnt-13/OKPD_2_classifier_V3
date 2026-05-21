# backend/preprocessing/cleaner.py
# Purpose: Text preprocessing — abbreviation expansion, normalization, optional stemming.

import re
from pathlib import Path
from typing import Optional, Dict

import pandas as pd


class TextCleaner:
    """
    Preprocesses product names:
    - expands abbreviations
    - removes GOST/TU/STO patterns
    - lowercases
    - removes punctuation
    - removes leading digits
    - normalizes whitespace
    """

    GOST_PATTERNS = [
        re.compile(r"\bгост\b\s*[\d\-]+", re.IGNORECASE),
        re.compile(r"\bту\b\s*[\d\-]+", re.IGNORECASE),
        re.compile(r"\bсто\b\s*[\d\-]+", re.IGNORECASE),
    ]

    def __init__(self, abbreviations_path: Optional[Path] = None):
        self.abbreviations: Dict[str, str] = {}
        if abbreviations_path and Path(abbreviations_path).exists():
            df = pd.read_excel(abbreviations_path, dtype=str)
            if "abbr" in df.columns and "expansion" in df.columns:
                self.abbreviations = dict(
                    zip(
                        df["abbr"].str.lower().str.strip().str.rstrip("."),
                        df["expansion"].str.strip(),
                    )
                )

    @staticmethod
    def remove_gost(text: str) -> str:
        for pattern in TextCleaner.GOST_PATTERNS:
            text = pattern.sub(" ", text)
        return text

    @staticmethod
    def normalise_punctuation(text: str) -> str:
        text = re.sub(r"[^а-яёa-z0-9\s]", " ", text, flags=re.IGNORECASE)
        return text

    def apply_abbreviations(self, text: str) -> str:
        if not self.abbreviations:
            return text
        tokens = re.findall(r"\b\w+(?:\.\w+)+\b|\b\w+\b|[^\w\s]", text)
        result = []
        for token in tokens:
            token_lower = token.lower().rstrip(".")
            if token_lower in self.abbreviations:
                result.append(self.abbreviations[token_lower])
            else:
                result.append(token)
        return " ".join(result)

    def clean(self, text: Optional[str], stemmer=None) -> str:
        """
        Main cleaning pipeline.
        If stemmer is provided, applies stemming (for retrieval / training).
        Otherwise returns full forms (for NER, cross-encoder, UI output).
        """
        if not text:
            return ""
        text = str(text).strip().lower()
        if not text:
            return ""

        text = self.apply_abbreviations(text)
        text = self.remove_gost(text)
        text = self.normalise_punctuation(text)
        text = re.sub(r"^\d+\s+", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        if stemmer:
            text = " ".join(stemmer.stem(w) for w in text.split())

        return text