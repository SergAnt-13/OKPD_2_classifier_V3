# backend/preprocessing/cleaner.py
import re
from pathlib import Path
from typing import Optional, Dict
import pandas as pd
from nltk.stem.snowball import SnowballStemmer

# Ленивая инициализация Natasha – чтобы не замедлять импорт всего модуля
_SEGMENTER = None
_MORPH_TAGGER = None
_VOCAB = None

def _init_natasha():
    """Отложенная загрузка компонентов Natasha (Segmenter, NewsMorphTagger, MorphVocab)."""
    global _SEGMENTER, _MORPH_TAGGER, _VOCAB
    if _SEGMENTER is None:
        from natasha import Doc, Segmenter, NewsEmbedding, NewsMorphTagger, MorphVocab
        _SEGMENTER = Segmenter()
        emb = NewsEmbedding()
        _MORPH_TAGGER = NewsMorphTagger(emb)
        _VOCAB = MorphVocab()

def lemmatize_text(text: str) -> str:
    """Лемматизация строки с помощью Natasha (нейросетевая + словарная)."""
    if not text:
        return ""
    _init_natasha()
    from natasha import Doc
    doc = Doc(text)
    doc.segment(_SEGMENTER)
    doc.tag_morph(_MORPH_TAGGER)
    for token in doc.tokens:
        token.lemmatize(_VOCAB)
    return " ".join(token.lemma for token in doc.tokens)


class TextCleaner:
    GOST_PATTERNS = [
        re.compile(r"\bгост\b\s*[\d\-]+", re.IGNORECASE),
        re.compile(r"\bту\b\s*[\d\-]+", re.IGNORECASE),
        re.compile(r"\bсто\b\s*[\d\-]+", re.IGNORECASE),
    ]

    def __init__(self, abbreviations_path: Optional[Path] = None):
        self.abbreviations: Dict[str, str] = {}
        # Стеммер остаётся для обратной совместимости
        self.stemmer = SnowballStemmer("russian")
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

    def clean(
        self,
        text: Optional[str],
        use_stemmer: bool = False,
        use_lemmatizer: bool = False
    ) -> str:
        """
        Основной конвейер очистки.

        Параметры:
        - use_stemmer: применить стемминг Snowball (старый режим).
        - use_lemmatizer: применить лемматизацию Natasha (новый режим).
          Если оба флага True, приоритет у лемматизации.

        Рекомендация:
        - для retrieval / FAISS-индекса использовать use_lemmatizer=True.
        - для кросс-энкодера, UI, NER – use_stemmer=False, use_lemmatizer=False.
        """
        if not text:
            return ""
        text = str(text).strip().lower()
        if not text:
            return ""

        # 1. Раскрытие сокращений
        text = self.apply_abbreviations(text)
        # 2. Удаление ГОСТ/ТУ/СТО
        text = self.remove_gost(text)
        # 3. Очистка пунктуации
        text = self.normalise_punctuation(text)
        # 4. Убираем ведущие числа (артикулы)
        text = re.sub(r"^\d+\s+", "", text)
        # 5. Нормализация пробелов
        text = re.sub(r"\s+", " ", text).strip()

        # 6. Стемминг или лемматизация
        if use_lemmatizer:
            text = lemmatize_text(text)
        elif use_stemmer:
            text = " ".join(self.stemmer.stem(w) for w in text.split())

        return text