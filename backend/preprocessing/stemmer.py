# backend/preprocessing/stemmer.py
# Purpose: Thin wrapper around SnowballStemmer for consistent imports.

from nltk.stem.snowball import SnowballStemmer

def get_stemmer() -> SnowballStemmer:
    return SnowballStemmer("russian")