# cli.py
# Purpose: Developer CLI for OKPD-2 Classifier V3.

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.settings import REFERENCE_DIR, FAISS_DIR
from backend.preprocessing.cleaner import TextCleaner
from backend.preprocessing.stemmer import get_stemmer
from backend.models.retriever import Retriever, build_faiss_index


def cmd_predict_text(args):
    cleaner = TextCleaner(abbreviations_path=REFERENCE_DIR / "сокращения.xlsx")
    stemmer = get_stemmer()
    text = args.text
    text_stemmed = cleaner.clean(text, stemmer=stemmer)

    retriever = Retriever(
        model_name=args.model,
        index_path=FAISS_DIR / "okpd_index.faiss",
        id_map_path=FAISS_DIR / "id_map.csv",
        bm25_path=FAISS_DIR / "bm25_index.pkl",
    )
    result = retriever.search(text_stemmed, top_k=5)

    print(f"Запрос: {text}")
    print(f"После очистки+стемминга: {text_stemmed}\n")
    print("Топ-5 кандидатов:")
    for i, cand in enumerate(result["candidates"], 1):
        print(f"  {i}. {cand['code']} | {cand['score']:.4f} | {cand['name']}")


def cmd_build_index(args):
    build_faiss_index(model_name=args.model)


def main():
    parser = argparse.ArgumentParser(description="OKPD-2 Classifier V3 CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pred_text = subparsers.add_parser("predict-text", help="Predict a single product name")
    pred_text.add_argument("text", help="Product name")
    pred_text.add_argument("--model", default="intfloat/multilingual-e5-large")
    pred_text.set_defaults(func=cmd_predict_text)

    build_idx = subparsers.add_parser("build-index", help="Build FAISS and BM25 index")
    build_idx.add_argument("--model", default="intfloat/multilingual-e5-large")
    build_idx.set_defaults(func=cmd_build_index)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()