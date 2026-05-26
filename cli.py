# cli.py
import argparse, sys, os
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
from config.settings import REFERENCE_DIR, FAISS_DIR
from backend.preprocessing.cleaner import TextCleaner
from backend.models.retriever import Retriever, build_faiss_index
from backend.models.engine import DecisionEngine

def cmd_predict_text(args):
    cleaner = TextCleaner(abbreviations_path=REFERENCE_DIR / "сокращения.xlsx")
    retriever = Retriever(model_name=args.model)
    engine = DecisionEngine(retriever)  # без классификатора
    result = engine.predict(args.text)
    print(f"Запрос: {args.text}")
    print(f"Режим: {result['mode']}, уверенность: {result['confidence']:.3f}")
    print(f"Предсказанный код: {result['predicted_code']}")
    print(f"Объяснение: {result['explanation']}")
    print("Топ-5 кандидатов:")
    for i, c in enumerate(result["top_candidates"], 1):
        print(f"  {i}. {c['code']} | {c['score']:.4f} | {c['name']}")

def cmd_build_index(args):
    build_faiss_index(model_name=args.model)

def main():
    parser = argparse.ArgumentParser(description="OKPD-2 Classifier V3 CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    pred_text = subparsers.add_parser("predict-text", help="Predict a single product name")
    pred_text.add_argument("text", help="Product name")
    pred_text.add_argument("--model", default="BAAI/bge-m3")
    pred_text.set_defaults(func=cmd_predict_text)
    build_idx = subparsers.add_parser("build-index", help="Build FAISS index")
    build_idx.add_argument("--model", default="BAAI/bge-m3")
    build_idx.set_defaults(func=cmd_build_index)
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()