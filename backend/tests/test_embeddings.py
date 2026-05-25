# test_embeddings.py
from sentence_transformers import SentenceTransformer
import numpy as np

text = "конфеты шоколадные"
base = SentenceTransformer("BAAI/bge-m3", device="cpu")
ft   = SentenceTransformer("../../artifacts/models/bge-m3-finetuned", device="cpu")

emb_base = base.encode([text])
emb_ft   = ft.encode([text])

diff = np.abs(emb_base - emb_ft).max()
print(f"Max difference between base and finetuned: {diff:.6f}")
if diff > 1e-4:
    print("✅ Модели разные – дообучение успешно.")
else:
    print("❌ Модели идентичны – finetuned не загрузилась.")