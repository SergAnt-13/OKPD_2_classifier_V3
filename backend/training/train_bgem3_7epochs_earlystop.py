# backend/training/train_bgem3_7epochs_earlystop.py
"""
Дообучение BGE‑M3 на пищевых классах (до 7 эпох) с early stopping по NDCG@10.
Patience = 3 (остановка после 3 эпох без улучшения NDCG@10).
После обучения — финальная оценка на всей золотой выборке.
"""
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pandas as pd
import torch
torch.set_default_device("cpu")
import faiss
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss

from config.settings import TRAINING_DATA_DIR, RAW_DATA_DIR, REFERENCE_DIR, FAISS_DIR
from backend.preprocessing.cleaner import TextCleaner

# ============================================================
# 1. ДАННЫЕ И ПАРЫ (как в V2)
# ============================================================
gold = pd.read_excel(TRAINING_DATA_DIR / "train.xlsx", dtype=str)
gold = gold[["Номенклатура", "Код ОКПД2"]].dropna()
gold.columns = ["text", "code"]

prom = pd.read_excel(RAW_DATA_DIR / "all_nomenclature.xlsx", dtype=str)
prom = prom[['nomenclature', 'okpd2_code']].dropna()
prom.columns = ["text", "code"]

all_data = pd.concat([gold, prom], ignore_index=True)
food_prefixes = ('01', '02', '03', '10')
all_data = all_data[all_data['code'].str.startswith(food_prefixes, na=False)]
print(f"Пищевых товаров: {len(all_data)}")

# Стеммированный справочник
ref_path = REFERENCE_DIR / "okpd_2_normalized_stemmed.csv"
if ref_path.exists():
    ref = pd.read_csv(ref_path, dtype=str)
else:
    okpd = pd.read_excel(REFERENCE_DIR / "okpd_2.xlsx", dtype=str)
    cleaner_tmp = TextCleaner(abbreviations_path=REFERENCE_DIR / "сокращения.xlsx")
    okpd["name_normalized"] = okpd["name"].apply(lambda x: cleaner_tmp.clean(x, use_stemmer=True))
    ref = okpd[["code", "parent_code", "name", "name_normalized"]]
    ref.to_csv(ref_path, index=False)
code_to_name = dict(zip(ref["code"], ref["name_normalized"]))

cleaner = TextCleaner(abbreviations_path=REFERENCE_DIR / "сокращения.xlsx")
pairs = []
for _, row in tqdm(all_data.iterrows(), total=len(all_data), desc="Пары"):
    text = cleaner.clean(row['text'], use_stemmer=True)
    target = code_to_name.get(row['code'].strip())
    if text and target:
        pairs.append((text, target))
print(f"Обучающих пар: {len(pairs)}")

train_pairs, eval_pairs = train_test_split(pairs, test_size=0.2, random_state=42)
train_examples = [InputExample(texts=[p[0], p[1]]) for p in train_pairs]

# Для мониторинга NDCG соберём уникальные коды eval‑выборки и их названия
eval_codes = set(p[1] for p in eval_pairs)  # это уже нормализованные названия (ключи)
# Но нам нужны исходные коды для оценки на золотой выборке позже, а здесь для NDCG достаточно названий
eval_queries = [p[0] for p in eval_pairs]
eval_true_codes = [p[1] for p in eval_pairs]  # стеммированные названия

# ============================================================
# 2. ФУНКЦИЯ ДЛЯ ВЫЧИСЛЕНИЯ NDCG@10 НА EVAL ВЫБОРКЕ
# ============================================================
def compute_ndcg(model, queries, true_codes, all_candidate_names):
    """
    Для каждого запроса кодируем его, считаем косинусное сходство со всеми candidate_names,
    ранжируем и вычисляем NDCG@10 (бинарная релевантность).
    """
    # Кодируем все кандидаты один раз
    cand_emb = model.encode(all_candidate_names, convert_to_numpy=True, show_progress_bar=False)
    faiss.normalize_L2(cand_emb)
    ndcg_sum = 0.0
    total = 0
    for q, true_name in zip(queries, true_codes):
        q_emb = model.encode([q], convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(q_emb)
        scores = np.dot(q_emb, cand_emb.T)[0]  # cosine similarities
        # Ранжируем индексы по убыванию сходства
        ranked_idx = np.argsort(scores)[::-1][:10]
        # Вычисляем NDCG@10
        for i, idx in enumerate(ranked_idx):
            if all_candidate_names[idx] == true_name:
                ndcg_sum += 1.0 / np.log2(i + 2)  # i+1 - ранг (1-indexed), i - позиция (0-indexed)
                break
        total += 1
    return ndcg_sum / total if total > 0 else 0.0

# Список всех уникальных названий из eval (кандидаты)
eval_candidate_names = list(eval_codes)
print(f"Уникальных кодов в eval: {len(eval_candidate_names)}")

# ============================================================
# 3. ОБУЧЕНИЕ С EARLY STOPPING ПО NDCG
# ============================================================
model = SentenceTransformer("BAAI/bge-m3", device="cpu")
model.to("cpu")

train_dataloader = torch.utils.data.DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = MultipleNegativesRankingLoss(model)

best_ndcg = -1.0
best_epoch = 0
best_model_path = None
no_improve_count = 0
patience = 3
max_epochs = 7

for epoch in range(1, max_epochs + 1):
    print(f"\n{'='*50}\n  ЭПОХА {epoch}\n{'='*50}")
    output_path = f"artifacts/models/bge-m3-food-only-epoch{epoch}"
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=None,  # мы сами оценим NDCG
        epochs=1,
        warmup_steps=50 if epoch > 1 else 100,
        output_path=output_path,
        save_best_model=False,
        show_progress_bar=True,
    )
    # Загружаем модель после эпохи
    model = SentenceTransformer(output_path, device="cpu")
    model.to("cpu")

    # Считаем NDCG на eval
    ndcg = compute_ndcg(model, eval_queries, eval_true_codes, eval_candidate_names)
    print(f"  NDCG@10 на eval: {ndcg:.4f}")

    # Ранний стоп
    if ndcg > best_ndcg:
        best_ndcg = ndcg
        best_epoch = epoch
        best_model_path = output_path
        no_improve_count = 0
        print(f"  ✅ Новый лучший NDCG, модель сохранена в {best_model_path}")
    else:
        no_improve_count += 1
        print(f"  ⚠️ Без улучшения {no_improve_count}/{patience}")
        if no_improve_count >= patience:
            print(f"\n🛑 Ранняя остановка после {patience} эпох без улучшения NDCG.")
            break

print(f"\nЛучшая модель: эпоха {best_epoch}, NDCG@10 = {best_ndcg:.4f}")

# ============================================================
# 4. ФИНАЛЬНАЯ ОЦЕНКА НА ВСЕЙ ЗОЛОТОЙ ВЫБОРКЕ
# ============================================================
print("\nСтроим стеммированный FAISS‑индекс для лучшей модели...")
best_model = SentenceTransformer(best_model_path, device="cpu")

okpd = pd.read_excel(REFERENCE_DIR / "okpd_2.xlsx", dtype=str)
okpd = okpd.dropna(subset=["name"])
okpd["name"] = okpd["name"].astype(str).str.strip()
okpd["name_stemmed"] = okpd["name"].apply(lambda x: cleaner.clean(x, use_stemmer=True))

emb = best_model.encode(okpd["name_stemmed"].tolist(), batch_size=8,
                        convert_to_numpy=True, show_progress_bar=True)
faiss.normalize_L2(emb)
idx = faiss.IndexFlatIP(emb.shape[1])
idx.add(emb)
idx_path = FAISS_DIR / "okpd_index_final.faiss"
id_map_path = FAISS_DIR / "id_map_final.csv"
faiss.write_index(idx, str(idx_path))
okpd[["code", "parent_code", "name"]].to_csv(id_map_path, index=False)

print("\nОценка на всей золотой выборке (1475 примеров)...")
idx = faiss.read_index(str(idx_path))
id_map = pd.read_csv(id_map_path, dtype=str)
codes = id_map["code"].values

hits = {1:0, 3:0, 5:0, 10:0}
ndcg_sum = 0.0
total = 0
for _, row in tqdm(gold.iterrows(), total=len(gold), desc="Оценка"):
    q = cleaner.clean(row["text"], use_stemmer=True)
    emb_q = best_model.encode([q], convert_to_numpy=True, show_progress_bar=False)
    faiss.normalize_L2(emb_q)
    _, indices = idx.search(emb_q, 10)
    retrieved = [codes[i] for i in indices[0] if 0 <= i < len(codes)]
    true_code = row["code"]
    for k in hits:
        if true_code in retrieved[:k]:
            hits[k] += 1
    # NDCG@10
    for i, code in enumerate(retrieved[:10]):
        if code == true_code:
            ndcg_sum += 1.0 / np.log2(i + 2)
            break
    total += 1

print("\n" + "="*60)
print("ФИНАЛЬНЫЕ МЕТРИКИ (вся золотая выборка, 1475 примеров)")
print("="*60)
print(f"Модель: {best_model_path}")
print(f"Recall@1:  {hits[1]/total:.4f}")
print(f"Recall@3:  {hits[3]/total:.4f}")
print(f"Recall@5:  {hits[5]/total:.4f}")
print(f"Recall@10: {hits[10]/total:.4f}")
print(f"NDCG@10:   {ndcg_sum/total:.4f}")