# backend/training/train_frozen_stratified.py
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
import numpy as np
import torch
torch.set_default_device("cpu")
import faiss
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
from config.settings import TRAINING_DATA_DIR, RAW_DATA_DIR, REFERENCE_DIR, FAISS_DIR
from backend.preprocessing.cleaner import TextCleaner
from collections import Counter

gold = pd.read_excel(TRAINING_DATA_DIR / "train.xlsx", dtype=str)
gold = gold[["Номенклатура", "Код ОКПД2"]].dropna()
gold.columns = ["text", "code"]

code_counts = Counter(gold["code"])
sorted_codes = [c for c, _ in code_counts.most_common()]
n_head = int(len(sorted_codes) * 0.2)
n_mid = int(len(sorted_codes) * 0.5)
head_codes = set(sorted_codes[:n_head])
mid_codes = set(sorted_codes[n_head:n_mid])
tail_codes = set(sorted_codes[n_mid:])

def get_group(code):
    if code in head_codes: return "head"
    if code in mid_codes: return "mid"
    return "tail"

gold["group"] = gold["code"].apply(get_group)

train_df, temp_df = train_test_split(gold, test_size=0.2, random_state=42, stratify=gold["group"])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["group"])

train_codes = set(train_df["code"].unique())
val_df = val_df[val_df["code"].isin(train_codes)]
test_df = test_df[test_df["code"].isin(train_codes)]

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
    print(f"  {name}: head={len(df[df['group']=='head'])} mid={len(df[df['group']=='mid'])} tail={len(df[df['group']=='tail'])}")

prom = pd.read_excel(RAW_DATA_DIR / "all_nomenclature.xlsx", dtype=str)
prom = prom[['nomenclature', 'okpd2_code']].dropna()
prom.columns = ["text", "code"]

all_train = pd.concat([train_df[["text", "code"]], prom], ignore_index=True)
food_prefixes = ('01', '02', '03', '10')
all_train = all_train[all_train['code'].str.startswith(food_prefixes, na=False)]

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
for _, row in tqdm(all_train.iterrows(), total=len(all_train), desc="Пары"):
    text = cleaner.clean(row['text'], use_stemmer=True)
    target = code_to_name.get(row['code'].strip())
    if text and target:
        pairs.append((text, target))
print(f"Обучающих пар: {len(pairs)}")

okpd = pd.read_excel(REFERENCE_DIR / "okpd_2.xlsx", dtype=str)
okpd = okpd.dropna(subset=["name"])
okpd["name_stemmed"] = okpd["name"].apply(lambda x: cleaner.clean(x, use_stemmer=True))
candidate_names = okpd["name_stemmed"].tolist()
candidate_codes = okpd["code"].tolist()

def compute_ndcg(model, queries_df):
    cand_emb = model.encode(candidate_names, convert_to_numpy=True, show_progress_bar=False)
    faiss.normalize_L2(cand_emb)
    ndcg_sum = 0.0
    total = 0
    for _, row in queries_df.iterrows():
        q = cleaner.clean(row["text"], use_stemmer=True)
        q_emb = model.encode([q], convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(q_emb)
        scores = np.dot(q_emb, cand_emb.T)[0]
        ranked_idx = np.argsort(scores)[::-1][:10]
        true_code = row["code"]
        for i, idx in enumerate(ranked_idx):
            if candidate_codes[idx] == true_code:
                ndcg_sum += 1.0 / np.log2(i + 2)
                break
        total += 1
    return ndcg_sum / total if total > 0 else 0.0

model = SentenceTransformer("BAAI/bge-m3", device="cpu")
model.to("cpu")

for name, param in model.named_parameters():
    if name.startswith("0.encoder.layer.") or name.startswith("1.encoder.layer."):
        parts = name.split(".")
        layer_num = int(parts[2]) if len(parts) > 2 else -1
        if layer_num < 6:
            param.requires_grad = False

train_examples = [InputExample(texts=[p[0], p[1]]) for p in pairs]
train_dataloader = torch.utils.data.DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = MultipleNegativesRankingLoss(model)

best_ndcg = -1.0
best_epoch = 0
best_model_path = None

for epoch in range(1, 7):
    print(f"\n{'='*50}\n  FROZEN EPOCH {epoch}\n{'='*50}")
    output_path = f"artifacts/models/bge-m3-frozen-stratified-epoch{epoch}"
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1,
        warmup_steps=50 if epoch > 1 else 100,
        output_path=output_path,
        save_best_model=False,
        show_progress_bar=True,
    )
    model = SentenceTransformer(output_path, device="cpu")
    model.to("cpu")
    ndcg = compute_ndcg(model, val_df)
    print(f"  NDCG@10 на Val: {ndcg:.4f}")
    if ndcg > best_ndcg:
        best_ndcg = ndcg
        best_epoch = epoch
        best_model_path = output_path
        print(f"  ✅ Новый лучший NDCG")
    else:
        print(f"  ⚠️ Без улучшения")

print(f"\nЛучшая модель: эпоха {best_epoch}, NDCG@10 Val = {best_ndcg:.4f}")

print("\nФИНАЛЬНАЯ ОЦЕНКА НА TEST...")
best_model = SentenceTransformer(best_model_path, device="cpu")

cand_emb = best_model.encode(candidate_names, convert_to_numpy=True, show_progress_bar=True)
faiss.normalize_L2(cand_emb)
idx = faiss.IndexFlatIP(cand_emb.shape[1])
idx.add(cand_emb)

hits = {1:0, 3:0, 5:0, 10:0}
ndcg_sum = 0.0
total = 0
for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Test"):
    q = cleaner.clean(row["text"], use_stemmer=True)
    q_emb = best_model.encode([q], convert_to_numpy=True, show_progress_bar=False)
    faiss.normalize_L2(q_emb)
    _, indices = idx.search(q_emb, 10)
    retrieved = [candidate_codes[i] for i in indices[0] if 0 <= i < len(candidate_codes)]
    true_code = row["code"]
    for k in hits:
        if true_code in retrieved[:k]:
            hits[k] += 1
    for i, code in enumerate(retrieved[:10]):
        if code == true_code:
            ndcg_sum += 1.0 / np.log2(i + 2)
            break
    total += 1

print("\n" + "="*70)
print(f"ФИНАЛЬНЫЕ МЕТРИКИ НА TEST (стратифицированный split)")
print("="*70)
print(f"Модель: {best_model_path}")
print(f"Recall@1:  {hits[1]/total:.4f}")
print(f"Recall@3:  {hits[3]/total:.4f}")
print(f"Recall@5:  {hits[5]/total:.4f}")
print(f"Recall@10: {hits[10]/total:.4f}")
print(f"NDCG@10:   {ndcg_sum/total:.4f}")