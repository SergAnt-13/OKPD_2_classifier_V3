# backend/evaluation/analyze_dataset.py
"""
Анализ золотой выборки (train.xlsx):
- распределение классов (head/mid/tail)
- топ-10 и самые редкие коды
- статистика по текстам (длины, пустые)
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from collections import Counter
from config.settings import TRAINING_DATA_DIR

# Загружаем золотую выборку
gold = pd.read_excel(TRAINING_DATA_DIR / "train.xlsx", dtype=str)
gold = gold[["Номенклатура", "Код ОКПД2"]].dropna()
gold.columns = ["text", "code"]

print(f"Всего примеров: {len(gold)}")
print(f"Уникальных кодов: {gold['code'].nunique()}")

# Распределение классов (head/mid/tail)
code_counts = Counter(gold["code"])
sorted_codes = [c for c, _ in code_counts.most_common()]
total_codes = len(sorted_codes)
head_threshold = int(total_codes * 0.2)
mid_threshold = int(total_codes * 0.5)

head_codes = set(sorted_codes[:head_threshold])
mid_codes = set(sorted_codes[head_threshold:mid_threshold])
tail_codes = set(sorted_codes[mid_threshold:])

head_count = sum(code_counts[c] for c in head_codes)
mid_count = sum(code_counts[c] for c in mid_codes)
tail_count = sum(code_counts[c] for c in tail_codes)

print(f"\n=== РАСПРЕДЕЛЕНИЕ КЛАССОВ (Head/Mid/Tail) ===")
print(f"Head (20% кодов): {len(head_codes)} классов, {head_count} примеров ({head_count/len(gold)*100:.1f}%)")
print(f"Mid (30% кодов):  {len(mid_codes)} классов, {mid_count} примеров ({mid_count/len(gold)*100:.1f}%)")
print(f"Tail (50% кодов): {len(tail_codes)} классов, {tail_count} примеров ({tail_count/len(gold)*100:.1f}%)")

# Топ-10 кодов
print(f"\n=== ТОП-10 КОДОВ ПО ЧАСТОТЕ ===")
for code, count in code_counts.most_common(10):
    print(f"  {code}: {count} примеров")

# Самые редкие коды (по 1 примеру)
rare = [c for c, cnt in code_counts.items() if cnt == 1]
print(f"\n=== САМЫЕ РЕДКИЕ КОДЫ (по 1 примеру): {len(rare)} классов ===")
if rare:
    print("  Примеры:", rare[:10])

# Статистика по текстам
gold["text_len"] = gold["text"].str.len()
print(f"\n=== СТАТИСТИКА ПО ТЕКСТАМ ===")
print(f"Средняя длина: {gold['text_len'].mean():.1f} символов")
print(f"Медианная длина: {gold['text_len'].median():.0f} символов")
print(f"Максимальная длина: {gold['text_len'].max()} символов")
print(f"Пустых текстов: {gold['text'].isna().sum()}")