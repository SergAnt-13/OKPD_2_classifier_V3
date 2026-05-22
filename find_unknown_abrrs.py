# find_unknown_abbrs.py
# Purpose: Найти токены, которые модель "не понимает" (OOD-score низкий)
from pathlib import Path
import pandas as pd
import numpy as np
import re
from collections import Counter
from config.settings import RAW_DATA_DIR

# Загружаем всю номенклатуру
df = pd.read_excel(RAW_DATA_DIR / "all_nomenclature.xlsx", dtype=str)
df.columns = ['nomenclature', 'nomenclature_name', 'nds_rate', 'okpd2_code']

# Собираем все токены длиной 2-10 символов, которые выглядят как сокращения
short_tokens = []
for text in df['nomenclature'].dropna():
    tokens = re.findall(r'\b[а-яёa-z]+\.[а-яёa-z.]+\b|\b[а-яёa-z]{2,5}\b', text.lower())
    short_tokens.extend(tokens)

# Считаем частоты и отбираем топ-100
freq = Counter(short_tokens)
top100 = freq.most_common(100)

print("Топ-100 подозрительных токенов:")
for token, count in top100:
    print(f"  {token}: {count}")

# Сохраняем
pd.DataFrame(top100, columns=['token', 'count']).to_csv(
    RAW_DATA_DIR / "unknown_abbrs.csv", index=False
)
print(f"\nСохранено в {RAW_DATA_DIR / 'unknown_abbrs.csv'}")