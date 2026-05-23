# find_unknown_abbrs.py
# Purpose: Найти подозрительные короткие токены в читаемых названиях товаров.
import pandas as pd
import re
from collections import Counter
from config.settings import RAW_DATA_DIR

df = pd.read_excel(RAW_DATA_DIR / "all_nomenclature.xlsx", dtype=str)

# Используем колонку с читаемым названием (в ней опечатка — nomnclature_name)
name_col = "nomnclature_name" if "nomnclature_name" in df.columns else df.columns[1]
texts = df[name_col].dropna().astype(str).tolist()

# Сокращения с точками: к.ш.в, Этик.самокл. и т.д.
abbr_pattern = re.compile(r'\b[а-яёa-z]+(?:\.[а-яёa-z]+)+\b', re.IGNORECASE)
# Короткие слова 2-4 буквы, без цифр
short_pattern = re.compile(r'\b[а-яёa-z]{2,4}\b', re.IGNORECASE)

abbrs = []
shorts = []

for text in texts:
    abbrs.extend(abbr_pattern.findall(text.lower()))
    shorts.extend(short_pattern.findall(text.lower()))

abbr_freq = Counter(abbrs).most_common(50)
short_freq = Counter(shorts).most_common(50)

print("Топ-50 сокращений с точками:")
for token, cnt in abbr_freq:
    print(f"  {token}: {cnt}")

print("\nТоп-50 коротких слов:")
for token, cnt in short_freq:
    print(f"  {token}: {cnt}")

pd.DataFrame(abbr_freq, columns=["token", "count"]).to_csv(
    RAW_DATA_DIR / "abbrs_with_dots.csv", index=False
)
pd.DataFrame(short_freq, columns=["token", "count"]).to_csv(
    RAW_DATA_DIR / "short_words.csv", index=False
)
print(f"\nСохранено: {RAW_DATA_DIR / 'abbrs_with_dots.csv'}, {RAW_DATA_DIR / 'short_words.csv'}")