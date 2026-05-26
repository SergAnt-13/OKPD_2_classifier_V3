# rewrite_llm_files_raw.py
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from config.settings import TRAINING_DATA_DIR, RAW_DATA_DIR, REFERENCE_DIR

# --- Золото ---
gold = pd.read_excel(TRAINING_DATA_DIR / "train.xlsx", dtype=str)
gold = gold[["Номенклатура", "Код ОКПД2"]].dropna()
gold.columns = ["name_raw", "code"]

okpd = pd.read_excel(REFERENCE_DIR / "okpd_2.xlsx", dtype=str)[["code", "name"]]
okpd.columns = ["code", "description"]

result_gold = gold.merge(okpd, on="code", how="left")
result_gold[["name_raw", "description"]].to_csv("gold_for_llm.csv", index=False)
print(f"gold_for_llm.csv: {len(result_gold)} записей (сырые названия)")

# --- Промка ---
prom = pd.read_excel(RAW_DATA_DIR / "all_nomenclature.xlsx", dtype=str)
prom = prom.rename(columns={'nomenclature': 'name_raw'})
if 'okpd2_code' in prom.columns:
    prom = prom.rename(columns={'okpd2_code': 'code'})
else:
    prom['code'] = None

result_prom = prom.merge(okpd, on="code", how="left")
result_prom[["name_raw", "description"]].to_csv("prom_for_llm.csv", index=False)
print(f"prom_for_llm.csv: {len(result_prom)} записей (сырые названия)")