# head_tail_metrics.py
import pandas as pd
from collections import Counter
from config.settings import TRAINING_DATA_DIR

gold = pd.read_excel(TRAINING_DATA_DIR / "train.xlsx", dtype=str)
gold = gold[["Номенклатура", "Код ОКПД2"]].dropna()
gold.columns = ["text", "true_code"]

code_counts = Counter(gold["true_code"])
total_codes = len(code_counts)
head_codes = set([code for code, _ in code_counts.most_common(int(total_codes * 0.2))])
mid_codes  = set([code for code, _ in code_counts.most_common(int(total_codes * 0.5))]) - head_codes
tail_codes = set(code_counts.keys()) - head_codes - mid_codes

print(f"Head classes: {len(head_codes)}")
print(f"Mid classes:  {len(mid_codes)}")
print(f"Tail classes: {len(tail_codes)}")