# test_llm_normalization.py
import json, random
from pathlib import Path
import pandas as pd
from llama_cpp import Llama

MODEL_PATH = Path("models/qwen3-4b-q4_k_m.gguf")
CTX_SIZE = 2048
TEMPERATURE = 0.1
TOP_P = 0.9
MAX_TOKENS = 60
STOP = ["\n", "<|im_end|>", "Вход:"]
N_SAMPLES = 30

def load_okpd_context():
    okpd = pd.read_excel("data/reference/okpd_2.xlsx", dtype=str)
    okpd = okpd.dropna(subset=["code", "name"])
    return dict(zip(okpd["code"].str.strip(), okpd["name"].str.strip()))

def main():
    llm = Llama(model_path=str(MODEL_PATH), n_gpu_layers=-1, n_ctx=CTX_SIZE, verbose=False)
    code_to_name = load_okpd_context()

    gold = pd.read_excel("data/training/train.xlsx", dtype=str)
    gold = gold[["Номенклатура", "Код ОКПД2"]].dropna()
    sample = gold.sample(N_SAMPLES, random_state=42)

    system_prompt = (
        "Ты — эксперт по нормализации названий продуктов питания. "
        "Преврати сырое ERP-название в чистое товарное название.\n"
        "Правила:\n"
        "- Раскрой ВСЕ сокращения на русском языке\n"
        "- Сохрани бренд если он есть\n"
        "- Убери артикулы, коды, объём упаковки\n"
        "- Максимум 8 слов\n"
        "- Только название, без пояснений\n"
        "Контекст: код ОКПД-2 и эталонное название помогают понять категорию товара.\n"
        "Отвечай СТРОГО: {\"normalized\": \"название\"}"
    )

    results = []
    for _, row in sample.iterrows():
        raw = row["Номенклатура"]
        code = str(row["Код ОКПД2"]).strip()
        ctx_name = code_to_name.get(code, "")
        prompt = f"{system_prompt}\nВход: {raw} | Код: {code} — {ctx_name}\nВыход:"
        resp = llm(prompt, max_tokens=MAX_TOKENS, temperature=TEMPERATURE, top_p=TOP_P, stop=STOP, echo=False)
        answer = resp["choices"][0]["text"].strip()
        start = answer.find("{")
        end = answer.rfind("}") + 1
        norm = raw
        if start != -1 and end > start:
            try:
                data = json.loads(answer[start:end])
                norm = data.get("normalized", raw)
            except Exception:
                pass
        results.append({"raw": raw, "code": code, "context": ctx_name, "normalized": norm})
        print(f"{raw}  ->  {norm}")

    pd.DataFrame(results).to_csv("data/training/llm_test_30.csv", index=False)
    print("\nРезультаты сохранены в data/training/llm_test_30.csv")

if __name__ == "__main__":
    main()