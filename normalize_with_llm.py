# normalize_with_llm.py
import json, os, sys, subprocess
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# ---------- настройки ----------
MODEL_URL = "https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/qwen3-4b-q4_k_m.gguf"
MODEL_PATH = Path("models/qwen3-4b-q4_k_m.gguf")
INPUT_FILES = [
    ("data/training/train.xlsx", "data/training/train_llm_normalized.csv"),
    ("data/uploads/all_nomenclature.xlsx", "data/uploads/all_nomenclature_llm_normalized.csv"),
]
N_GPU_LAYERS = -1          # все слои на GPU
CTX_SIZE = 2048
TEMPERATURE = 0.1
TOP_P = 0.9
MAX_TOKENS = 60
STOP = ["\n", "<|im_end|>", "Вход:"]
# --------------------------------

def ensure_model():
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Скачиваем модель в {MODEL_PATH}...")
        import urllib.request
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH

def load_okpd_context():
    okpd = pd.read_excel("data/reference/okpd_2.xlsx", dtype=str)
    okpd = okpd.dropna(subset=["code", "name"])
    return dict(zip(okpd["code"].str.strip(), okpd["name"].str.strip()))

def main():
    # установка llama-cpp-python с CUDA, если нет
    try:
        from llama_cpp import Llama
    except ImportError:
        print("Устанавливаем llama-cpp-python с CUDA...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "llama-cpp-python",
                               "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cu118"])
        from llama_cpp import Llama

    model_path = ensure_model()
    llm = Llama(model_path=str(model_path), n_gpu_layers=N_GPU_LAYERS, n_ctx=CTX_SIZE, verbose=False)
    code_to_name = load_okpd_context()

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

    for input_path, output_path in INPUT_FILES:
        if not Path(input_path).exists():
            print(f"Файл {input_path} не найден, пропускаем.")
            continue
        df = pd.read_excel(input_path, dtype=str) if input_path.endswith(".xlsx") else pd.read_csv(input_path, dtype=str)
        name_col = next((c for c in df.columns if "номенкл" in c.lower() or "наимен" in c.lower()), df.columns[0])
        code_col = next((c for c in df.columns if "код" in c.lower() or "okpd" in c.lower()), None)

        print(f"Обрабатываем {input_path} ({len(df)} записей)")
        results = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            raw = str(row[name_col])
            code = str(row[code_col]).strip() if code_col and pd.notna(row.get(code_col)) else ""
            ctx_name = code_to_name.get(code, "")
            prompt = (
                f"{system_prompt}\n"
                f"Вход: {raw} | Код: {code} — {ctx_name}\n"
                "Выход:"
            )
            try:
                resp = llm(
                    prompt,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    stop=STOP,
                    echo=False,
                )
                answer = resp["choices"][0]["text"].strip()
                # парсим JSON
                start = answer.find("{")
                end = answer.rfind("}") + 1
                if start != -1 and end > start:
                    data = json.loads(answer[start:end])
                    results.append(data.get("normalized", raw))
                else:
                    results.append(raw)
            except Exception:
                results.append(raw)

        df["normalized_by_llm"] = results
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"Сохранено в {output_path}")

if __name__ == "__main__":
    main()