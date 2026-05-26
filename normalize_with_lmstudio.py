# normalize_with_lmstudio.py
import sys, json, time, requests, pandas as pd
from tqdm import tqdm

# --- Проверка аргумента ---
if len(sys.argv) < 2:
    print("Укажите режим: gold или prom")
    sys.exit(1)

mode = sys.argv[1].lower()
if mode == "gold":
    INPUT_FILE = "gold_for_llm.csv"
    OUTPUT_FILE = "gold_normalized.csv"
    IS_GOLD = True
elif mode == "prom":
    INPUT_FILE = "prom_for_llm.csv"
    OUTPUT_FILE = "prom_normalized.csv"
    IS_GOLD = False
else:
    print("Неизвестный режим. Допустимо: gold или prom")
    sys.exit(1)

# --- Настройки ---
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
MODEL_NAME = "t-lite-it-1.0"
CONFIDENCE_THRESHOLD = 0.7
DELAY = 0.0

# --- Промпты  ---
SYSTEM_PROMPT = """Ты — эксперт по товарной номенклатуре продуктов питания и упаковки в России.
Твоя задача — превратить сырое название товара из ERP в осмысленное торговое наименование.

КОНТЕКСТ ДОМЕНА:
- битки, зразы, тефтели, поджарка → мясные полуфабрикаты
- голубцы → мясной полуфабрикат с капустой
- скумбрия, минтай, хек, горбуша → рыба
- кальмар → морепродукт (моллюск)
- Эт., Этик., этикетка → самоклеящаяся этикетка
- к.ш.в. → конфеты шоколадные весовые
- лук.кольца → луковые кольца (снек)
- слив. → сливочный
- раст. → растительное масло
- дезодор. → дезодорированное
- пит. → питьевой
- охл. → охлаждённый
- замор. → замороженный
- бренды: MiniFree, Ozera, Яшкино, Актив и др. — сохраняй.

ПРАВИЛА (строго):
1. Раскрой ВСЕ сокращения.
2. Сохрани бренд/торговую марку в начале.
3. Убери цифры, вес, объём, количество штук (40г/5/6, 100г, 12 шт. и т.п.).
4. Если после раскрытия осталось только видовое слово (скумбрия, зразы, голубцы), дополни его родовым понятием из КОНТЕКСТА: «рыба скумбрия», «мясные полуфабрикаты зразы».
5. Если родовое понятие уже есть (например, «масло подсолнечное»), не добавляй лишнего.
6. Не придумывай новые ингредиенты или способы обработки (не пиши «консервированный», «картофель», если их не было).
7. Максимум 10 слов.

Отвечай СТРОГО одной строкой JSON: {"normalized": "нормализованное название", "confidence": 0.9}
Без пояснений. Только JSON."""

USER_GOLD = """Название товара: {TEXT}
Описание категории: {DESCRIPTION}
Подсказка точная, используй её для уточнения.
Выход:"""

USER_PROM = """Название товара: {TEXT}
Описание категории: {DESCRIPTION}
ВНИМАНИЕ: описание может быть неточным, используй как слабую подсказку.
Выход:"""

def call_llm(text, description, is_gold):
    template = USER_GOLD if is_gold else USER_PROM
    user_msg = template.format(TEXT=text.strip(), DESCRIPTION=description.strip())
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg}
        ],
        "temperature": 0.0,
        "max_tokens": 120,
        "stop": ["<|im_end|>"]
    }
    try:
        r = requests.post(LM_STUDIO_URL, json=payload, timeout=30)
        raw = r.json()["choices"][0]["message"]["content"].strip()
        obj = json.loads(raw)
        norm = obj.get("normalized", text)
        conf = float(obj.get("confidence", 0.5))
        return norm, conf, ""
    except Exception as e:
        return text, 0.0, str(e)

def process_file(csv_path, is_gold):
    df = pd.read_csv(csv_path, dtype=str)
    if 'name_raw' not in df.columns:
        print(f"Ошибка: в {csv_path} нет колонки 'name_raw'")
        return
    if 'description' not in df.columns:
        df['description'] = ''
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Золото" if is_gold else "Промка"):
        name = str(row.get("name_raw", "")).strip()
        desc = str(row.get("description", "")).strip()
        if not desc:
            desc = "Продукт питания"
        norm, conf, err = call_llm(name, desc, is_gold)
        results.append({
            "text_original": name,
            "text_normalized": norm,
            "confidence": conf,
            "needs_review": conf < CONFIDENCE_THRESHOLD or bool(err),
            "llm_error": err
        })
        if DELAY > 0:
            time.sleep(DELAY)
    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"Сохранено {len(result_df)} записей в {OUTPUT_FILE}")

# --- Запуск ---
print(f"Обработка файла: {INPUT_FILE} -> {OUTPUT_FILE}")
process_file(INPUT_FILE, IS_GOLD)
print("Готово!")