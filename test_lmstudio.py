import json, requests, pandas as pd

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
MODEL_NAME = "t-lite-it-1.0"
INPUT_FILE = "gold_for_llm.csv"

SYSTEM_PROMPT = """Ты — эксперт по товарной номенклатуре продуктов питания и упаковки в России.
Твоя задача — нормализовать сырое название товара из ERP-системы предприятия.

КОНТЕКСТ ДОМЕНА:
- битки, зразы, тефтели, поджарка — мясные полуфабрикаты
- голубцы — полуфабрикат из мяса с капустой
- скумбрия, минтай, хек — рыба и рыбные полуфабрикаты
- Эт., Этик. — этикетка
- к.ш.в — конфеты шоколадные весовые
- раст. — растительное, дезодор. — дезодорированное, пит. — питьевой
- бренды: Ozera, Яшкино, Актив и др.

ПРАВИЛА:
1. Раскрой ВСЕ сокращения
2. Сохрани бренд если он явно указан
3. Убери артикулы, объём, количество штук
4. Максимум 10 слов
5. Не придумывай новые слова — только раскрывай то, что есть

Отвечай СТРОГО одной строкой JSON: {"normalized": "название", "confidence": 0.9}
Не давай никаких пояснений. Только JSON."""

USER_PROMPT = """Название товара: {TEXT}
Описание категории: {DESCRIPTION}
Подсказка точная, используй её для уточнения.
Выход:"""

# Читаем первые 10 записей
df = pd.read_csv(INPUT_FILE, dtype=str).head(10)

for _, row in df.iterrows():
    name = row["name_raw"]
    desc = row.get("description", "Продукт питания")

    user_msg = USER_PROMPT.format(TEXT=name, DESCRIPTION=desc)

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
        # Пробуем разобрать JSON
        try:
            obj = json.loads(raw)
            norm = obj.get("normalized", name)
            conf = float(obj.get("confidence", 0.5))
        except:
            norm = raw
            conf = 0.0
    except Exception as e:
        norm = name
        conf = 0.0
        raw = str(e)

    print(f"Исходное:      {name}")
    print(f"Нормализация:  {norm}")
    print(f"Уверенность:   {conf:.2f}")
    print("-" * 60)