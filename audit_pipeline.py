# audit_pipeline.py
# Purpose: Аудит пайплайна с помощью метрик ECI и ECIF.
# Отвечает на вопрос: "Стоит ли дообучать Bi-encoder с hard negatives?"

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy
import pandas as pd
from pathlib import Path

# ============================================================
# 1. ЗАГРУЗКА ДАННЫХ (замени на свои пути)
# ============================================================
# Загружаем 1500 экспертных примеров
expert_df = pd.read_excel("data/uploads/train.xlsx")
expert_texts = expert_df["Номенклатура"].tolist()[:100]  # первые 100 для скорости
expert_codes = expert_df["Код ОКПД2"].tolist()[:100]

# Загружаем 2000 примеров из "промки"
prom_df = pd.read_excel("data/uploads/all_nomenclature.xlsx", nrows=20000)
prom_texts = prom_df["nomenclature"].tolist()

# Загружаем эталонные названия кодов
okpd_df = pd.read_excel("data/reference/okpd_2.xlsx")
code_to_name = dict(zip(okpd_df["code"], okpd_df["name"]))

# ============================================================
# 2. ЗАГЛУШКИ ДЛЯ RETRIEVAL И RERANKER (замени на реальные модели)
# ============================================================
# Для демонстрации используем случайные скоры.
# В реальности здесь должны быть вызовы BM25 и Cross-Encoder.
def get_bm25_candidates(text, top_k=50):
    # Заглушка: возвращает случайные коды и скоры
    codes = np.random.choice(list(code_to_name.keys()), top_k, replace=False)
    scores = np.random.rand(top_k)
    return list(zip(codes, scores))

def get_reranker_candidates(text, top_k=50):
    codes = np.random.choice(list(code_to_name.keys()), top_k, replace=False)
    scores = np.random.rand(top_k)
    return list(zip(codes, scores))

# ============================================================
# 3. ФУНКЦИИ ДЛЯ РАСЧЁТА ECI И ECIF
# ============================================================

def compute_eci(
    positive_code: str,
    hard_negatives: list[tuple[str, float]],
    alpha: float = 0.5,
    tau: float = 0.8,
) -> float:
    """
    Вычисляет ECI (Effective Contrastive Information) для одного позитивного примера.
    alpha: баланс между Hardness и Safety.
    tau: порог для Safety (чем выше, тем строже фильтр ложных негативов).
    """
    if not hard_negatives:
        return 0.0

    # Информационная ёмкость (Information Capacity)
    info_capacity = np.log1p(len(hard_negatives))

    # Различительная эффективность (Discriminative Efficiency)
    scores = [score for _, score in hard_negatives]
    hardness = np.mean(scores)  # средняя "сложность" негативов

    # Безопасность (Safety) — доля негативов, которые не являются ложными (score < tau)
    safe_count = sum(1 for _, score in hard_negatives if score < tau)
    safety = safe_count / len(hard_negatives)

    # Итоговая Discriminative Efficiency
    disc_efficiency = 2 * hardness * safety / (hardness + safety + 1e-8)

    # ECI = Information Capacity * Discriminative Efficiency
    eci = info_capacity * disc_efficiency
    return eci


def compute_ecif(
    positive_code: str,
    negative_candidates: list[tuple[str, float]],
    contamination_threshold: float = 0.95,
) -> float:
    """
    Вычисляет "токсичность" примера.
    Возвращает score выше порога, если пример "токсичен" (много ложных негативов).
    contamination_threshold: скор выше которого негатив считается ложным.
    """
    if not negative_candidates:
        return 0.0

    # Считаем долю негативов с очень высоким скором (похожи на правильный ответ)
    false_negatives = sum(1 for _, score in negative_candidates if score > contamination_threshold)
    return false_negatives / len(negative_candidates)


# ============================================================
# 4. ЗАПУСК АУДИТА
# ============================================================

print("=" * 60)
print("АУДИТ ПАЙПЛАЙНА: ECI и ECIF")
print("=" * 60)

# --- Тест 1: ECI для трёх стратегий ---
print("\n>>> ТЕСТ 1: Оценка качества Hard Negatives (ECI)")
eci_bm25 = []
eci_reranker = []
eci_hybrid = []

for text, true_code in zip(expert_texts, expert_codes):
    # BM25
    bm25_cands = [(c, s) for c, s in get_bm25_candidates(text) if c != true_code]
    eci_bm25.append(compute_eci(true_code, bm25_cands))

    # Reranker
    reranker_cands = [(c, s) for c, s in get_reranker_candidates(text) if c != true_code]
    eci_reranker.append(compute_eci(true_code, reranker_cands))

    # Гибрид (объединение)
    hybrid_cands = list(set(bm25_cands + reranker_cands))
    eci_hybrid.append(compute_eci(true_code, hybrid_cands))

# Средние значения
avg_eci_bm25 = np.mean(eci_bm25)
avg_eci_reranker = np.mean(eci_reranker)
avg_eci_hybrid = np.mean(eci_hybrid)

print(f"  ECI (BM25):            {avg_eci_bm25:.4f}")
print(f"  ECI (Cross-Encoder):   {avg_eci_reranker:.4f}")
print(f"  ECI (Гибрид):          {avg_eci_hybrid:.4f}")

# --- Тест 2: ECIF для "промки" ---
print("\n>>> ТЕСТ 2: Оценка качества данных 'промки' (ECIF)")
toxic_examples = 0

for text in prom_texts:
    candidates = get_reranker_candidates(text)
    toxicity = compute_ecif("", candidates)
    if toxicity > 0.1:  # порог: >10% ложных негативов
        toxic_examples += 1

toxic_ratio = toxic_examples / len(prom_texts)
print(f"  Токсичных примеров:    {toxic_examples} / {len(prom_texts)} ({toxic_ratio:.2%})")

# ============================================================
# 5. ИТОГОВЫЙ ВЕРДИКТ
# ============================================================
print("\n" + "=" * 60)
print("ВЕРДИКТ")
print("=" * 60)

# Условия принятия решения
if avg_eci_hybrid > avg_eci_bm25 * 1.2 and toxic_ratio < 0.1:
    print("✅ Пайплайн ПОДТВЕРЖДЁН.")
    print("   Hard negatives (гибрид) полезны, данные чистые.")
    print("   РЕКОМЕНДАЦИЯ: Запустить полное дообучение Bi-encoder.")
elif avg_eci_hybrid > avg_eci_bm25 * 1.2 and toxic_ratio >= 0.1:
    print("⚠️  Пайплайн ЧАСТИЧНО ПОДТВЕРЖДЁН.")
    print("   Hard negatives полезны, но данные 'промки' требуют очистки.")
    print("   РЕКОМЕНДАЦИЯ: Удалить токсичные примеры, затем дообучить Bi-encoder.")
elif avg_eci_hybrid <= avg_eci_bm25 * 1.2 and toxic_ratio < 0.1:
    print("❌ Пайплайн ОПРОВЕРГНУТ.")
    print("   Hard negatives не дают значимого прироста. Данные чистые.")
    print("   РЕКОМЕНДАЦИЯ: Отказаться от дообучения с hard negatives. Фокусироваться на GLiClass и мета-модели.")
else:
    print("❌ Пайплайн ОПРОВЕРГНУТ.")
    print("   Hard negatives не дают прироста, данные зашумлены.")
    print("   РЕКОМЕНДАЦИЯ: Сфокусироваться на GLiClass и мета-модели, очистить данные 'промки'.")

print("=" * 60)
print("Аудит завершён. На основе вердикта принимаем решение о дальнейших шагах.")