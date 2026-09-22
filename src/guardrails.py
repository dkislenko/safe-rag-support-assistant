import re
from typing import Any

from langchain_core.documents import Document


FORBIDDEN_PATTERNS = [
    r"парол[ьяей]",
    r"взлом",
    r"взломать",
    r"обойти",
    r"обход",
    r"обойти\s+оплат",
    r"обход\s+оплат",
    r"обойти\s+тариф",
    r"лимит.*тариф",
    r"тариф.*лимит",
    r"обойти.*лимит",
    r"добавить.*сверх.*лимит",
    r"украсть",
    r"получить\s+чуж",
    r"чужие\s+данные",
    r"данные\s+клиент.*без",
    r"без\s+прав\s+доступ",
    r"sql\s*injection",
    r"xss",
    r"эксплойт",
    r"токен\s+доступ",
    r"как\s+стать\s+администратором\s+без",
]


def is_unsafe_query(
    query: str,
) -> tuple[bool, str]:
    lowered = query.lower()

    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, lowered):
            return True, pattern

    return False, ""


def is_price_question(query: str) -> bool:
    lowered = query.lower()

    price_words = [
        "сколько стоит",
        "цена",
        "стоимость",
        "стоит",
        "руб",
        "рублей",
        "₽",
        "доллар",
        "$",
        "помесячно",
        "оплата",
    ]

    return any(
        word in lowered
        for word in price_words
    )


def groundedness_check(
    answer: str,
    documents: list[Document],
    query: str,
) -> dict[str, Any]:
    context = " ".join(
        document.page_content
        for document in documents
    ).lower()

    answer_lower = answer.lower()

    has_sources = "источники:" in answer_lower

    safe_price_refusal = (
        is_price_question(query)
        and "нет информации" in answer_lower
        and "не могу назвать" in answer_lower
    )

    suspicious_markers = [
        "₽",
        "доллар",
        "$",
        "мобильное приложение",
        "ios",
        "android",
        "навсегда бесплатно",
        "гарантированно исправим",
        "точная цена",
        "скидка",
        "16000",
        "16 000",
    ]

    suspicious_hits = [
        marker
        for marker in suspicious_markers
        if marker in answer_lower
        and marker not in context
    ]

    if (
        is_price_question(query)
        and not safe_price_refusal
        and re.search(
            r"\d+\s*(руб|₽|\$|доллар)",
            answer_lower,
        )
    ):
        suspicious_hits.append(
            "точная цена без источника"
        )

    unsafe, unsafe_pattern = is_unsafe_query(query)

    answer_words = set(
        re.findall(
            r"[а-яa-z0-9]{4,}",
            answer_lower,
        )
    )

    context_words = set(
        re.findall(
            r"[а-яa-z0-9]{4,}",
            context,
        )
    )

    overlap_ratio = 0.0

    if answer_words:
        overlap_ratio = (
            len(answer_words & context_words)
            / len(answer_words)
        )

    risk = "низкий"
    reasons: list[str] = []

    if suspicious_hits:
        risk = "высокий"
        reasons.append(
            "ответ содержит подозрительные утверждения, "
            "которых нет в контексте"
        )

    if not has_sources:
        if risk != "высокий":
            risk = "средний"

        reasons.append(
            "ответ не содержит источников"
        )

    if (
        overlap_ratio < 0.2
        and not safe_price_refusal
    ):
        if risk != "высокий":
            risk = "средний"

        reasons.append(
            "низкое пересечение ответа с контекстом"
        )

    if unsafe:
        risk = "высокий"
        reasons.append(
            f"опасный запрос: {unsafe_pattern}"
        )

    if safe_price_refusal:
        risk = "низкий"
        reasons = [
            "вопрос о цене обработан безопасным "
            "отказом без выдумывания стоимости"
        ]

    if not reasons:
        reasons.append(
            "критичных признаков галлюцинации не найдено"
        )

    return {
        "overlap_ratio": round(overlap_ratio, 3),
        "has_sources": has_sources,
        "suspicious_hits": suspicious_hits,
        "hallucination_risk": risk,
        "reasons": reasons,
    }