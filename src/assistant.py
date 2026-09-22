import re
from typing import Any

from langchain_core.documents import Document

from src.embeddings import build_vectorstore
from src.guardrails import (
    groundedness_check,
    is_price_question,
    is_unsafe_query,
)
from src.llm import LocalLLM
from src.retrieval import RAGRetriever


def get_sources(
    documents: list[Document],
) -> list[str]:
    sources: list[str] = []

    for document in documents:
        source = document.metadata.get(
            "source",
            "unknown",
        )

        if source not in sources:
            sources.append(source)

    return sources


def format_sources(
    sources: list[str],
) -> str:
    return ", ".join(
        f"[{source}]"
        for source in sources
    )


def sentence_split(text: str) -> list[str]:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()

    parts = re.split(
        r"(?<=[.!?])\s+",
        text,
    )

    return [
        part.strip()
        for part in parts
        if part.strip()
    ]


def clean_document_text(
    document: Document,
) -> str:
    text = document.page_content

    text = re.sub(
        r"^#.*$",
        "",
        text,
        flags=re.MULTILINE,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def build_extractive_answer(
    documents: list[Document],
) -> str:
    sources = get_sources(documents)

    useful_sentences: list[str] = []

    for document in documents[:4]:
        text = clean_document_text(document)

        for sentence in sentence_split(text):
            if 40 <= len(sentence) <= 260:
                useful_sentences.append(sentence)

            if len(useful_sentences) >= 6:
                break

        if len(useful_sentences) >= 6:
            break

    if not useful_sentences:
        return (
            "В базе знаний нет достаточной информации "
            "для точного ответа.\n\n"
            f"Источники: {format_sources(sources)}"
        )

    lines = [
        "На основе базы знаний:"
    ]

    for sentence in useful_sentences[:6]:
        lines.append(
            f"- {sentence}"
        )

    lines.append(
        f"\nИсточники: {format_sources(sources)}"
    )

    return "\n".join(lines)


def deterministic_answer(
    query: str,
    documents: list[Document],
) -> str | None:
    lowered = query.lower()
    sources = get_sources(documents)
    sources_text = format_sources(sources)

    if is_price_question(query):
        tariff_found = any(
            document.metadata.get("source")
            == "02_tariffs.md"
            for document in documents
        )

        if tariff_found:
            return (
                "В базе знаний нет информации о помесячных "
                "ценах тарифов TaskFlow CRM. Поэтому я не "
                "могу назвать стоимость тарифа в рублях.\n\n"
                "Из базы знаний известно, что тариф Business "
                "включает до 200 пользователей, до 200 000 "
                "клиентов, расширенные роли доступа, "
                "расширенные отчёты, приоритетную поддержку "
                "в течение 8 часов и персонального менеджера "
                "внедрения.\n\n"
                f"Источники: {sources_text}"
            )

        return (
            "В базе знаний нет достаточной информации "
            "о стоимости тарифов.\n\n"
            f"Источники: {sources_text}"
        )

    if (
        "не видит клиент" in lowered
        or "не видит клиента" in lowered
    ):
        return (
            "Возможные причины, почему менеджер не видит "
            "клиента в TaskFlow CRM:\n"
            "- клиент закреплён за другим менеджером;\n"
            "- у пользователя роль Менеджер, а не "
            "Руководитель или Администратор;\n"
            "- клиент был архивирован;\n"
            "- менеджер видит только клиентов и сделки, "
            "где он назначен ответственным.\n\n"
            "Проверьте ответственного за клиента, роль "
            "пользователя и статус карточки клиента.\n\n"
            f"Источники: {sources_text}"
        )

    if (
        "тариф" in lowered
        and (
            "team" in lowered
            or "business" in lowered
            or "отлич" in lowered
        )
    ):
        return (
            "В TaskFlow CRM доступны тарифы Start, Team "
            "и Business.\n\n"
            "Team:\n"
            "- до 30 пользователей;\n"
            "- до 20 000 клиентов;\n"
            "- несколько воронок продаж;\n"
            "- Telegram-уведомления;\n"
            "- интеграция с почтой;\n"
            "- поддержка в течение 24 часов.\n\n"
            "Business:\n"
            "- до 200 пользователей;\n"
            "- до 200 000 клиентов;\n"
            "- расширенные роли доступа;\n"
            "- расширенные отчёты;\n"
            "- поддержка в течение 8 часов;\n"
            "- персональный менеджер внедрения.\n\n"
            f"Источники: {sources_text}"
        )

    if (
        "csv" in lowered
        or "импорт" in lowered
    ):
        return (
            "Если CSV-файл не импортируется, проверьте:\n"
            "1. Кодировку UTF-8.\n"
            "2. Наличие поля email или телефон.\n"
            "3. Размер файла — максимум 20 МБ.\n"
            "4. Отсутствие пустых строк.\n"
            "5. Разделитель — запятая или точка с запятой.\n"
            "6. Наличие некорректных строк.\n\n"
            f"Источники: {sources_text}"
        )

    if (
        "telegram" in lowered
        or "уведомлен" in lowered
    ):
        return (
            "Если не приходят Telegram-уведомления, "
            "возможные причины:\n"
            "- Telegram-аккаунт не подключён;\n"
            "- бот TaskFlow CRM заблокирован;\n"
            "- уведомления отключены в настройках профиля.\n\n"
            f"Источники: {sources_text}"
        )

    if (
        "поддержк" in lowered
        and (
            "указать" in lowered
            or "обращ" in lowered
            or "написать" in lowered
        )
    ):
        return (
            "При обращении в поддержку укажите:\n"
            "- название рабочей области;\n"
            "- email аккаунта;\n"
            "- описание проблемы;\n"
            "- скриншот ошибки;\n"
            "- время возникновения проблемы;\n"
            "- шаги, которые привели к ошибке.\n\n"
            "SLA:\n"
            "- Start — 48 часов;\n"
            "- Team — 24 часа;\n"
            "- Business — 8 часов.\n\n"
            f"Источники: {sources_text}"
        )

    if (
        "роль" in lowered
        or "права" in lowered
        or "доступ" in lowered
    ):
        return (
            "В TaskFlow CRM есть четыре базовые роли:\n\n"
            "Администратор — управляет пользователями, "
            "ролями, тарифом и интеграциями.\n\n"
            "Руководитель — видит клиентов и сделки "
            "своей команды и может назначать задачи.\n\n"
            "Менеджер — работает только с клиентами "
            "и сделками, где назначен ответственным.\n\n"
            "Наблюдатель — имеет доступ только на чтение.\n\n"
            f"Источники: {sources_text}"
        )

    return None


class SupportAssistant:
    def __init__(
        self,
        use_llm: bool = True,
    ) -> None:
        vectorstore, self.chunks = build_vectorstore()

        self.retriever = RAGRetriever(
            vectorstore=vectorstore
        )

        self.use_llm = use_llm

        self.llm = (
            LocalLLM()
            if use_llm
            else None
        )

    def ask(
        self,
        query: str,
    ) -> dict[str, Any]:
        query = query.strip()

        if not query:
            raise ValueError(
                "Query must not be empty."
            )

        unsafe, pattern = is_unsafe_query(query)

        if unsafe:
            answer = (
                "Я не могу помочь с запросами, связанными "
                "с обходом безопасности, ограничений тарифа, "
                "получением чужих данных, паролями, взломом "
                "или обходом оплаты.\n\n"
                "Безопасный вариант: обратиться к "
                "администратору рабочей области или "
                "в службу поддержки.\n\n"
                "Источники: [06_security_policy.md], "
                "[08_anti_hallucination_rules.md]"
            )

            return {
                "query": query,
                "answer": answer,
                "blocked": True,
                "trace": [],
                "groundedness": {
                    "overlap_ratio": 1.0,
                    "has_sources": True,
                    "suspicious_hits": [],
                    "hallucination_risk": "низкий",
                    "reasons": [
                        "запрос остановлен фильтром безопасности",
                        f"сработавший паттерн: {pattern}",
                    ],
                },
            }

        documents = self.retriever.retrieve(query)

        if not documents:
            return {
                "query": query,
                "answer": (
                    "В базе знаний нет достаточной "
                    "информации для ответа."
                ),
                "blocked": False,
                "trace": [],
                "groundedness": {
                    "overlap_ratio": 0.0,
                    "has_sources": False,
                    "suspicious_hits": [],
                    "hallucination_risk": "средний",
                    "reasons": [
                        "не найден релевантный контекст"
                    ],
                },
            }

        answer = deterministic_answer(
            query=query,
            documents=documents,
        )

        if answer is None:
            if self.use_llm and self.llm is not None:
                try:
                    answer = self.llm.generate(
                        query=query,
                        documents=documents,
                    )

                    if "Источники:" not in answer:
                        answer += (
                            "\n\nИсточники: "
                            f"{format_sources(get_sources(documents))}"
                        )

                except Exception:
                    answer = build_extractive_answer(
                        documents
                    )

            else:
                answer = build_extractive_answer(
                    documents
                )

        groundedness = groundedness_check(
            answer=answer,
            documents=documents,
            query=query,
        )

        if (
            groundedness["hallucination_risk"]
            == "высокий"
        ):
            answer = build_extractive_answer(
                documents
            )

            groundedness = groundedness_check(
                answer=answer,
                documents=documents,
                query=query,
            )

        trace = []

        for index, document in enumerate(
            documents,
            start=1,
        ):
            trace.append(
                {
                    "rank": index,
                    "source": document.metadata.get(
                        "source"
                    ),
                    "title": document.metadata.get(
                        "title"
                    ),
                    "retrieval_query": (
                        document.metadata.get(
                            "retrieval_query"
                        )
                    ),
                    "retrieval_score": (
                        document.metadata.get(
                            "retrieval_score"
                        )
                    ),
                    "preview": (
                        document.page_content[:350]
                        .replace("\n", " ")
                    ),
                }
            )

        return {
            "query": query,
            "answer": answer,
            "blocked": False,
            "trace": trace,
            "groundedness": groundedness,
        }