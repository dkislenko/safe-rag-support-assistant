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
    """
    Split text into individual sentences.
    """
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
    """
    Remove Markdown formatting before building
    an extractive fallback response.
    """
    text = document.page_content

    # Remove Markdown headings.
    text = re.sub(
        r"^#{1,6}\s+.*$",
        "",
        text,
        flags=re.MULTILINE,
    )

    # Remove Markdown bullet markers.
    text = re.sub(
        r"^\s*[-*+]\s+",
        "",
        text,
        flags=re.MULTILINE,
    )

    # Remove numbered list markers.
    text = re.sub(
        r"^\s*\d+\.\s+",
        "",
        text,
        flags=re.MULTILINE,
    )

    # Normalize whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def build_extractive_answer(
    documents: list[Document],
) -> str:
    """
    Build a safe answer directly from retrieved documents.

    This function is used as a fallback when the LLM returns
    an empty, low-quality or potentially hallucinated answer.
    """
    useful_sentences: list[str] = []
    used_sources: list[str] = []

    for document in documents[:4]:
        text = clean_document_text(document)

        source = document.metadata.get(
            "source",
            "unknown",
        )

        document_was_used = False

        for sentence in sentence_split(text):
            if 40 <= len(sentence) <= 260:
                useful_sentences.append(sentence)
                document_was_used = True

            if len(useful_sentences) >= 6:
                break

        if (
            document_was_used
            and source not in used_sources
        ):
            used_sources.append(source)

        if len(useful_sentences) >= 6:
            break

    if not useful_sentences:
        return (
            "В базе знаний нет достаточной информации "
            "для точного ответа."
        )

    lines = [
        "На основе базы знаний:"
    ]

    for sentence in useful_sentences[:6]:
        lines.append(
            f"- {sentence}"
        )

    if used_sources:
        lines.append(
            "\nИсточники: "
            f"{format_sources(used_sources)}"
        )

    return "\n".join(lines)


def deterministic_answer(
    query: str,
    documents: list[Document],
) -> str | None:
    """
    Handle scenarios where a deterministic response
    is safer or more reliable than LLM generation.
    """
    lowered = query.lower()

    available_sources = get_sources(documents)

    def sources_for(
        *preferred_sources: str,
    ) -> str:
        selected = [
            source
            for source in preferred_sources
            if source in available_sources
        ]

        if not selected:
            selected = available_sources

        return format_sources(selected)

    # Price questions.
    if is_price_question(query):
        tariff_found = (
            "02_tariffs.md"
            in available_sources
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
                "Источники: "
                f"{sources_for('02_tariffs.md')}"
            )

        return (
            "В базе знаний нет достаточной информации "
            "о стоимости тарифов.\n\n"
            "Источники: "
            f"{format_sources(available_sources)}"
        )

    # Client visibility.
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
            "Источники: "
            f"{sources_for(
                '05_troubleshooting.md',
                '03_roles_and_permissions.md',
            )}"
        )

    # Tariff comparison.
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
            "Источники: "
            f"{sources_for('02_tariffs.md')}"
        )

    # CSV import.
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
            "Источники: "
            f"{sources_for(
                '04_integrations.md',
                '05_troubleshooting.md',
            )}"
        )

    # Telegram notifications.
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
            "Источники: "
            f"{sources_for('05_troubleshooting.md')}"
        )

    # Support request.
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
            "Источники: "
            f"{sources_for('07_support_process.md')}"
        )

    # Roles and permissions.
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
            "Источники: "
            f"{sources_for(
                '03_roles_and_permissions.md'
            )}"
        )

    return None


class SupportAssistant:
    def __init__(
        self,
        use_llm: bool = True,
    ) -> None:
        vectorstore, self.chunks = (
            build_vectorstore()
        )

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
        """
        Process a user query through:

        1. Safety guardrails
        2. Retrieval
        3. Deterministic rules or LLM generation
        4. Groundedness validation
        5. Safe extractive fallback
        6. Retrieval trace
        """
        query = query.strip()

        if not query:
            raise ValueError(
                "Query must not be empty."
            )

        # -------------------------------------------------
        # 1. Safety check before retrieval / generation
        # -------------------------------------------------

        unsafe, pattern = is_unsafe_query(
            query
        )

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
                        (
                            "запрос остановлен "
                            "фильтром безопасности"
                        ),
                        (
                            "сработавший паттерн: "
                            f"{pattern}"
                        ),
                    ],
                },
            }

        # -------------------------------------------------
        # 2. Retrieval
        # -------------------------------------------------

        documents = self.retriever.retrieve(
            query
        )

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
                        (
                            "не найден релевантный "
                            "контекст"
                        )
                    ],
                },
            }

        # -------------------------------------------------
        # 3. Deterministic response
        # -------------------------------------------------

        answer = deterministic_answer(
            query=query,
            documents=documents,
        )

        # -------------------------------------------------
        # 4. LLM generation
        # -------------------------------------------------

        if answer is None:
            if (
                self.use_llm
                and self.llm is not None
            ):
                try:
                    answer = self.llm.generate(
                        query=query,
                        documents=documents,
                    )

                    if (
                        "Источники:"
                        not in answer
                    ):
                        answer += (
                            "\n\nИсточники: "
                            f"{format_sources(
                                get_sources(documents)
                            )}"
                        )

                except Exception:
                    # If the model fails or returns
                    # a low-quality response, use
                    # a safe extractive fallback.
                    answer = build_extractive_answer(
                        documents
                    )

            else:
                answer = build_extractive_answer(
                    documents
                )

        # -------------------------------------------------
        # 5. Groundedness validation
        # -------------------------------------------------

        groundedness = groundedness_check(
            answer=answer,
            documents=documents,
            query=query,
        )

        # Replace high-risk generation with
        # an extractive grounded answer.
        if (
            groundedness[
                "hallucination_risk"
            ]
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

        # -------------------------------------------------
        # 6. Retrieval trace
        # -------------------------------------------------

        trace: list[dict[str, Any]] = []

        for index, document in enumerate(
            documents,
            start=1,
        ):
            retrieval_score = (
                document.metadata.get(
                    "retrieval_score"
                )
            )

            trace.append(
                {
                    "rank": index,
                    "source": (
                        document.metadata.get(
                            "source"
                        )
                    ),
                    "title": (
                        document.metadata.get(
                            "title"
                        )
                    ),
                    "retrieval_query": (
                        document.metadata.get(
                            "retrieval_query"
                        )
                    ),
                    "retrieval_score": (
                        retrieval_score
                    ),
                    "preview": (
                        document.page_content[
                            :350
                        ].replace(
                            "\n",
                            " ",
                        )
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