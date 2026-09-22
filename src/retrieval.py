from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.config import (
    MMR_FETCH_K,
    MMR_LAMBDA,
    RETRIEVAL_K,
)


QUERY_EXPANSIONS = {
    "тариф": [
        "план",
        "ограничения тарифа",
        "SLA тарифа",
    ],
    "csv": [
        "импорт CSV",
        "ошибка импорта",
        "файл клиентов",
    ],
    "уведомления": [
        "Telegram-уведомления",
        "настройки уведомлений",
    ],
    "telegram": [
        "Telegram-уведомления",
        "бот TaskFlow CRM",
    ],
    "права": [
        "роли доступа",
        "разрешения пользователей",
    ],
    "роль": [
        "роли доступа",
        "права пользователя",
    ],
    "доступ": [
        "роль пользователя",
        "восстановление доступа",
    ],
    "клиент": [
        "карточка клиента",
        "видимость клиента",
    ],
    "почта": [
        "почтовая интеграция",
        "email клиента",
    ],
    "поддержка": [
        "обращение в поддержку",
        "SLA поддержки",
    ],
}


def rewrite_query(query: str) -> list[str]:
    query = query.strip()
    lowered = query.lower()

    variants = [query]

    for keyword, expansions in QUERY_EXPANSIONS.items():
        if keyword not in lowered:
            continue

        for expansion in expansions:
            variants.append(
                f"{query}. Связанные термины: {expansion}"
            )

    unique: list[str] = []

    for variant in variants:
        if variant not in unique:
            unique.append(variant)

    return unique[:4]


class RAGRetriever:
    def __init__(
        self,
        vectorstore: FAISS,
    ) -> None:
        self.vectorstore = vectorstore

    def retrieve(
        self,
        query: str,
        k: int = RETRIEVAL_K,
    ) -> list[Document]:
        query_variants = rewrite_query(query)

        retrieved: list[Document] = []
        seen: set[tuple[str | None, str]] = set()

        for retrieval_query in query_variants:
            results = self.vectorstore.similarity_search_with_score(
                retrieval_query,
                k=k,
            )

            for document, score in results:
                key = (
                    document.metadata.get("source"),
                    document.page_content[:180],
                )

                if key in seen:
                    continue

                new_document = Document(
                    page_content=document.page_content,
                    metadata=dict(document.metadata),
                )

                new_document.metadata["retrieval_score"] = float(score)
                new_document.metadata["retrieval_query"] = retrieval_query

                retrieved.append(new_document)
                seen.add(key)

        try:
            mmr_documents = (
                self.vectorstore.max_marginal_relevance_search(
                    query,
                    k=k,
                    fetch_k=MMR_FETCH_K,
                    lambda_mult=MMR_LAMBDA,
                )
            )

            for document in mmr_documents:
                key = (
                    document.metadata.get("source"),
                    document.page_content[:180],
                )

                if key in seen:
                    continue

                new_document = Document(
                    page_content=document.page_content,
                    metadata=dict(document.metadata),
                )

                new_document.metadata["retrieval_score"] = None
                new_document.metadata["retrieval_query"] = "MMR"

                retrieved.append(new_document)
                seen.add(key)

        except Exception:
            pass

        return retrieved[:k]