from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.assistant import SupportAssistant


app = FastAPI(
    title="Safe RAG Support Assistant",
    description=(
        "RAG assistant for SaaS technical support "
        "with semantic retrieval and safety guardrails."
    ),
    version="1.0.0",
)


class AskRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=2,
        max_length=2000,
        examples=[
            "Почему менеджер не видит клиента?"
        ],
    )


class AskResponse(BaseModel):
    query: str
    answer: str
    blocked: bool
    trace: list[dict[str, Any]]
    groundedness: dict[str, Any]


@lru_cache
def get_assistant() -> SupportAssistant:
    return SupportAssistant()


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
    }


@app.post(
    "/ask",
    response_model=AskResponse,
)
def ask(
    request: AskRequest,
) -> AskResponse:
    try:
        result = get_assistant().ask(
            request.query
        )

        return AskResponse(**result)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc