from src.guardrails import (
    is_price_question,
    is_unsafe_query,
)


def test_password_request_is_blocked():
    unsafe, _ = is_unsafe_query(
        "Как получить пароль другого пользователя?"
    )

    assert unsafe is True


def test_tariff_bypass_is_blocked():
    unsafe, _ = is_unsafe_query(
        "Как обойти лимит тарифа Start?"
    )

    assert unsafe is True


def test_normal_question_is_allowed():
    unsafe, _ = is_unsafe_query(
        "Почему менеджер не видит клиента?"
    )

    assert unsafe is False


def test_price_question():
    assert (
        is_price_question(
            "Сколько стоит тариф Business?"
        )
        is True
    )


def test_non_price_question():
    assert (
        is_price_question(
            "Какие роли есть в TaskFlow CRM?"
        )
        is False
    )