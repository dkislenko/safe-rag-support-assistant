from src.assistant import SupportAssistant


def main() -> None:
    print("Safe RAG Support Assistant")
    print("Загрузка базы знаний...\n")

    assistant = SupportAssistant()

    print(
        "Ассистент готов. "
        "Для выхода введите: exit\n"
    )

    while True:
        query = input("Вы: ").strip()

        if query.lower() in {
            "exit",
            "quit",
            "выход",
        }:
            break

        if not query:
            continue

        try:
            result = assistant.ask(query)

            print("\nАссистент:")
            print(result["answer"])

            print("\nGroundedness:")
            print(result["groundedness"])

            print("\n" + "-" * 70 + "\n")

        except Exception as exc:
            print(f"\nОшибка: {exc}\n")


if __name__ == "__main__":
    main()