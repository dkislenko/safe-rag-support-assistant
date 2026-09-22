from pathlib import Path

from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "data" / "knowledge_base"


def _extract_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        line = line.strip()

        if line.startswith("# "):
            return line[2:].strip()

    return fallback


def load_knowledge_base(
    directory: Path = DEFAULT_KNOWLEDGE_BASE_DIR,
) -> list[Document]:
    if not directory.exists():
        raise FileNotFoundError(
            f"Knowledge base directory not found: {directory}"
        )

    documents: list[Document] = []

    for file_path in sorted(directory.glob("*.md")):
        text = file_path.read_text(encoding="utf-8").strip()

        if not text:
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": file_path.name,
                    "title": _extract_title(
                        text=text,
                        fallback=file_path.stem,
                    ),
                },
            )
        )

    if not documents:
        raise ValueError(
            f"No Markdown documents found in: {directory}"
        )

    return documents


if __name__ == "__main__":
    documents = load_knowledge_base()

    print(f"Loaded documents: {len(documents)}")

    for document in documents:
        print(
            f"- {document.metadata['title']} "
            f"({document.metadata['source']})"
        )