import traceback
from pathlib import Path

from rag.chat import ask_rag
from rag.config import VECTORSTORE_DIR


def log_api_error(context: str, exc: Exception):
    print(f"[ERROR] {context}: {exc}")

    for attr in ("response", "details", "message"):
        value = getattr(exc, attr, None)
        if value and value is not exc:
            print(f"[API DETAIL] {attr}: {value}")

    extra = getattr(exc, "errors", None)
    if extra:
        print(f"[API ERRORS] {extra}")

    traceback.print_exc()


def ensure_vectorstore():
    index_path = Path(VECTORSTORE_DIR) / "chroma.sqlite3"
    if not index_path.exists():
        print("[INFO] No vector store found. Running ingestion now...\n")
        from rag.ingest import run_ingestion
        run_ingestion()
        print()


def chat_loop():
    print("Type your trade idea or question. Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("Bot: Goodbye")
            break

        if not question:
            continue

        try:
            answer = ask_rag(question)
            print("\nBot:", answer, "\n")
        except Exception as exc:
            log_api_error("Chat failed", exc)


def main():
    print("=== Trading RAG SMC Assistant ===\n")
    ensure_vectorstore()
    chat_loop()


if __name__ == "__main__":
    main()
