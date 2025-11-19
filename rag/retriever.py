from pathlib import Path

from langchain_chroma import Chroma

from .config import VECTORSTORE_DIR
from .embeddings import get_embeddings

CHROMA_INDEX_FILE = "chroma.sqlite3"


def _ensure_vectorstore_exists():
    index_path = Path(VECTORSTORE_DIR) / CHROMA_INDEX_FILE
    if not index_path.exists():
        raise RuntimeError(
            f"Vector store not found at '{VECTORSTORE_DIR}'. "
            "Run ingestion first to build the knowledge base."
        )


def get_retriever(k: int = 4):
    """
    Load persisted Chroma vector store and return a retriever.
    """
    _ensure_vectorstore_exists()

    embeddings = get_embeddings()

    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    return retriever
