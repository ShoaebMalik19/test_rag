from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from .config import DATA_DIR, VECTORSTORE_DIR
from .embeddings import get_embeddings


def load_documents():
    """
    Load all .md files from data/strategies as LangChain Documents.
    """
    loader = DirectoryLoader(
        str(DATA_DIR),
        glob="*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()
    print(f"[INGEST] Loaded {len(docs)} documents from {DATA_DIR}")
    return docs


def split_documents(docs):
    """
    Split docs into smaller chunks with overlap for better retrieval.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80,
        separators=["\n##", "\n#", "\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"[INGEST] Split into {len(chunks)} chunks")
    return chunks


def build_vectorstore(chunks):
    """
    Create or overwrite a Chroma vector store with the given chunks.
    """
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    embeddings = get_embeddings()

    print(f"[INGEST] Building Chroma store at {VECTORSTORE_DIR}")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
    )
    vectordb.persist()
    print("[INGEST] Vector store built and persisted.")
    return vectordb


def run_ingestion():
    docs = load_documents()
    chunks = split_documents(docs)
    build_vectorstore(chunks)


if __name__ == "__main__":
    run_ingestion()
