from typing import List

import google.generativeai as genai
from langchain_core.embeddings import Embeddings

from .config import GOOGLE_API_KEY, EMBEDDING_MODEL

genai.configure(api_key=GOOGLE_API_KEY)


class GeminiEmbeddings(Embeddings):
    """
    Minimal LangChain-compatible wrapper around the Gemini embedding endpoint.
    """

    def __init__(self, model: str):
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

    def _embed(self, text: str) -> List[float]:
        payload = text or ""
        result = genai.embed_content(model=self.model, content=payload)
        embedding = result.get("embedding")
        if embedding is None:
            raise RuntimeError("Gemini embedding response did not contain 'embedding'.")
        return embedding


def get_embeddings() -> Embeddings:
    return GeminiEmbeddings(model=EMBEDDING_MODEL)

