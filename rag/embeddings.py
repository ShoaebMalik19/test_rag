from typing import List

import google.genai as genai
from langchain_core.embeddings import Embeddings

from .config import GOOGLE_API_KEY, EMBEDDING_MODEL


class GeminiEmbeddings(Embeddings):
    def __init__(self, model: str):
        self.model = model
        self._client = genai.Client(api_key=GOOGLE_API_KEY)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

    def _embed(self, text: str) -> List[float]:
        result = self._client.models.embed_content(
            model=self.model,
            contents=[text or " "],
        )
        return list(result.embeddings[0].values)


def get_embeddings() -> Embeddings:
    return GeminiEmbeddings(model=EMBEDDING_MODEL)
