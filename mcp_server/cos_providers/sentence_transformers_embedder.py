"""
Embedder that runs a sentence-transformers model in-process, on CPU, with no API calls.
"""

import asyncio
from collections.abc import Iterable

from pydantic import Field

from graphiti_core.embedder.client import EmbedderClient, EmbedderConfig

DEFAULT_EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'


class SentenceTransformersEmbedderConfig(EmbedderConfig):
    embedding_model: str = Field(default=DEFAULT_EMBEDDING_MODEL)


class SentenceTransformersEmbedder(EmbedderClient):
    def __init__(self, config: SentenceTransformersEmbedderConfig | None = None):
        from sentence_transformers import SentenceTransformer

        self.config = config or SentenceTransformersEmbedderConfig()
        self.model = SentenceTransformer(self.config.embedding_model, device='cpu')

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [v[: self.config.embedding_dim].tolist() for v in vectors]

    async def create(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        if isinstance(input_data, str):
            texts = [input_data]
        elif isinstance(input_data, list) and all(isinstance(i, str) for i in input_data):
            texts = input_data
        else:
            raise ValueError('SentenceTransformersEmbedder only embeds text')
        return (await asyncio.to_thread(self._encode, texts))[0]

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._encode, input_data_list)
