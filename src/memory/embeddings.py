"""Pinned embedding model wrapper.

Uses OpenAI text-embedding-3-small for vector search.
The embedding model is infrastructure — it never reasons, never converses,
never touches user-facing output. It maps text to coordinates.
"""

from __future__ import annotations

import logging
from typing import Sequence

log = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
EMBEDDING_VERSION = "v1"


class EmbeddingClient:
    def __init__(self, api_key: str, model: str = EMBEDDING_MODEL):
        self.model = model
        self._client = None
        self._api_key = api_key

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        client = self._get_client()
        response = client.embeddings.create(model=self.model, input=text)
        return response.data[0].embedding

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed multiple texts in a single API call."""
        if not texts:
            return []
        client = self._get_client()
        response = client.embeddings.create(model=self.model, input=list(texts))
        return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
