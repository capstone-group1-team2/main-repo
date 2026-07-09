"""Loads BAAI/bge-large-en-v1.5 once and exposes document/query embedding.

Reused as-is by retrieval/weaviate_retriever.py and agent/groundedness.py
(M3) so corpus chunks, incoming queries, and groundedness checks all live
in the same vector space — this is the ONE place that model gets loaded.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from ingestion.config import EMBEDDING_MODEL_NAME

# bge-large-en-v1.5 expects this instruction prefix on the QUERY side only —
# never on documents. See https://huggingface.co/BAAI/bge-large-en-v1.5.
_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self._model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list) -> np.ndarray:
        """Embeds document/chunk texts — no instruction prefix."""
        if not texts:
            return np.empty((0, self._model.get_sentence_embedding_dimension()))
        return self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

    def embed_query(self, text: str) -> np.ndarray:
        return self._model.encode(
            _QUERY_INSTRUCTION + text, normalize_embeddings=True, convert_to_numpy=True
        )
