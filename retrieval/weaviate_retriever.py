"""Dense (vector) search against Weaviate — the semantic half of hybrid
retrieval (ARCHITECTURE.md §6). Read-only; the collection itself is owned
by ingestion/weaviate_loader.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import weaviate
from weaviate.classes.query import Filter, MetadataQuery

from app.config import (
    RETRIEVAL_TOP_K,
    WEAVIATE_COLLECTION_NAME,
    WEAVIATE_GRPC_PORT,
    WEAVIATE_HOST,
    WEAVIATE_PORT,
)

_RETURN_PROPERTIES = ["chunk_id", "source_file", "category", "heading", "text"]


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    source_file: str
    category: str
    heading: str
    score: float  # cosine similarity, higher is better, roughly in [0, 1]
    # M13: the exact embedding vector Weaviate already stored for this chunk
    # at ingestion time (`ingestion/weaviate_loader.py`'s `upsert_chunks()`),
    # fetched via `include_vector=True` on both query methods below.
    # Optional/defaulted so any existing construction site without it still
    # works. Lets agent/groundedness.py skip re-embedding retrieved chunk
    # text on every request — see ANALYSIS.md's M13 entry.
    vector: Optional[list] = None


def _extract_vector(obj) -> Optional[list]:
    """Weaviate v4's `include_vector=True` returns a dict keyed by vector
    name even for a collection with a single, unnamed vector (verified
    empirically against this project's real collection: `{"default": [...]}`)
    — this collection was created with `Configure.Vectorizer.none()` and a
    single self-provided vector per object (`ingestion/weaviate_loader.py`),
    so "default" is always the right key."""
    if not obj.vector:
        return None
    return obj.vector.get("default")


class WeaviateRetriever:
    def __init__(
        self,
        embedder,
        host: str = WEAVIATE_HOST,
        port: int = WEAVIATE_PORT,
        grpc_port: int = WEAVIATE_GRPC_PORT,
        collection_name: str = WEAVIATE_COLLECTION_NAME,
    ):
        self._embedder = embedder
        self._client = weaviate.connect_to_local(host=host, port=port, grpc_port=grpc_port)
        self._collection_name = collection_name

    def close(self):
        self._client.close()

    def is_ready(self) -> bool:
        """Liveness check for GET /health (M5) — reuses this already-open
        connection instead of a throwaway per-request client (M1's original
        health check did the latter; flagged there as needing replacement
        once the real app.state-backed lifespan wiring existed)."""
        try:
            return self._client.is_ready()
        except Exception:
            return False

    def _to_chunk(self, obj) -> RetrievedChunk:
        props = obj.properties
        # Weaviate's default HNSW distance metric is cosine; embeddings are
        # unit-normalized (ingestion/embedder.py), so 1 - distance = cosine
        # similarity. Verified empirically against the M2 corpus in M3.
        similarity = 1.0 - obj.metadata.distance
        return RetrievedChunk(
            chunk_id=props["chunk_id"],
            text=props["text"],
            source_file=props["source_file"],
            category=props["category"],
            heading=props["heading"],
            score=similarity,
            vector=_extract_vector(obj),
        )

    def search(self, query: str, top_k: int = RETRIEVAL_TOP_K, categories=None) -> list:
        """`categories`, if given, restricts the search to chunks in those
        categories only — used by hybrid_retriever.broaden_via_graph() to
        pull candidates specifically from related categories."""
        vector = self._embedder.embed_query(query)
        collection = self._client.collections.get(self._collection_name)
        filters = Filter.by_property("category").contains_any(list(categories)) if categories else None
        result = collection.query.near_vector(
            near_vector=vector.tolist(),
            limit=top_k,
            filters=filters,
            return_metadata=MetadataQuery(distance=True),
            return_properties=_RETURN_PROPERTIES,
            include_vector=True,
        )
        return [self._to_chunk(obj) for obj in result.objects]

    def fetch_all_chunks(self) -> list:
        """Every chunk currently in Weaviate, no vector search involved —
        used by hybrid_retriever to build its in-memory BM25 index.
        `include_vector=True` (M13) matters here too, not just in
        search(): a chunk surfaced only via the BM25 side of hybrid
        fusion (never in search()'s dense top-k) is sourced from THIS
        method, so it needs its own vector fetched here or
        agent/groundedness.py's reuse optimization would silently miss it."""
        collection = self._client.collections.get(self._collection_name)
        result = collection.query.fetch_objects(
            limit=10_000, return_properties=_RETURN_PROPERTIES, include_vector=True
        )
        return [
            RetrievedChunk(
                chunk_id=obj.properties["chunk_id"],
                text=obj.properties["text"],
                source_file=obj.properties["source_file"],
                category=obj.properties["category"],
                heading=obj.properties["heading"],
                score=0.0,
                vector=_extract_vector(obj),
            )
            for obj in result.objects
        ]
