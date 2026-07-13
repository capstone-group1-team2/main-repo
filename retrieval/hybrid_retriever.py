"""Fuses BM25 keyword search with dense vector search, pulls related graph
concepts for context, and labels retrieval confidence (ARCHITECTURE.md §6).

Fusion algorithm: ARCHITECTURE.md doesn't specify one, so this uses
Reciprocal Rank Fusion (RRF) — a standard, parameter-light way to combine
two differently-scaled rankings (BM25 scores and cosine similarities aren't
comparable on the same scale; RRF sidesteps that by fusing on RANK, not raw
score). See ANALYSIS.md's M3 entry for why the *confidence label* still uses
the top chunk's raw dense cosine similarity rather than the RRF score itself
(RRF scores are ~1/60, incompatible with the 0.75/0.45 thresholds).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi

from app.config import MIN_CONFIDENCE_HIGH, MIN_CONFIDENCE_MED, RETRIEVAL_TOP_K
import logging


logger = logging.getLogger("retrieval.hybrid_retriever")

_RRF_K = 60
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list:
    return _TOKEN_RE.findall(text.lower())


def confidence_label(top_score: float) -> str:
    if top_score >= MIN_CONFIDENCE_HIGH:
        return "High"
    if top_score >= MIN_CONFIDENCE_MED:
        return "Medium"
    return "Low"


@dataclass(frozen=True)
class RetrievalResult:
    chunks: list
    related_concepts: set
    confidence: str
    top_dense_score: float = 0.0
    # M8: max BM25 score across the whole corpus for this query — a second,
    # independent relevance signal alongside top_dense_score. See
    # ANALYSIS.md's M8 entry: dense similarity alone can't distinguish a
    # fluent-but-irrelevant query from a genuinely low-confidence real one
    # (they can score nearly identically), but a query with near-zero
    # literal vocabulary overlap with the corpus almost always also has a
    # low BM25 score even when its dense score is misleadingly high.
    bm25_max_score: float = 0.0


class HybridRetriever:
    def __init__(self, vec, graph, top_k: int = RETRIEVAL_TOP_K):
        self.vec = vec
        self.graph = graph
        self.top_k = top_k
        self._bm25 = None
        self._bm25_chunk_ids = []
        self._bm25_chunks_by_id = {}
        self.refresh_bm25_index()

    def close(self):
        self.vec.close()
        self.graph.close()

    def refresh_bm25_index(self):
        """Builds an in-memory BM25 index over every chunk currently in
        Weaviate. Built once at construction time — fine at this corpus's
        scale (93 chunks); a long-lived backend process should call this
        again after each ingestion run rather than holding a static
        snapshot forever (flagged in ANALYSIS.md as an M5 wiring note)."""
        all_chunks = self.vec.fetch_all_chunks()
        self._bm25_chunk_ids = [c.chunk_id for c in all_chunks]
        self._bm25_chunks_by_id = {c.chunk_id: c for c in all_chunks}
        tokenized = [_tokenize(c.text) for c in all_chunks]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def _bm25_scores(self, query: str) -> list:
        if self._bm25 is None:
            return []
        return self._bm25.get_scores(_tokenize(query))

    def _bm25_ranked_ids(self, query: str, limit: int, scores: list = None) -> list:
        scores = self._bm25_scores(query) if scores is None else scores
        if len(scores) == 0:
            return []
        ranked = sorted(zip(self._bm25_chunk_ids, scores), key=lambda p: p[1], reverse=True)
        return [cid for cid, score in ranked[:limit] if score > 0]

    def _fuse(self, ranked_id_lists) -> list:
        """Reciprocal Rank Fusion across any number of ranked id lists."""
        scores = {}
        for ranked_ids in ranked_id_lists:
            for rank, cid in enumerate(ranked_ids):
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank + 1)
        return sorted(scores, key=lambda cid: scores[cid], reverse=True)

    def _build_result(
        self, fused_ids: list, chunks_by_id: dict, top_dense_score: float, bm25_max_score: float
    ) -> "RetrievalResult":
        chunks = [chunks_by_id[cid] for cid in fused_ids[: self.top_k] if cid in chunks_by_id]
        categories = {c.category for c in chunks}
        related_concepts = self.graph.related_categories(categories)
        return RetrievalResult(
            chunks=chunks,
            related_concepts=related_concepts,
            confidence=confidence_label(top_dense_score),
            top_dense_score=top_dense_score,
            bm25_max_score=bm25_max_score,
        )

    def retrieve(self, query: str) -> RetrievalResult:
        pool_size = self.top_k * 3
        dense_chunks = self.vec.search(query, top_k=pool_size)
        dense_by_id = {c.chunk_id: c for c in dense_chunks}
        dense_ranked_ids = [c.chunk_id for c in dense_chunks]
        bm25_scores = self._bm25_scores(query)
        bm25_ranked_ids = self._bm25_ranked_ids(query, pool_size, scores=bm25_scores)

        fused_ids = self._fuse([dense_ranked_ids, bm25_ranked_ids])
        chunks_by_id = {**self._bm25_chunks_by_id, **dense_by_id}
        top_dense_score = dense_chunks[0].score if dense_chunks else 0.0
        bm25_max_score = max(bm25_scores) if len(bm25_scores) else 0.0

        return self._build_result(fused_ids, chunks_by_id, top_dense_score, bm25_max_score)

    def broaden_via_graph(self, original_query: str, first_pass_chunks: list) -> RetrievalResult:
        """Retried once after a failed groundedness check (ARCHITECTURE.md
        §8, routing path a) — broadens the search to include chunks from
        categories related to the first pass's, without spending an extra
        Groq call to rewrite the query. Exact signature is a cross-milestone
        interface: agent/agent.py (M4) calls this directly per §15.
        """
        
        first_pass_categories = {c.category for c in first_pass_chunks}
        related = self.graph.related_categories(first_pass_categories)

        logger.info(
        "broaden_via_graph: first-pass categories=%s -> related=%s",
        sorted(first_pass_categories), sorted(related),
    )
        
        pool_size = self.top_k * 3
        dense_chunks = self.vec.search(original_query, top_k=pool_size)
        dense_by_id = {c.chunk_id: c for c in dense_chunks}
        dense_ranked_ids = [c.chunk_id for c in dense_chunks]
        bm25_scores = self._bm25_scores(original_query)
        bm25_ranked_ids = self._bm25_ranked_ids(original_query, pool_size, scores=bm25_scores)

        related_ranked_ids = []
        related_by_id = {}
        if related:
            related_chunks = self.vec.search(original_query, top_k=pool_size, categories=related)
            related_ranked_ids = [c.chunk_id for c in related_chunks]
            related_by_id = {c.chunk_id: c for c in related_chunks}

        fused_ids = self._fuse([dense_ranked_ids, bm25_ranked_ids, related_ranked_ids])
        chunks_by_id = {**self._bm25_chunks_by_id, **dense_by_id, **related_by_id}
        top_dense_score = dense_chunks[0].score if dense_chunks else 0.0
        bm25_max_score = max(bm25_scores) if len(bm25_scores) else 0.0

        result = self._build_result(fused_ids, chunks_by_id, top_dense_score, bm25_max_score)
        logger.info(
            "broaden_via_graph: returned %d chunks, confidence=%s, top_dense_score=%.4f",
            len(result.chunks), result.confidence, result.top_dense_score,
        )
        return result