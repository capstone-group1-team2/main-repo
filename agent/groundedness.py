"""Local, cost-free groundedness check — no Groq call (ARCHITECTURE.md §8).

Also reused by eval/run_eval.py (M6) for answer-correctness scoring against
a separate ANSWER_MATCH_THRESHOLD, per ARCHITECTURE.md §12.3's explicit
instruction not to build a second scoring mechanism from scratch: one
function, two thresholds depending on the caller.
"""

from __future__ import annotations

import numpy as np

from app.config import GROUNDEDNESS_SIMILARITY_THRESHOLD


def max_similarity_to_any(embedder, answer: str, references: list, reference_vectors: list = None) -> float:
    """Cosine similarity between `answer` and its single best-matching
    reference text — not an average. This is the groundedness check's
    actual shape: is the answer supported by AT LEAST ONE retrieved chunk,
    not by the chunks collectively. `embedder` is injected (an
    ingestion.embedder.Embedder instance) rather than imported here, so
    every caller shares the one already-loaded model instance.

    `reference_vectors` (M13, optional, defaults to None): the retrieved
    chunks' embeddings, already computed once at ingestion time and stored
    in Weaviate (`retrieval/weaviate_retriever.py`'s `RetrievedChunk.vector`,
    fetched via `include_vector=True`). When given and complete (every
    entry non-None), re-embedding `references` is skipped entirely — this
    was the single largest, most avoidable cost in an information-route
    request (see ANALYSIS.md's M12/M13 entries: up to ~90% of end-to-end
    latency on the slowest examples). `answer` is always freshly embedded
    regardless — it's newly generated text with no precomputed vector to
    reuse. Callers that never pass `reference_vectors` (e.g.
    eval/run_eval.py's answer-correctness check against a hand-written
    `expected_answer`, which was never ingested and has no stored vector)
    behave exactly as before this parameter existed."""
    if not references:
        return 0.0
    answer_vec = embedder.embed_texts([answer])[0]
    if reference_vectors is not None and all(v is not None for v in reference_vectors):
        reference_vecs = np.array(reference_vectors)
    else:
        reference_vecs = embedder.embed_texts(references)
    sims = reference_vecs @ answer_vec / (
        np.linalg.norm(reference_vecs, axis=1) * np.linalg.norm(answer_vec)
    )
    return float(np.max(sims))


def is_grounded(
    embedder, answer: str, references: list, threshold: float = GROUNDEDNESS_SIMILARITY_THRESHOLD,
    reference_vectors: list = None,
):
    """Returns (passed, score). `threshold` defaults to
    GROUNDEDNESS_SIMILARITY_THRESHOLD but eval/run_eval.py (M6) passes
    ANSWER_MATCH_THRESHOLD instead when scoring answer correctness against
    `expected_answer` — same mechanism, different bar. `reference_vectors`
    (M13) is a pure performance passthrough to `max_similarity_to_any` —
    see its docstring."""
    score = max_similarity_to_any(embedder, answer, references, reference_vectors=reference_vectors)
    return score >= threshold, score
