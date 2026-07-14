import numpy as np

from agent.groundedness import is_grounded, max_similarity_to_any


class FakeEmbedder:
    """Maps known strings to fixed vectors for deterministic cosine-similarity
    testing, without loading the real model."""

    def __init__(self, vectors: dict):
        self.vectors = vectors

    def embed_texts(self, texts):
        return np.array([self.vectors[t] for t in texts])


def test_max_similarity_to_any_picks_best_reference():
    embedder = FakeEmbedder(
        {
            "answer": [1.0, 0.0],
            "close_ref": [0.9, 0.436],  # cosine similarity ~0.9
            "far_ref": [0.0, 1.0],  # cosine similarity 0.0
        }
    )
    score = max_similarity_to_any(embedder, "answer", ["close_ref", "far_ref"])
    assert score > 0.85


def test_max_similarity_to_any_empty_references():
    embedder = FakeEmbedder({"answer": [1.0, 0.0]})
    assert max_similarity_to_any(embedder, "answer", []) == 0.0


def test_is_grounded_passes_above_threshold():
    embedder = FakeEmbedder({"answer": [1.0, 0.0], "ref": [1.0, 0.0]})
    passed, score = is_grounded(embedder, "answer", ["ref"], threshold=0.55)
    assert passed is True
    assert score == 1.0


def test_is_grounded_fails_below_threshold():
    embedder = FakeEmbedder({"answer": [1.0, 0.0], "ref": [0.0, 1.0]})
    passed, score = is_grounded(embedder, "answer", ["ref"], threshold=0.55)
    assert passed is False
    assert score == 0.0


def test_is_grounded_reusable_with_different_threshold():
    """Same function, different threshold — per ARCHITECTURE.md §12.3,
    eval/run_eval.py (M6) reuses this with ANSWER_MATCH_THRESHOLD instead of
    GROUNDEDNESS_SIMILARITY_THRESHOLD. One mechanism, not two."""
    embedder = FakeEmbedder({"answer": [1.0, 0.0], "ref": [0.8, 0.6]})  # cos sim = 0.8

    passed_strict, score = is_grounded(embedder, "answer", ["ref"], threshold=0.9)
    passed_loose, _ = is_grounded(embedder, "answer", ["ref"], threshold=0.5)

    assert passed_strict is False
    assert passed_loose is True
    assert abs(score - 0.8) < 1e-6


# --- reference_vectors (M13) — pure performance passthrough, same scores ----


class _EmbedderThatMustNotBeCalledForReferences(FakeEmbedder):
    """Fails the test if embed_texts() is ever asked to embed more than
    just the answer -- proves the reference-embedding call was genuinely
    skipped, not just coincidentally equal."""

    def embed_texts(self, texts):
        if len(texts) != 1:
            raise AssertionError(f"embed_texts() called for references too: {texts!r}")
        return super().embed_texts(texts)


def test_max_similarity_to_any_with_reference_vectors_matches_embedding_fresh():
    embedder = FakeEmbedder(
        {"answer": [1.0, 0.0], "ref_a": [0.9, 0.436], "ref_b": [0.0, 1.0]}
    )
    score_fresh = max_similarity_to_any(embedder, "answer", ["ref_a", "ref_b"])
    score_precomputed = max_similarity_to_any(
        embedder, "answer", ["ref_a", "ref_b"], reference_vectors=[[0.9, 0.436], [0.0, 1.0]]
    )
    assert abs(score_fresh - score_precomputed) < 1e-9


def test_max_similarity_to_any_skips_embedding_references_when_vectors_given():
    embedder = _EmbedderThatMustNotBeCalledForReferences({"answer": [1.0, 0.0]})
    # "ref_a"/"ref_b" are deliberately absent from the embedder's known
    # strings -- if embed_texts() were called for them, FakeEmbedder's own
    # dict lookup would raise a KeyError before the assertion even ran.
    score = max_similarity_to_any(
        embedder, "answer", ["ref_a", "ref_b"], reference_vectors=[[0.9, 0.436], [0.0, 1.0]]
    )
    assert score > 0.85


def test_max_similarity_to_any_falls_back_to_embedding_when_any_vector_missing():
    # All-or-nothing: one missing vector (e.g. a chunk sourced only via
    # BM25 before weaviate_retriever's fetch_all_chunks() fix) falls back
    # to embedding ALL references fresh, rather than silently mishandling
    # a partial list.
    embedder = FakeEmbedder({"answer": [1.0, 0.0], "ref_a": [0.9, 0.436], "ref_b": [0.0, 1.0]})
    score = max_similarity_to_any(
        embedder, "answer", ["ref_a", "ref_b"], reference_vectors=[[0.9, 0.436], None]
    )
    assert score > 0.85  # correct answer -- proves the fallback path actually ran, not a crash


def test_is_grounded_with_reference_vectors_matches_embedding_fresh():
    embedder = FakeEmbedder({"answer": [1.0, 0.0], "ref": [0.8, 0.6]})  # cos sim = 0.8
    passed_fresh, score_fresh = is_grounded(embedder, "answer", ["ref"], threshold=0.7)
    passed_precomputed, score_precomputed = is_grounded(
        embedder, "answer", ["ref"], threshold=0.7, reference_vectors=[[0.8, 0.6]]
    )
    assert passed_fresh == passed_precomputed is True
    assert abs(score_fresh - score_precomputed) < 1e-9


def test_is_grounded_reference_vectors_defaults_to_none_unchanged_behavior():
    embedder = FakeEmbedder({"answer": [1.0, 0.0], "ref": [1.0, 0.0]})
    passed, score = is_grounded(embedder, "answer", ["ref"], threshold=0.55)
    assert passed is True
    assert score == 1.0
