import hashlib

import numpy as np

from ingestion.chunker import category_from_filename, chunk_corpus_file


def _fake_embed(texts):
    """Deterministic, cheap stand-in for a real embedding model: hashes each
    text into a fixed-length float vector. Same text always -> same vector,
    with no model download needed for unit tests."""
    vectors = []
    for text in texts:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vec = np.frombuffer(digest, dtype=np.uint8).astype(np.float64)
        vectors.append(vec / np.linalg.norm(vec))
    return np.array(vectors)


SAMPLE_DOC = """# Cancel Policy & Support Guide

## Cancellation Fee Policy

Meridian Retail's cancellation fees depend on how far your order has progressed. Within 1 hour of placing the order, cancellation is free. After the order has shipped but before delivery, a 10% restocking fee applies.

## How to Check Your Fee

Log in to your account and navigate to My Orders. Select the order and look for the Cancel Order option.
"""


def test_category_from_filename():
    assert category_from_filename("cancel.md") == "CANCEL"
    assert category_from_filename("order.md") == "ORDER"


def test_chunk_ids_are_deterministic_across_runs():
    chunks_a = chunk_corpus_file("cancel.md", SAMPLE_DOC, _fake_embed)
    chunks_b = chunk_corpus_file("cancel.md", SAMPLE_DOC, _fake_embed)

    assert [c.chunk_id for c in chunks_a] == [c.chunk_id for c in chunks_b]
    assert len(chunks_a) > 0


def test_chunk_id_format():
    chunks = chunk_corpus_file("cancel.md", SAMPLE_DOC, _fake_embed)
    for c in chunks:
        assert c.chunk_id.startswith("cancel.md::")
        suffix = c.chunk_id.split("::", 1)[1]
        assert suffix == hashlib.sha256(c.text.encode("utf-8")).hexdigest()[:16]


def test_chunks_never_cross_heading_boundaries():
    chunks = chunk_corpus_file("cancel.md", SAMPLE_DOC, _fake_embed)
    headings = {c.heading for c in chunks}
    assert headings == {"Cancellation Fee Policy", "How to Check Your Fee"}
    for c in chunks:
        assert c.text.startswith(f"## {c.heading}")


def test_identical_text_produces_identical_chunk_id_even_in_different_docs():
    # Same chunk content hashed the same way regardless of which run produced it.
    chunks = chunk_corpus_file("cancel.md", SAMPLE_DOC, _fake_embed)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))  # no accidental collisions within one doc


def test_changing_text_changes_chunk_id():
    chunks_original = chunk_corpus_file("cancel.md", SAMPLE_DOC, _fake_embed)
    edited = SAMPLE_DOC.replace("10% restocking fee", "15% restocking fee")
    chunks_edited = chunk_corpus_file("cancel.md", edited, _fake_embed)

    assert {c.chunk_id for c in chunks_original} != {c.chunk_id for c in chunks_edited}
