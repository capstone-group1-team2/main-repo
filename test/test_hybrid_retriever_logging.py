"""Unit tests for HybridRetriever.broaden_via_graph()'s logging.
 
Run from the repo root:
 
    pytest test/test_hybrid_retriever_logging.py -v
 
Deliberately uses fakes for `vec` and `graph` (no live Weaviate/Neo4j
needed) — this only verifies that broaden_via_graph() logs what it's
supposed to, not retrieval quality against a real corpus (that's what
test_retrieval_pipeline.py's live-stack tests are for).
 
Uses pytest's built-in `caplog` fixture rather than capsys: caplog
captures LogRecords directly, sidestepping the capsys stdout-binding/phase
gotcha that broke the first version of test_logging_ingest_style.py (see
that file's docstring for the full explanation).
"""
 
from __future__ import annotations
 
import logging
 
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.weaviate_retriever import RetrievedChunk
 
 
def _chunk(chunk_id: str, category: str, text: str = "some policy text") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id, text=text, source_file=f"{category.lower()}.md",
        category=category, heading="Some Heading", score=0.5,
    )
 
 
class FakeWeaviateRetriever:
    """Minimal stand-in for retrieval.weaviate_retriever.WeaviateRetriever.
    Returns the same fixed chunk list for every search() call and every
    fetch_all_chunks() call, regardless of query — real search relevance
    isn't what this test is checking."""
 
    def __init__(self, chunks: list):
        self._chunks = chunks
 
    def fetch_all_chunks(self) -> list:
        return self._chunks
 
    def search(self, query: str, top_k: int = 5, categories=None) -> list:
        if categories:
            return [c for c in self._chunks if c.category in categories]
        return self._chunks
 
    def close(self):
        pass
 
 
class FakeGraphRetriever:
    """Minimal stand-in for retrieval.graph_retriever.GraphRetriever.
    Returns a fixed related-categories set regardless of input, so the
    test can assert on exactly what gets passed to and logged from it."""
 
    def __init__(self, related: set):
        self._related = related
 
    def related_categories(self, categories) -> set:
        return self._related
 
    def close(self):
        pass
 
 
def test_broaden_via_graph_logs_first_pass_and_related_categories(caplog):
    first_pass_chunks = [_chunk("cancel.md::1", "CANCEL")]
    vec = FakeWeaviateRetriever(chunks=[_chunk("delivery.md::1", "DELIVERY")])
    graph = FakeGraphRetriever(related={"DELIVERY", "REFUND"})
    retriever = HybridRetriever(vec=vec, graph=graph, top_k=5)
 
    with caplog.at_level(logging.INFO, logger="retrieval.hybrid_retriever"):
        retriever.broaden_via_graph("some query", first_pass_chunks)
 
    messages = [r.getMessage() for r in caplog.records if r.name == "retrieval.hybrid_retriever"]
    assert any("first-pass categories=['CANCEL']" in m and "related=['DELIVERY', 'REFUND']" in m for m in messages), (
        f"expected a log line reporting first-pass and related categories, got: {messages}"
    )
 
 
def test_broaden_via_graph_logs_result_summary(caplog):
    first_pass_chunks = [_chunk("cancel.md::1", "CANCEL")]
    returned_chunk = _chunk("delivery.md::1", "DELIVERY")
    vec = FakeWeaviateRetriever(chunks=[returned_chunk])
    graph = FakeGraphRetriever(related={"DELIVERY"})
    retriever = HybridRetriever(vec=vec, graph=graph, top_k=5)
 
    with caplog.at_level(logging.INFO, logger="retrieval.hybrid_retriever"):
        result = retriever.broaden_via_graph("some query", first_pass_chunks)
 
    messages = [r.getMessage() for r in caplog.records if r.name == "retrieval.hybrid_retriever"]
    summary_lines = [m for m in messages if m.startswith("broaden_via_graph: returned")]
    assert len(summary_lines) == 1, f"expected exactly one result-summary log line, got: {summary_lines}"
    assert f"returned {len(result.chunks)} chunks" in summary_lines[0]
    assert f"confidence={result.confidence}" in summary_lines[0]
 
 
def test_broaden_via_graph_logs_nothing_below_info_level(caplog):
    """At WARNING level, neither log line should appear — confirms these
    are genuinely INFO-level (diagnostic), not something that would spam
    production logs configured at a higher threshold."""
    first_pass_chunks = [_chunk("cancel.md::1", "CANCEL")]
    vec = FakeWeaviateRetriever(chunks=[_chunk("delivery.md::1", "DELIVERY")])
    graph = FakeGraphRetriever(related={"DELIVERY"})
    retriever = HybridRetriever(vec=vec, graph=graph, top_k=5)
 
    with caplog.at_level(logging.WARNING, logger="retrieval.hybrid_retriever"):
        retriever.broaden_via_graph("some query", first_pass_chunks)
 
    messages = [r.getMessage() for r in caplog.records if r.name == "retrieval.hybrid_retriever"]
    assert messages == [], f"expected no log records at WARNING level, got: {messages}"
 