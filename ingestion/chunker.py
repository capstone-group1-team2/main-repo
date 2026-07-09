"""Splits a corpus markdown document into semantically coherent chunks.

Hierarchy: heading -> paragraph -> sentence, per ARCHITECTURE.md §4.
`##` headings are always a hard chunk boundary (each is already a coherent
policy topic, e.g. "## How to Cancel an Order"). Within a heading section,
sentences are grouped using an embedding-similarity breakpoint search — the
same approach as LlamaIndex's SemanticSplitterNodeParser: combine each
sentence with its CHUNK_BUFFER_SIZE neighbors on each side, embed each
combined group, and cut wherever the cosine distance to the next group
exceeds the CHUNK_BREAKPOINT_PERCENTILE-th percentile of distances within
that section. Hand-rolled here (not a llama-index dependency) since this is
the only place in the codebase that needs it.

chunk_id = f"{source_file}::{sha256(chunk_text)[:16]}" — deterministic and
content-addressed, so identical text always maps to the same chunk_id no
matter how many times it's re-ingested.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import numpy as np

from ingestion.config import CHUNK_BREAKPOINT_PERCENTILE, CHUNK_BUFFER_SIZE

_HEADING_RE = re.compile(r"^##\s+(.*)$", re.MULTILINE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    source_file: str
    category: str
    heading: str


def category_from_filename(source_file: str) -> str:
    return source_file.rsplit(".", 1)[0].upper()


def _split_sentences(text: str) -> list:
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def _split_sections(markdown_text: str) -> list:
    """Splits on '## ' headings into (heading, body) pairs. The top-level
    '# Title' line carries no chunk-worthy content of its own and is dropped."""
    matches = list(_HEADING_RE.finditer(markdown_text))
    sections = []
    for i, m in enumerate(matches):
        heading = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown_text)
        body = markdown_text[start:end].strip()
        if body:
            sections.append((heading, body))
    return sections


def _semantic_groups(sentences: list, embed_fn, buffer_size: int, breakpoint_percentile: float) -> list:
    """Groups a section's sentences into chunk-worthy blocks using
    embedding-similarity breakpoints. Deterministic given a deterministic
    embed_fn: same sentences in, same groups out, every time."""
    if len(sentences) <= 1:
        return [sentences] if sentences else []

    combined = []
    for i in range(len(sentences)):
        lo = max(0, i - buffer_size)
        hi = min(len(sentences), i + buffer_size + 1)
        combined.append(" ".join(sentences[lo:hi]))

    embeddings = np.asarray(embed_fn(combined))
    distances = []
    for i in range(len(embeddings) - 1):
        a, b = embeddings[i], embeddings[i + 1]
        sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        distances.append(1.0 - sim)

    threshold = float(np.percentile(distances, breakpoint_percentile))
    groups = []
    current = [sentences[0]]
    for i, dist in enumerate(distances):
        if dist > threshold:
            groups.append(current)
            current = []
        current.append(sentences[i + 1])
    groups.append(current)
    return groups


def chunk_corpus_file(
    source_file: str,
    markdown_text: str,
    embed_fn,
    buffer_size: int = CHUNK_BUFFER_SIZE,
    breakpoint_percentile: float = CHUNK_BREAKPOINT_PERCENTILE,
) -> list:
    """embed_fn: list[str] -> array-like of embeddings, e.g. Embedder.embed_texts.
    Injected rather than imported directly so this module stays unit-testable
    with a cheap deterministic fake instead of loading the real model."""
    category = category_from_filename(source_file)
    chunks: list = []
    for heading, body in _split_sections(markdown_text):
        sentences = _split_sentences(body)
        for group in _semantic_groups(sentences, embed_fn, buffer_size, breakpoint_percentile):
            text = f"## {heading}\n\n" + " ".join(group)
            chunk_id = f"{source_file}::{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}"
            chunks.append(
                Chunk(chunk_id=chunk_id, text=text, source_file=source_file, category=category, heading=heading)
            )
    return chunks
