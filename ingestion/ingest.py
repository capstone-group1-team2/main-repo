"""Orchestrates the full ingestion pipeline end-to-end. Run as:

    python -m ingestion.ingest

Safe to run repeatedly: a whole-file content hash gates chunking/embedding
per document (hash_store.py), so a re-run with no corpus edits skips every
file and processes zero chunks. Only chunks that are actually new or removed
touch Weaviate or Neo4j.

Self-healing: an unchanged file's chunk_ids are also checked against what's
actually present in Weaviate and Neo4j before trusting the fast-path skip.
If either store was reset independently of hash_store.json (e.g. `docker
volume rm` on just one of them), the affected file is reprocessed even
though its content hash still matches 
"""

from __future__ import annotations

import logging
import os

from ingestion.chunker import chunk_corpus_file
from ingestion.config import CORPUS_DIR, HASH_STORE_PATH
from ingestion.embedder import Embedder
from ingestion.graph_builder import GraphBuilder
from ingestion.hash_store import FileDiff, HashStore, file_hash
from ingestion.weaviate_loader import WeaviateLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ingestion.ingest")


def _load_corpus_files() -> dict:
    files = {}
    for name in sorted(os.listdir(CORPUS_DIR)):
        if name.endswith(".md"):
            with open(os.path.join(CORPUS_DIR, name), "r", encoding="utf-8") as f:
                files[name] = f.read()
    return files


def run() -> None:
    embedder = Embedder()
    hash_store = HashStore(HASH_STORE_PATH)
    weaviate_loader = WeaviateLoader()
    graph_builder = GraphBuilder()

    try:
        weaviate_loader.init_schema()
        graph_builder.init_schema()
        graph_builder.seed_category_relations()

        previous_files = hash_store.load().get("files", {})
        corpus_files = _load_corpus_files()

        # Per-file plan: chunk (if needed) and diff everything up front — this
        # part is pure computation, nothing is written to any store yet.
        plans = {}
        for name, raw_text in corpus_files.items():
            expected_ids = set(previous_files.get(name, {}).get("chunk_ids", []))
            content_unchanged = hash_store.file_unchanged(name, raw_text)

            missing_from_weaviate = set()
            missing_from_neo4j = set()
            if content_unchanged and expected_ids:
                missing_from_weaviate = expected_ids - weaviate_loader.chunk_ids_present(expected_ids)
                missing_from_neo4j = expected_ids - graph_builder.chunk_ids_present(expected_ids)
            needs_repair = bool(missing_from_weaviate or missing_from_neo4j)

            if content_unchanged and not needs_repair:
                logger.info(
                    "%s: file content unchanged and already present in Weaviate+Neo4j — "
                    "skipped chunking and embedding entirely.",
                    name,
                )
                continue

            if content_unchanged and needs_repair:
                logger.warning(
                    "%s: file content unchanged but missing downstream (weaviate: %d, neo4j: %d) — "
                    "reprocessing to self-heal (Weaviate/Neo4j reset independently of hash_store.json).",
                    name, len(missing_from_weaviate), len(missing_from_neo4j),
                )

            chunks = chunk_corpus_file(name, raw_text, embedder.embed_texts)
            chunk_ids = [c.chunk_id for c in chunks]
            chunks_by_id = {c.chunk_id: c for c in chunks}

            if content_unchanged:
                # Chunk-id-level diff would show nothing added/removed since
                # the content truly didn't change — the real work is
                # store-repair, using the per-store target sets directly.
                diff = FileDiff(source_file=name, added=frozenset(missing_from_weaviate | missing_from_neo4j))
                weaviate_ids, neo4j_ids = missing_from_weaviate, missing_from_neo4j
            else:
                diff = hash_store.diff_chunks(name, chunk_ids)
                weaviate_ids, neo4j_ids = diff.added, diff.added

            if diff.changed:
                plans[name] = {
                    "diff": diff,
                    "chunks_by_id": chunks_by_id,
                    "weaviate_ids": weaviate_ids,
                    "neo4j_ids": neo4j_ids,
                    "file_hash": file_hash(raw_text),
                    "chunk_ids": chunk_ids,
                }
            else:
                logger.info("%s: no chunk-level changes (%d chunks), skipped.", name, len(diff.unchanged))

        removed_files = [name for name in previous_files if name not in corpus_files]

        if not plans and not removed_files:
            logger.info("No corpus changes detected — 0 chunks processed.")
            return

        version = hash_store.next_version()
        total_added = total_removed = 0

        # Apply phase: each file's stores are written, THEN immediately
        # committed to hash_store.json — never the other way around. If the
        # process dies partway through, only genuinely-completed files are
        # recorded as done; anything after that point is picked up again
        # (as new/changed, or via the self-heal check) on the next run.
        for name, plan in plans.items():
            diff = plan["diff"]
            chunks_by_id = plan["chunks_by_id"]

            weaviate_chunks = [chunks_by_id[cid] for cid in plan["weaviate_ids"]]
            if weaviate_chunks:
                vectors = embedder.embed_texts([c.text for c in weaviate_chunks])
                weaviate_loader.upsert_chunks(weaviate_chunks, vectors, version)

            for cid in plan["neo4j_ids"]:
                c = chunks_by_id[cid]
                graph_builder.link_chunk_to_category(c.chunk_id, c.category, c.source_file, version)

            if diff.removed:
                weaviate_loader.delete_chunks(list(diff.removed))
                for chunk_id in diff.removed:
                    graph_builder.deactivate_chunk(chunk_id)

            hash_store.commit_file(name, plan["file_hash"], plan["chunk_ids"], version)

            total_added += len(diff.added)
            total_removed += len(diff.removed)
            logger.info(
                "%s: +%d added, -%d removed, %d unchanged (version %d).",
                name, len(diff.added), len(diff.removed), len(diff.unchanged), version,
            )

        for name in removed_files:
            diff = hash_store.removed_file_diff(name)
            weaviate_loader.delete_chunks(list(diff.removed))
            for chunk_id in diff.removed:
                graph_builder.deactivate_chunk(chunk_id)
            hash_store.commit_removed_file(name, version)
            total_removed += len(diff.removed)
            logger.info("%s: removed from corpus, -%d chunks deleted (version %d).", name, len(diff.removed), version)

        logger.info(
            "Ingestion run complete: %d chunks added, %d removed, version=%d.",
            total_added, total_removed, version,
        )
    finally:
        weaviate_loader.close()
        graph_builder.close()


if __name__ == "__main__":
    run()
