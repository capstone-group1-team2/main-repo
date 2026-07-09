"""Tracks per-document content hashes and chunk_ids across ingestion runs.

A whole-file content hash is checked FIRST, before any chunking or embedding
happens, so a genuinely unchanged document is skipped entirely rather than
merely re-chunked-then-discarded. Only files whose content actually changed
get their chunk_ids diffed against the previous run to find which specific
chunks were added or removed.

State persists as JSON at HASH_STORE_PATH:
    {"version": int, "files": {source_file: {"file_hash": str, "chunk_ids": [str, ...]}}}
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

from ingestion.config import HASH_STORE_PATH


def file_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FileDiff:
    source_file: str
    added: frozenset = frozenset()
    removed: frozenset = frozenset()
    unchanged: frozenset = frozenset()

    @property
    def changed(self) -> bool:
        return bool(self.added or self.removed)


class HashStore:
    def __init__(self, path: str = HASH_STORE_PATH):
        self.path = path

    def load(self) -> dict:
        if not os.path.exists(self.path):
            return {"version": 0, "files": {}}
        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)

    def file_unchanged(self, source_file: str, raw_text: str) -> bool:
        """True if `source_file`'s whole-content hash matches the last run's.
        Callers should skip chunking/embedding entirely when this is True."""
        previous = self.load().get("files", {}).get(source_file)
        return previous is not None and previous.get("file_hash") == file_hash(raw_text)

    def diff_chunks(self, source_file: str, current_chunk_ids: list) -> FileDiff:
        """Diffs a changed-or-new file's freshly computed chunk_ids against
        the previous run's chunk_ids for that same file."""
        previous_ids = set(self.load().get("files", {}).get(source_file, {}).get("chunk_ids", []))
        current = set(current_chunk_ids)
        return FileDiff(
            source_file=source_file,
            added=frozenset(current - previous_ids),
            removed=frozenset(previous_ids - current),
            unchanged=frozenset(current & previous_ids),
        )

    def removed_file_diff(self, source_file: str) -> FileDiff:
        """For a file present in the previous run but absent from the corpus
        directory now — all of its previous chunk_ids are removed."""
        previous_ids = self.load().get("files", {}).get(source_file, {}).get("chunk_ids", [])
        return FileDiff(source_file=source_file, removed=frozenset(previous_ids))

    def save(self, files_state: dict, any_changed: bool) -> int:
        """Bulk write, used only for the no-op path (nothing changed this
        run, so there is no downstream write to be crash-inconsistent
        with). For a real ingestion run, use next_version()/commit_file()/
        commit_removed_file() instead — see their docstrings for why."""
        previous = self.load()
        new_version = previous.get("version", 0) + (1 if any_changed else 0)
        new_state = {"version": new_version, "files": files_state}
        self._write(new_state)
        return new_version

    def next_version(self) -> int:
        """Peeks at the version this run will use, without writing anything."""
        return self.load().get("version", 0) + 1

    def commit_file(self, source_file: str, file_hash_value: str, chunk_ids: list, version: int) -> None:
        """Persists ONE file's state immediately after its chunks have
        actually been written to Weaviate AND Neo4j — not before, and not
        batched with other files. This is what keeps hash_store.json crash-
        consistent with reality: if the process dies partway through a run,
        only genuinely-completed files are recorded as done, so the next run
        (or the self-heal check) correctly picks up wherever it left off,
        rather than the cache prematurely claiming every file succeeded."""
        state = self.load()
        state.setdefault("files", {})[source_file] = {"file_hash": file_hash_value, "chunk_ids": chunk_ids}
        state["version"] = version
        self._write(state)

    def commit_removed_file(self, source_file: str, version: int) -> None:
        """Persists the removal of a file's entry immediately after its
        chunks have actually been deleted from Weaviate AND Neo4j."""
        state = self.load()
        state.setdefault("files", {}).pop(source_file, None)
        state["version"] = version
        self._write(state)

    def _write(self, state: dict) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, sort_keys=True)
