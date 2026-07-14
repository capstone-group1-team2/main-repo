import json
import os
import tempfile

from ingestion.hash_store import HashStore, file_hash


def _store():
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    os.remove(path)  # HashStore.load() must handle a missing file gracefully
    return HashStore(path), path


def test_load_missing_file_returns_empty_state():
    store, path = _store()
    assert store.load() == {"version": 0, "files": {}}


def test_file_unchanged_false_when_never_seen():
    store, path = _store()
    assert store.file_unchanged("cancel.md", "some text") is False


def test_file_unchanged_true_after_matching_save():
    store, path = _store()
    raw_text = "## Heading\n\nSome policy text."
    store.save({"cancel.md": {"file_hash": file_hash(raw_text), "chunk_ids": ["cancel.md::abc"]}}, any_changed=True)
    assert store.file_unchanged("cancel.md", raw_text) is True
    assert store.file_unchanged("cancel.md", raw_text + " edited") is False
    os.remove(path)


def test_diff_chunks_detects_added_removed_unchanged():
    store, path = _store()
    store.save({"cancel.md": {"file_hash": "h1", "chunk_ids": ["c::1", "c::2", "c::3"]}}, any_changed=True)

    diff = store.diff_chunks("cancel.md", ["c::2", "c::3", "c::4"])

    assert diff.added == frozenset({"c::4"})
    assert diff.removed == frozenset({"c::1"})
    assert diff.unchanged == frozenset({"c::2", "c::3"})
    assert diff.changed is True
    os.remove(path)


def test_diff_chunks_no_changes():
    store, path = _store()
    store.save({"cancel.md": {"file_hash": "h1", "chunk_ids": ["c::1", "c::2"]}}, any_changed=True)

    diff = store.diff_chunks("cancel.md", ["c::1", "c::2"])

    assert diff.added == frozenset()
    assert diff.removed == frozenset()
    assert diff.changed is False
    os.remove(path)


def test_removed_file_diff_flags_all_previous_chunks():
    store, path = _store()
    store.save({"cancel.md": {"file_hash": "h1", "chunk_ids": ["c::1", "c::2"]}}, any_changed=True)

    diff = store.removed_file_diff("cancel.md")

    assert diff.removed == frozenset({"c::1", "c::2"})
    assert diff.changed is True
    os.remove(path)


def test_save_bumps_version_only_when_changed():
    store, path = _store()
    v1 = store.save({"a.md": {"file_hash": "h1", "chunk_ids": []}}, any_changed=True)
    v2 = store.save({"a.md": {"file_hash": "h1", "chunk_ids": []}}, any_changed=False)
    v3 = store.save({"a.md": {"file_hash": "h2", "chunk_ids": ["a::1"]}}, any_changed=True)

    assert v1 == 1
    assert v2 == 1
    assert v3 == 2
    os.remove(path)


def test_save_persists_valid_json():
    store, path = _store()
    store.save({"a.md": {"file_hash": "h1", "chunk_ids": ["a::1"]}}, any_changed=True)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["version"] == 1
    assert data["files"]["a.md"]["chunk_ids"] == ["a::1"]
    os.remove(path)


def test_next_version_peeks_without_writing():
    store, path = _store()
    assert store.next_version() == 1
    assert not os.path.exists(path)  # peeking must not create the file

    store.commit_file("a.md", "h1", ["a::1"], version=1)
    assert store.next_version() == 2
    os.remove(path)


def test_commit_file_only_updates_that_file():
    store, path = _store()
    store.commit_file("a.md", "ha", ["a::1"], version=1)
    store.commit_file("b.md", "hb", ["b::1"], version=1)

    state = store.load()
    assert state["version"] == 1
    assert state["files"]["a.md"] == {"file_hash": "ha", "chunk_ids": ["a::1"]}
    assert state["files"]["b.md"] == {"file_hash": "hb", "chunk_ids": ["b::1"]}
    os.remove(path)


def test_commit_file_reflects_partial_run_correctly():
    """If a run dies after committing file A but before touching file B,
    the on-disk state must show A as done and B as untouched — not both, and
    not neither. This is what makes a crash mid-run safely resumable."""
    store, path = _store()
    store.commit_file("a.md", "ha", ["a::1"], version=1)
    # Simulate a crash: file B's chunks were never written to Weaviate/Neo4j
    # and commit_file("b.md", ...) never ran.

    state = store.load()
    assert "a.md" in state["files"]
    assert "b.md" not in state["files"]
    assert store.file_unchanged("a.md", "irrelevant, only hash comparison matters") is False
    os.remove(path)


def test_commit_removed_file_deletes_entry():
    store, path = _store()
    store.commit_file("a.md", "ha", ["a::1"], version=1)
    store.commit_removed_file("a.md", version=2)

    state = store.load()
    assert "a.md" not in state["files"]
    assert state["version"] == 2
    os.remove(path)
