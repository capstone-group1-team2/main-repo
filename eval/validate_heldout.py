"""Pre-freeze integrity check for eval/heldout.jsonl.
 
Run this BEFORE freezing the set, and again in CI afterwards to prove it hasn't
been edited in a way that breaks the harness's assumptions:
 
    python -m eval.validate_heldout
 
Pure stdlib — no Groq key, no Docker stack, no pydantic required, so it runs
anywhere (including a git pre-commit hook). Exits non-zero on any failure.
 
What it enforces, and why each one exists:
 
  - unique, contiguous ids           the guide forbids renumbering post-freeze;
                                     a duplicate id silently overwrites results
  - >= 50 examples                   rubric R3's hard floor
  - every route label is valid       a typo'd route can never be matched, so it
                                     would silently subtract from accuracy
  - all 3 routes present             a held-out set with no escalate examples
                                     proves nothing about the escalation logic
  - metadata dims on every example   no metadata -> no error grid (guide's
                                     "no metadata on the held-out set" pitfall)
  - gold source iff information      recall@k needs a gold document; action and
                                     escalate examples have none, and silently
                                     scoring them would corrupt the metric
  - referenced corpus files exist    a typo'd "cancle.md" would score recall@k
                                     as 0% forever, looking like a retrieval bug
"""
 
from __future__ import annotations
 
import json
import sys
from collections import Counter
from pathlib import Path
 
HELDOUT_PATH = Path("eval/heldout.jsonl")
CORPUS_DIR = Path("data/corpus")
 
VALID_ROUTES = {"information", "action", "escalate"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}
MIN_EXAMPLES = 50
 
 
def validate(heldout_path: Path = HELDOUT_PATH, corpus_dir: Path = CORPUS_DIR) -> list:
    """Returns a list of human-readable problems. Empty list == the set is sound."""
    problems = []
    rows = [json.loads(line) for line in heldout_path.open(encoding="utf-8") if line.strip()]
 
    if len(rows) < MIN_EXAMPLES:
        problems.append(f"only {len(rows)} examples; rubric R3 requires >= {MIN_EXAMPLES}")
 
    ids = [r["id"] for r in rows]
    duplicates = [i for i, count in Counter(ids).items() if count > 1]
    if duplicates:
        problems.append(f"duplicate ids: {duplicates}")
 
    expected_ids = [f"hold-{i:04d}" for i in range(1, len(rows) + 1)]
    if ids != expected_ids:
        problems.append("ids are not contiguous hold-0001..hold-NNNN in file order")
 
    routes = Counter()
    for row in rows:
        rid = row["id"]
 
        for field in ("input", "expected_answer", "expected_route", "metadata"):
            if field not in row:
                problems.append(f"{rid}: missing required field {field!r}")
 
        route = row.get("expected_route")
        routes[route] += 1
        if route not in VALID_ROUTES:
            problems.append(f"{rid}: invalid expected_route {route!r}")
 
        if not str(row.get("input", "")).strip():
            problems.append(f"{rid}: empty input")
 
        meta = row.get("metadata", {})
        if not meta.get("category"):
            problems.append(f"{rid}: missing metadata.category (error grid needs it)")
        if meta.get("difficulty") not in VALID_DIFFICULTIES:
            problems.append(f"{rid}: metadata.difficulty must be one of {sorted(VALID_DIFFICULTIES)}")
 
        gold = meta.get("expected_source_file")
        if route == "information":
            if not gold:
                problems.append(f"{rid}: information examples need metadata.expected_source_file for recall@k")
            elif not (corpus_dir / gold).exists():
                problems.append(f"{rid}: expected_source_file {gold!r} does not exist in {corpus_dir}/")
        elif gold:
            problems.append(f"{rid}: {route} examples must have expected_source_file=null (nothing to retrieve)")
 
    for required_route in VALID_ROUTES:
        if routes[required_route] == 0:
            problems.append(f"no examples with expected_route={required_route!r} — that path is untested")
 
    return problems
 
 
def main() -> None:
    if not HELDOUT_PATH.exists():
        print(f"FAIL: {HELDOUT_PATH} not found (run from the repo root).")
        sys.exit(1)
 
    rows = [json.loads(line) for line in HELDOUT_PATH.open(encoding="utf-8") if line.strip()]
    problems = validate()
 
    print(f"Held-out set: {len(rows)} examples in {HELDOUT_PATH}\n")
    print("  routes:      ", dict(Counter(r["expected_route"] for r in rows)))
    print("  categories:  ", dict(Counter(r["metadata"]["category"] for r in rows)))
    print("  difficulty:  ", dict(Counter(r["metadata"]["difficulty"] for r in rows)))
 
    n_info = sum(1 for r in rows if r["expected_route"] == "information")
    print(f"\n  Majority-class ('always information') routing accuracy: {n_info / len(rows):.4f}")
    print("  ^ this is exactly what the plain-RAG baseline scores. The agent must beat it.\n")
 
    if problems:
        print(f"FAIL — {len(problems)} problem(s):")
        for problem in problems:
            print(f"  - {problem}")
        sys.exit(1)
 
    print("PASS — held-out set is internally consistent and safe to freeze.")
 
 
if __name__ == "__main__":
    main()
 