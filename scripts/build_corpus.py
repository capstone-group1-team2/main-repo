"""HISTORICAL / REFERENCE ONLY — DO NOT RUN AGAINST THE REAL CORPUS.

`data/corpus/*.md` is a hand-finalized, verified
artifact. This script reproduces the *original* generation step that
produced an early, unresolved-placeholder version of those files
(aggregating `data/bitext_customer_support.csv`'s `response` text by
`category` into markdown, `##` section headers per intent) — it does
NOT reproduce the finalized corpus, because the finalized corpus
resolves every `{{placeholder}}` (e.g. `{{Customer Support Phone
Number}}`) into a single consistent set of illustrative facts for the
fictional "Meridian Retail" (see ARCHITECTURE.md §3.2's fact table),
and this script does not perform that resolution.

"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

_DEFAULT_CSV = "data/bitext_customer_support.csv"
_FORBIDDEN_OUT_DIR = "data/corpus"


def _intent_title(intent: str) -> str:
    return intent.replace("_", " ").title()


def build_corpus(csv_path: str, out_dir: str) -> dict:
    if os.path.abspath(out_dir) == os.path.abspath(_FORBIDDEN_OUT_DIR):
        raise SystemExit(
            f"Refusing to write to {_FORBIDDEN_OUT_DIR} — that directory is the "
            "hand-finalized corpus (ARCHITECTURE.md §3.2). Use a different --out-dir."
        )

    by_category = defaultdict(lambda: defaultdict(list))
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_category[row["category"]][row["intent"]].append(row["response"])

    os.makedirs(out_dir, exist_ok=True)
    written = {}
    for category, intents in sorted(by_category.items()):
        lines = [f"# {category.title()}", ""]
        for intent, responses in sorted(intents.items()):
            lines.append(f"## {_intent_title(intent)}")
            lines.append("")
            # One representative (unresolved-placeholder) response per intent —
            # illustrative aggregation, not the finalized corpus's hand-edited prose.
            lines.append(responses[0].strip())
            lines.append("")
        out_path = os.path.join(out_dir, f"{category.lower()}.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        written[category] = out_path
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-path", default=_DEFAULT_CSV, help=f"Source CSV (default: {_DEFAULT_CSV}).")
    parser.add_argument(
        "--out-dir",
        required=True,
        help=f"Output directory. MUST NOT be {_FORBIDDEN_OUT_DIR} (refused at runtime) — "
        "use a scratch/temp directory to inspect this script's historical output.",
    )
    args = parser.parse_args()
    written = build_corpus(args.csv_path, args.out_dir)
    for category, path in written.items():
        print(f"{category} -> {path}")


if __name__ == "__main__":
    main()
