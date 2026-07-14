# Dataset Card — Meridian Retail Customer Support Corpus

This card covers two related artifacts: the raw source dataset (`data/bitext_customer_support.csv`) and the corpus derived from it (`data/corpus/*.md`), which is what the agent actually retrieves from at runtime.

## Dataset Summary

The agent's knowledge base is built from the **Bitext Customer Support LLM Chatbot Training Dataset**, a synthetic, template-based dataset of customer support exchanges. The raw dataset's template placeholders (e.g. `{{Customer Support Phone Number}}`) were resolved into a single, consistent set of facts for a fictional company, "Meridian Retail," and reorganized into 11 category-level markdown policy documents that the ingestion pipeline chunks, embeds, and indexes for retrieval.

## Source Data

| | |
|---|---|
| **Name** | Bitext Customer Support LLM Chatbot Training Dataset |
| **Publisher** | Bitext Innovations International, S.L. |
| **Source** | Hugging Face Hub (`bitext/Bitext-customer-support-llm-chatbot-training-dataset`) |
| **License** | Community Data License Agreement – Sharing, version 1.0 (CDLA-Sharing-1.0) |
| **Size** | 26,872 rows |
| **Categories** | 11 (ACCOUNT, CANCEL, CONTACT, DELIVERY, FEEDBACK, INVOICE, ORDER, PAYMENT, REFUND, SHIPPING, SUBSCRIPTION) |
| **Intents** | 27, hardcoded in `agent/intent_classifier.py`'s `SUPPORT_INTENTS` after verifying against the actual CSV column, not assumed |
| **Columns** | `flags`, `instruction`, `category`, `intent`, `response` |
| **Download** | `scripts/fetch_data.py` — reproducible re-download from the official source; the checked-in CSV should always match what this script would produce |

**Attribution:** required by CDLA-Sharing-1.0 and provided in full in `NOTICE.md`. This project's own code is separately licensed under MIT (`LICENSE`) — the two licenses cover different things and neither supersedes the other.

The `flags` column encodes Bitext's own tagging scheme for phrasing style (politeness, colloquialism, noise/typos, etc.) and is not consumed by any code in this project — it's part of the source format, not something this project defines.

## From raw dataset to corpus — what actually changed

The raw dataset is **never edited**. It is aggregated and transformed into the retrieval corpus in two stages:

1. **`scripts/build_corpus.py` (historical/reference only — do not run against the real corpus)** — reproduces the *original* generation step: `response` text grouped by `category`, with `##` section headers per intent, producing an early version with unresolved template placeholders still in place. This script's output is **not** what the agent uses.
2. **Hand-finalization** — the placeholder-laden output was manually resolved into a consistent, internally-coherent set of facts for a single fictional company ("Meridian Retail") and saved as the 11 files in `data/corpus/`. These files are the source of truth; `build_corpus.py` is not re-run over them.

| Fact | Value |
|---|---|
| Website | www.meridianretail.com |
| Phone | 1-800-555-0199 |
| Support hours | Monday–Friday, 9 AM–7 PM EST |
| Standard shipping | 3–5 business days |
| Expedited shipping | 1–2 business days |
| Refund window | 30 days from delivery |
| Cancellation fee | Free within 1 hour of order; free before shipping; 10% restocking fee after shipping, before delivery; unavailable after delivery |

Per-transaction values (order number, invoice number, refund amount) were deliberately **not** hardcoded — they're phrased as dynamic references ("your order number") since a real system would substitute these per customer.

## Ingestion into the retrieval system

Each corpus document is split into semantically coherent chunks (heading-first, then embedding-similarity boundaries within a section — see `ingestion/chunker.py`), embedded with `BAAI/bge-large-en-v1.5`, and stored in Weaviate. Category labels also seed a small concept graph in Neo4j (`ingestion/graph_seed.py`'s `CATEGORY_RELATIONS`), used to broaden retrieval on a failed groundedness check. The pipeline is incremental — a content hash gates re-processing, so an edit to one file only reprocesses that file's changed chunks.

## Held-out evaluation set

`eval/heldout.jsonl` (60 examples) is a separately hand-authored evaluation set, not sampled from the raw Bitext data — it includes hand-written adversarial examples (off-topic-but-fluent questions, unintelligible input) specifically because the raw dataset is too clean to naturally contain cases that should escalate. See `eval/validate_heldout.py` for its integrity constraints.

## Known Limitations and Biases

- **Synthetic, template-based source data.** The underlying Bitext responses were generated from templates, not real customer service transcripts — phrasing is often repetitive and formulaic (many near-duplicate responses per intent), and tone varies inconsistently between clinical and overly effusive across otherwise-identical intents.
- **Uneven corpus depth by category.** Some categories are substantially thinner than others — `subscription.md` in particular has minimal concrete factual content relative to categories like `order.md` or `cancel.md`. This has a measured effect: in real eval runs, SUBSCRIPTION was the worst-performing category (6 of 15 examples wrong), and root-caused to corpus thinness causing low-confidence classification, not a retrieval or generation defect. See `eval/failure_cases.md` for the full analysis.
- **No real customer data.** All examples are synthetic. This dataset should not be treated as representative of real customer language, real complaint distributions, or real edge cases beyond what's been deliberately hand-added to the held-out set.
- **Single fictional company, single locale.** Facts (hours, fees, shipping windows) are specific to one invented company profile and are not meant to generalize to any real retailer's actual policies.
- **English only.** No multilingual coverage.

## Considerations for Use

- This corpus is intended for a demonstration/capstone retrieval-augmented agent, not as a general-purpose customer support knowledge base.
- Because the source data is synthetic and template-derived, this dataset is not suitable for evaluating real-world customer support quality without substantial additional real-world data.
- The mock order store (`data/mock_orders.json`) referenced by the agent's action intents is a small, explicitly fabricated dataset for demo purposes and is disclosed as such — it is not connected to any real order system.
