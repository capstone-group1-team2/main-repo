"""Scoring functions for the evaluation harness.
 
Every function here is PURE: it takes already-collected per-example results
and returns a number or a dict. Nothing in this module calls Groq, touches
Weaviate, or reads a file — which is what makes the scoring logic testable
on its own (see `eval/test_metrics.py`) without standing up the stack.
 
Vocabulary used throughout: a "row" is one scored example, produced by
`eval/run_eval.py`, shaped like:
 
    {
      "id": "hold-0001",
      "expected_route": "information",
      "predicted_route": "information",
      "grounded": True | False | None,         # post-hoc, vs first-pass chunks
      "answer_grounded": True | False | None,  # the Criterion-2 metric; see below
      "reason": "ungrounded_after_retry" | None,   # escalation reason, if escalated
      "recall_hit": True | False | None,       # None = no gold source to check
      "answer_match": True | False | None,     # None = not applicable
      "first_turn_latency_s": 1.83,
      "total_latency_s": 1.83,
      "metadata": {"category": "CANCEL", "difficulty": "easy", ...},
    }
 
`None` means "this metric does not apply to this example" and is EXCLUDED
from that metric's denominator — never silently counted as a failure. Each
metric therefore reports its own `n` alongside its value, so a reader can
always see how many examples a number was actually computed over.
 
--------------------------------------------------------------------------
WHY THERE ARE TWO GROUNDEDNESS FIELDS
--------------------------------------------------------------------------
`agent.py::_answer_information` only ever emits an information-route answer if
that answer ALREADY passed `is_grounded()` at request time. Anything that fails
is retried once via `broaden_via_graph()`, and if it still fails it is removed
from the information route entirely and becomes an escalation
(`reason="ungrounded_after_retry"`).
 
So "groundedness measured over information-route answers" is close to VACUOUS:
the ungrounded answers have been selected out of the denominator. It measures
the runtime guard's self-consistency, not the RAG pipeline's quality. It would
read ~1.00 no matter how bad retrieval was.
 
  `grounded`         -> the vacuous one. Kept as a conservative cross-check
                        only. Scored post-hoc against FIRST-PASS chunks, so an
                        answer that a graph-broadened retry rescued is scored
                        against the wrong references and may read False. It is
                        depressed by that artifact and inflated by the selection
                        effect above, in unknown proportion. Do not report it as
                        Criterion 2.
 
  `answer_grounded`  -> the real one. Puts the ungrounded escalations BACK into
                        the denominator:
 
                            grounded answers
                        ----------------------------------
                        grounded answers + ungrounded_after_retry escalations
 
                        "Of every question the agent TRIED to answer from the
                        corpus, what fraction produced an answer the corpus
                        actually supports?" That is what a stakeholder means by
                        Criterion 2, and it can genuinely fall below 0.90.
 
For the baseline the two coincide: it never escalates, so every question is an
attempt, and `answer_grounded` is just its post-hoc groundedness. That makes the
head-to-head comparison fair and meaningful.
"""
 
from __future__ import annotations
 
import math
from collections import defaultdict
 
# The route the escalation precision/recall metrics treat as the positive class.
ESCALATE = "escalate"
 
 
# ---------------------------------------------------------------------------
# Primary metric
# ---------------------------------------------------------------------------
 
 
def routing_accuracy(rows: list) -> dict:
    """PRIMARY METRIC. Fraction of examples whose terminal route matches the
    hand-labeled `expected_route`.
 
    "Terminal" matters: the agent's first reply to an unclear message is a
    clarifying question (route="information"), and it only escalates if the
    follow-up is still unclear. Scoring the first turn alone would mark every
    escalate example wrong. `run_eval.py` drives each example to the route the
    customer would actually end up at — see its `run_agent_to_terminal()`.
    """
    if not rows:
        return {"value": 0.0, "n": 0}
    correct = sum(1 for r in rows if r["predicted_route"] == r["expected_route"])
    return {"value": correct / len(rows), "n": len(rows)}
 
 
# ---------------------------------------------------------------------------
# Secondary metrics
# ---------------------------------------------------------------------------
 
 
def escalation_prf(rows: list) -> dict:
    """Precision / recall / F1 for the ESCALATE class specifically.
 
    Routing accuracy alone hides the failure that actually matters here: this
    held-out set is 75% `information`, so a system that NEVER escalates still
    scores 0.75. Escalation recall is what separates "knows when to stop" from
    "always guesses". Precision guards the opposite failure — escalating on
    questions it should have simply answered, which is expensive in a real
    support org because a human gets paged for nothing.
 
    Precision is None (not 0.0) when the system made zero escalate
    predictions: 0/0 is undefined, and reporting it as 0.0 would understate a
    baseline that simply never plays the class. The plain-RAG baseline hits
    this case by construction.
    """
    tp = sum(1 for r in rows if r["predicted_route"] == ESCALATE and r["expected_route"] == ESCALATE)
    fp = sum(1 for r in rows if r["predicted_route"] == ESCALATE and r["expected_route"] != ESCALATE)
    fn = sum(1 for r in rows if r["predicted_route"] != ESCALATE and r["expected_route"] == ESCALATE)
 
    predicted_positives = tp + fp
    actual_positives = tp + fn
 
    precision = tp / predicted_positives if predicted_positives else None
    recall = tp / actual_positives if actual_positives else None
 
    if precision and recall and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0 if (precision is not None and recall is not None) else None
 
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_actual_escalate": actual_positives,
    }
 
 
def groundedness_rate(rows: list) -> dict:
    """Fraction of GENERATED ANSWERS that are supported by the retrieved
    chunks (cosine similarity >= GROUNDEDNESS_SIMILARITY_THRESHOLD).
 
    Denominator is only the examples where the system actually generated a
    free-text answer from retrieval. Action results and escalation notices are
    templated strings, not generated prose — scoring them for groundedness
    against policy chunks would be meaningless, so `run_eval.py` sets
    `grounded=None` for them and they're excluded here.
    """
    applicable = [r for r in rows if r["grounded"] is not None]
    if not applicable:
        return {"value": None, "n": 0}
    return {"value": sum(1 for r in applicable if r["grounded"]) / len(applicable), "n": len(applicable)}
 
 
def recall_at_k(rows: list) -> dict:
    """Fraction of examples where the correct source document appears anywhere
    in the top-k retrieved chunks.
 
    Gold label is `metadata.expected_source_file` (e.g. "cancel.md"). Only
    information-route examples carry one — action and escalate examples have no
    "correct document" to retrieve, so they're excluded (`recall_hit=None`).
    """
    applicable = [r for r in rows if r["recall_hit"] is not None]
    if not applicable:
        return {"value": None, "n": 0}
    return {"value": sum(1 for r in applicable if r["recall_hit"]) / len(applicable), "n": len(applicable)}
 
 
def answer_attempt_groundedness(rows: list) -> dict:
    """**This is the Criterion 2 metric.** See the module docstring.
 
    Denominator = every example where the system ATTEMPTED to answer from the
    corpus, which for the agent means (a grounded answer it emitted) OR (an
    escalation it raised precisely because it could not ground an answer).
 
    A note on the numerator's provenance, because it matters: for the agent, a
    `True` comes from the agent's own runtime `is_grounded()` verdict, not from
    a second check here. That is deliberate, not laziness — the runtime check
    ran against the chunks the answer was actually generated from (post-retry,
    post-graph-broadening), which this harness does not have. Re-checking it
    here against first-pass chunks would be strictly less accurate. Both use the
    identical function and the identical threshold.
 
    For the baseline, `True` comes from this harness's post-hoc check. The
    baseline has no retry, so its first-pass chunks ARE the chunks its answer was
    generated from, and the two are equivalent.
    """
    applicable = [r for r in rows if r.get("answer_grounded") is not None]
    if not applicable:
        return {"value": None, "n": 0}
    grounded = sum(1 for r in applicable if r["answer_grounded"])
    return {"value": grounded / len(applicable), "n": len(applicable), "grounded": grounded}
 
 
def escalation_reasons(rows: list) -> dict:
    """Counts of why the system escalated. Agent-only in practice — the baseline
    has no escalation path, so this is always empty for it.
 
    Reading this breakdown is how you tell a *healthy* escalation rate from an
    unhealthy one. `ungrounded_after_retry` means retrieval or the corpus is
    weak. `order_not_found` on an order that plainly exists means a lookup bug.
    `still_unclear_after_clarification_low_retrieval_relevance` on short, easy
    questions means an escalation floor is mis-calibrated. Same headline number,
    three completely different engineering responses.
    """
    counts: dict = defaultdict(int)
    for row in rows:
        if row["predicted_route"] == ESCALATE and row.get("reason"):
            counts[row["reason"]] += 1
    return dict(counts)
 
 
# ---------------------------------------------------------------------------
# Latency (Criterion 3)
# ---------------------------------------------------------------------------
 
LATENCY_TARGET_SECONDS = 3.0
 
 
def latency_stats(values: list, target_seconds: float = LATENCY_TARGET_SECONDS) -> dict:
    """p50 / p95 / max over raw per-request latencies, plus the fraction under
    the target.
 
    Percentiles, not a mean. The agent has a fast path (classify + generate) and
    a slow path (classify + generate + broaden_via_graph + generate again). The
    slow path is roughly double. A mean averages the two into a number that
    describes no actual request. The p95 is where a 3-second budget actually
    breaks, and it is the only number worth putting on a slide.
    """
    usable = sorted(v for v in values if v is not None)
    if not usable:
        return {"p50": None, "p95": None, "max": None, "mean": None, "pct_under_target": None, "n": 0}
 
    def pct(p: float) -> float:
        # Linear interpolation between closest ranks, matching numpy's default.
        if len(usable) == 1:
            return usable[0]
        idx = (len(usable) - 1) * p
        lo, hi = int(idx), min(int(idx) + 1, len(usable) - 1)
        return usable[lo] + (usable[hi] - usable[lo]) * (idx - lo)
 
    return {
        "p50": pct(0.50),
        "p95": pct(0.95),
        "max": usable[-1],
        "mean": sum(usable) / len(usable),
        "pct_under_target": sum(1 for v in usable if v < target_seconds) / len(usable),
        "n": len(usable),
    }
 
 
def answer_match_rate(rows: list) -> dict:
    """Diagnostic (not in the proposal's metric list, reported alongside):
    fraction of generated answers semantically matching the hand-written
    `expected_answer` above ANSWER_MATCH_THRESHOLD.
 
    Deliberately NOT exact string matching — Bitext's responses are
    paraphrase-heavy and two correct answers can share almost no tokens.
    """
    applicable = [r for r in rows if r["answer_match"] is not None]
    if not applicable:
        return {"value": None, "n": 0}
    return {"value": sum(1 for r in applicable if r["answer_match"]) / len(applicable), "n": len(applicable)}
 
 
# ---------------------------------------------------------------------------
# Error analysis (R5): group errors across >= 2 dimensions
# ---------------------------------------------------------------------------
 
 
def error_grid(rows: list, dim1: str = "category", dim2: str = "difficulty") -> dict:
    """Per-cell routing error rate across two metadata dimensions.
 
    Returns {(dim1_value, dim2_value): {correct, wrong, error_rate, n}}.
    Cells with no examples simply don't appear — an empty cell and a
    zero-error cell are different things and shouldn't render identically.
    """
    grid: dict = defaultdict(lambda: {"correct": 0, "wrong": 0})
    for row in rows:
        key = (row["metadata"].get(dim1, "unknown"), row["metadata"].get(dim2, "unknown"))
        bucket = "correct" if row["predicted_route"] == row["expected_route"] else "wrong"
        grid[key][bucket] += 1
 
    out = {}
    for key, counts in grid.items():
        n = counts["correct"] + counts["wrong"]
        out[key] = {**counts, "n": n, "error_rate": counts["wrong"] / n if n else 0.0}
    return out
 
 
def format_error_grid(grid: dict, dim1: str = "category", dim2: str = "difficulty") -> str:
    """Renders the error grid as a fixed-width table: rows = dim1, cols = dim2.
    Each cell shows `error_rate (wrong/n)`; empty cells show a dash."""
    row_keys = sorted({k[0] for k in grid})
    col_keys = sorted({k[1] for k in grid}, key=lambda d: {"easy": 0, "medium": 1, "hard": 2}.get(d, 99))
 
    width = max([len(str(r)) for r in row_keys] + [len(dim1)]) + 2
    header = f"{dim1:<{width}}" + "".join(f"{c:>16}" for c in col_keys) + f"{'ROW TOTAL':>16}"
    lines = [header, "-" * len(header)]
 
    for r in row_keys:
        cells = []
        row_wrong = row_n = 0
        for c in col_keys:
            cell = grid.get((r, c))
            if not cell:
                cells.append(f"{'-':>16}")
                continue
            row_wrong += cell["wrong"]
            row_n += cell["n"]
            cells.append(f"{cell['error_rate']:.2f} ({cell['wrong']}/{cell['n']})".rjust(16))
        total = f"{row_wrong / row_n:.2f} ({row_wrong}/{row_n})" if row_n else "-"
        lines.append(f"{str(r):<{width}}" + "".join(cells) + total.rjust(16))
 
    return "\n".join(lines)
 
 
# ---------------------------------------------------------------------------
# Multi-seed aggregation (R4)
# ---------------------------------------------------------------------------
 
 
def mean_std(values: list) -> dict:
    """Population mean and standard deviation across seeds.
 
    Population (n), not sample (n-1): these three seeds ARE the full set of
    runs being reported, not a sample drawn from a larger population of runs.
    `None` entries (an undefined metric, e.g. baseline escalation precision)
    are dropped; if nothing is left, mean/std are None rather than 0.0 — a
    metric that couldn't be computed is not a metric that scored zero.
    """
    usable = [v for v in values if v is not None]
    if not usable:
        return {"mean": None, "std": None, "n_seeds": 0, "values": values}
    mean = sum(usable) / len(usable)
    variance = sum((v - mean) ** 2 for v in usable) / len(usable)
    return {"mean": mean, "std": math.sqrt(variance), "n_seeds": len(usable), "values": values}
 
 
def summarize(rows: list) -> dict:
    """All metrics for one system on one seed's run."""
    return {
        "routing_accuracy": routing_accuracy(rows),
        "escalation": escalation_prf(rows),
        "answer_attempt_groundedness": answer_attempt_groundedness(rows),  # Criterion 2
        "groundedness_rate": groundedness_rate(rows),  # conservative cross-check only
        "recall_at_k": recall_at_k(rows),
        "answer_match_rate": answer_match_rate(rows),
        "escalation_reasons": escalation_reasons(rows),
    }
 
 
# ---------------------------------------------------------------------------
# Success criteria (from the Project Proposal)
# ---------------------------------------------------------------------------
 
ROUTING_ACCURACY_TARGET = 0.85
GROUNDEDNESS_TARGET = 0.90
 
 
def check_success_criteria(agent_agg: dict, agent_latency: dict) -> list:
    """Evaluates the three stated success criteria against the aggregated run.
 
    Each returns (name, target, observed, passed). `passed` is None when the
    metric could not be computed — an unmeasurable criterion is not a failed
    one, and must not be silently rendered as a red X.
 
    Criterion 1 and 2 are judged on the MEAN across seeds. A stricter reading
    would require mean − 1 stddev to clear the bar; we report the stddev right
    next to it so a reader can apply that judgement themselves.
    """
    results = []
 
    routing = agent_agg["routing_accuracy"]["mean"]
    results.append(
        (
            "C1  Routing accuracy >= 0.85",
            ROUTING_ACCURACY_TARGET,
            routing,
            None if routing is None else routing >= ROUTING_ACCURACY_TARGET,
        )
    )
 
    grounded = agent_agg["answer_attempt_groundedness"]["mean"]
    results.append(
        (
            "C2  Groundedness >= 0.90",
            GROUNDEDNESS_TARGET,
            grounded,
            None if grounded is None else grounded >= GROUNDEDNESS_TARGET,
        )
    )
 
    p95 = agent_latency.get("p95")
    results.append(
        (
            "C3  Latency p95 < 3.0s (optional)",
            LATENCY_TARGET_SECONDS,
            p95,
            None if p95 is None else p95 < LATENCY_TARGET_SECONDS,
        )
    )
    return results
 