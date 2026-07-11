"""Evaluation harness for the AI.SPIRE Capstone — agentic customer support assistant.
 
Runs the full agent and the plain-RAG baseline (eval/baseline.py) against the
frozen held-out set, computes the primary metric plus secondaries, aggregates
across 3 seeds, and prints an error grid.
 
PRIMARY METRIC
    Routing accuracy — the fraction of held-out examples where the agent's
    terminal route (information / action / escalate) matches the hand-labeled
    expected_route.
 
Run it from the repo root:
 
    python -m eval.run_eval                       # full 3-seed run
    python -m eval.run_eval --limit 5             # quick smoke test
    python -m eval.run_eval --seeds 42            # single seed
    python -m eval.run_eval --skip-baseline       # agent only, saves Groq quota
 
Requires the local Docker stack (Weaviate, Neo4j, Redis) to be up and
`python -m ingestion.ingest` to have run at least once.
 
 
-------------------------------------------------------------------------------
Two design decisions that materially affect the numbers. Both are deliberate.
-------------------------------------------------------------------------------
 
1. TERMINAL ROUTE, NOT FIRST-TURN ROUTE.
 
   `agent.py` (M9) does not escalate on the first unclear message. It asks ONE
   clarifying question — returning route="information" and setting
   pending="awaiting_clarification" — and escalates only if the reply is still
   unclear. Likewise an ACTION intent with no order number first asks for the
   slot (pending="awaiting_slot").
 
   So a single `handle_message()` call CANNOT measure routing accuracy: every
   example whose expected_route is "escalate" would return "information" on
   turn 1 and be scored wrong. A harness written that way would report ~0%
   escalation recall on a system whose escalation logic works correctly.
 
   `run_agent_to_terminal()` therefore drives each example through the same
   round-trips a real customer would, until the agent reaches a state it will
   not ask another question from, and scores THAT route. The simulated replies
   are intentionally uninformative (see `_reply_for_pending`), which is the
   correct adversarial assumption: we measure what the agent does when the
   customer cannot clarify further, not when they helpfully rescue it.
 
2. SIDE-EFFECT ISOLATION.
 
   A real escalation writes a row to SQLite and POSTs to Slack. Running 60
   examples x 3 seeds against the production handler would spam the team's
   Slack channel with ~24 fake escalations per run and pollute the real
   escalation store that `GET /escalations` reads.
 
   The harness therefore constructs `EscalationHandler` with its own SQLite
   path (`eval/eval_escalations.db`) and an empty webhook URL. This is not a
   mock: it is the real class, running its real contact-capture and packet
   logic, pointed at a scratch store. `escalation.py` already treats an empty
   webhook as "skip the POST and log it", so no Slack traffic is generated.
   The agent's behavior is unchanged; only where its records land differs.
"""
 
from __future__ import annotations
 
import argparse
import datetime
import json
import random
import time
import uuid
from pathlib import Path
 
import numpy as np
 
from eval import metrics as M
 
# NOTE: the agent/, app/, ingestion/ and retrieval/ imports are deliberately
# NOT at module scope. Importing `app.config` raises if GROQ_API_KEY is unset,
# and importing `ingestion.embedder` loads a ~1.3GB model. Keeping them inside
# the functions that need them means `run_agent_to_terminal()` — the subtlest
# logic in this file — can be unit-tested against a fake agent with no Redis,
# no Weaviate, no Neo4j and no Groq key. See eval/test_metrics.py.
 
DEFAULT_SEEDS = [42, 1337, 2024]
 
# Escalations produced by the harness land here, never in the real store.
EVAL_ESCALATION_DB = "eval/eval_escalations.db"
 
# The agent hard-caps every clarify/slot/contact ask at exactly one (§8.1), so
# a conversation provably terminates within 2 turns. 4 is a safety net against
# an infinite loop if that invariant is ever broken by a future change — if it
# ever trips, that is itself a bug worth surfacing loudly.
MAX_TURNS = 4
 
 
# ---------------------------------------------------------------------------
# Seeding (R4)
# ---------------------------------------------------------------------------
 
 
def set_all_seeds(seed: int) -> None:
    """Seeds every source of randomness we control.
 
    NOTE on Groq: `agent/llm_client.py` does not expose a `seed` parameter on
    its `complete()` call, so we cannot pin the LLM's sampling. We do not treat
    this as a defect to work around. The three runs are therefore genuinely
    independent samples of the model's nondeterminism, which is the more
    conservative measurement — a seeded LLM would understate real run-to-run
    variance, and run-to-run variance is exactly what mean +/- stddev exists to
    report. Retrieval is fully deterministic given the frozen, versioned corpus.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch is a transitive dep of sentence-transformers; guard anyway
        import torch
 
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
 
 
# ---------------------------------------------------------------------------
# System construction
# ---------------------------------------------------------------------------
 
 
def build_systems():
    """Constructs the real agent and the real baseline, sharing one embedder,
    one retriever, and one LLM client — the same objects, so any difference in
    their scores comes from the agent's decision logic, never from a
    differently-configured retriever or model."""
    import app.config as cfg
    from agent.agent import Agent
    from agent.escalation import EscalationHandler
    from agent.llm_client import LLMClient
    from agent.session_state import SessionStateStore
    from eval.baseline import RagBaseline
    from ingestion.embedder import Embedder
    from retrieval.graph_retriever import GraphRetriever
    from retrieval.hybrid_retriever import HybridRetriever
    from retrieval.weaviate_retriever import WeaviateRetriever
 
    embedder = Embedder(model_name=cfg.EMBEDDING_MODEL_NAME)
    retriever = HybridRetriever(
        vec=WeaviateRetriever(
            embedder, cfg.WEAVIATE_HOST, cfg.WEAVIATE_PORT, cfg.WEAVIATE_GRPC_PORT, cfg.WEAVIATE_COLLECTION_NAME
        ),
        graph=GraphRetriever(cfg.NEO4J_URI, cfg.NEO4J_USERNAME, cfg.NEO4J_PASSWORD),
        top_k=cfg.RETRIEVAL_TOP_K,
    )
    llm = LLMClient(
        api_key=cfg.GROQ_API_KEY,
        default_model=cfg.GROQ_MODEL_DEFAULT,
        upgrade_model=cfg.GROQ_MODEL_UPGRADE,
        redis_url=cfg.REDIS_URL,
    )
    session_store = SessionStateStore(redis_url=cfg.REDIS_URL)
 
    # Real class, real logic, scratch destination — see the module docstring.
    escalation = EscalationHandler(store_path=EVAL_ESCALATION_DB, slack_webhook_url="")
 
    agent = Agent(
        retriever=retriever, llm=llm, embedder=embedder, session_store=session_store, escalation=escalation
    )
    baseline = RagBaseline(retriever=retriever, llm=llm)
    return agent, baseline, retriever, embedder
 
 
# ---------------------------------------------------------------------------
# Driving the agent to a terminal route
# ---------------------------------------------------------------------------
 
 
def _reply_for_pending(pending: str, original_query: str) -> str:
    """What a customer who cannot clarify further would send next.
 
    - awaiting_clarification: they restate the same request verbatim. This is
      the cleanest possible probe — the agent re-runs the identical confidence
      check on identical text, so if it was too unclear to act on the first
      time, it is still too unclear now, and must escalate. It also avoids
      accidentally *helping* the agent with new vocabulary that a hand-written
      "I don't know" reply would smuggle in.
    - awaiting_slot: they state plainly that they don't have the order number.
      This is the realistic failure and the one the escalation path exists for.
    """
    if pending == "awaiting_clarification":
        return original_query
    if pending == "awaiting_slot":
        return "I don't have the order number."
    raise ValueError(f"No simulated reply defined for pending state: {pending!r}")
 
 
def run_agent_to_terminal(agent, query: str) -> dict:
    """Sends `query` on a fresh session and keeps replying until the agent stops
    asking questions. Returns the terminal response plus a transcript.
 
    Terminal states:
      - pending is None            -> the agent answered, acted, or finished escalating
      - pending == awaiting_contact -> the agent has ALREADY decided to escalate
                                       (route is "escalate", escalation_id is
                                       still None because the packet isn't
                                       written until contact resolves). Routing
                                       is decided; we stop here rather than
                                       feeding it a fake email, which would only
                                       exercise escalation.py's storage path and
                                       tell us nothing about routing.
 
    Latency (Criterion 3) is timed around `handle_message()` and NOTHING else.
    The harness's own scoring embeds are not part of a production request and
    must not be counted. We record every turn separately:
 
      - `first_turn_latency_s` is the number a customer experiences on their
        opening message, and the one Criterion 3 is about.
      - `total_latency_s` sums the whole conversation, which is what an
        escalating example actually costs end to end.
    """
    session_id = f"eval-{uuid.uuid4()}"
    transcript = []
    latencies = []
 
    started = time.perf_counter()
    response = agent.handle_message(query, session_id)
    latencies.append(time.perf_counter() - started)
    transcript.append({"sent": query, "route": response.route, "detail": response.detail, "latency_s": latencies[0]})
 
    for _ in range(MAX_TURNS - 1):
        pending = agent.session_store.get_state(session_id).pending
        if pending is None or pending == "awaiting_contact":
            break
        reply = _reply_for_pending(pending, query)
        started = time.perf_counter()
        response = agent.handle_message(reply, session_id)
        elapsed = time.perf_counter() - started
        latencies.append(elapsed)
        transcript.append({"sent": reply, "route": response.route, "detail": response.detail, "latency_s": elapsed})
 
    state = agent.session_store.get_state(session_id)
    return {
        "response": response,
        "transcript": transcript,
        "final_pending": state.pending,
        "pending_context": state.pending_context,
        "session_id": session_id,
        "first_turn_latency_s": latencies[0],
        "total_latency_s": sum(latencies),
    }
 
 
# ---------------------------------------------------------------------------
# Scoring one example
# ---------------------------------------------------------------------------
 
 
def _recall_hit(retrieval_result, expected_source_file) -> bool:
    """Did the correct source document appear anywhere in the top-k chunks?"""
    if not expected_source_file:
        return None
    return any(c.source_file == expected_source_file for c in retrieval_result.chunks)
 
 
def _escalation_reason(agent, outcome) -> str:
    """The agent's own `reason` string for an escalation — at zero Groq cost.
 
    An escalation ends in one of exactly two states, and the reason is readable
    from a different place in each:
 
      1. Contact was NOT already known -> the agent parked at
         pending="awaiting_contact" and stashed everything it knows, including
         `reason`, in `pending_context`. No packet was written and Slack never
         fired. We read it straight out of the session state.
 
      2. Contact WAS already known (an email/phone in the query, or a slot) ->
         `_finalize_escalation` ran immediately, so a real packet exists in the
         sandboxed SQLite store and carries `escalation_id`. We look it up.
 
    Returning None here would silently collapse every escalation into one
    undifferentiated bucket, which is precisely the analysis we're trying to get.
    """
    response = outcome["response"]
    if response.route != "escalate":
        return None
 
    if outcome["final_pending"] == "awaiting_contact":
        return outcome["pending_context"].get("reason")
 
    if response.escalation_id:
        for packet in agent.escalation.list_escalations():
            if packet.escalation_id == response.escalation_id:
                return packet.reason
 
    return None
 
 
def _score_groundedness(embedder, answer: str, retrieval_result) -> bool:
    """Is the generated answer supported by at least one retrieved chunk?
 
    Reuses agent/groundedness.py exactly — the same function the agent uses on
    itself at request time, with the same threshold. We do not reimplement it.
    """
    from agent.groundedness import is_grounded
 
    chunks = retrieval_result.chunks
    if not chunks or not answer:
        return None
    grounded, _ = is_grounded(
        embedder,
        answer,
        [c.text for c in chunks],
        reference_vectors=[c.vector for c in chunks],
    )
    return grounded
 
 
def _score_answer_match(embedder, answer: str, expected_answer: str) -> bool:
    """Semantic match against the hand-written expected answer, thresholded at
    ANSWER_MATCH_THRESHOLD (0.60) — a separate, higher bar than groundedness's
    0.55, because a generically-phrased answer can score ~0.57 against a policy
    answer it doesn't actually address."""
    import app.config as cfg
    from agent.groundedness import max_similarity_to_any
 
    if not answer or not expected_answer:
        return None
    score = max_similarity_to_any(embedder, answer, [expected_answer])
    return score >= cfg.ANSWER_MATCH_THRESHOLD
 
 
def score_agent_example(agent, embedder, gold_retrieval, example: dict) -> dict:
    """Runs the full agent on one example and scores every metric for it."""
    meta = example["metadata"]
    try:
        outcome = run_agent_to_terminal(agent, example["input"])
    except Exception as e:  # one bad example must not kill a 3-seed run
        return {
            "id": example["id"],
            "expected_route": example["expected_route"],
            "predicted_route": "error",
            "detail": f"{type(e).__name__}: {e}",
            "grounded": None,
            "answer_grounded": None,
            "reason": None,
            "recall_hit": None,
            "answer_match": None,
            "first_turn_latency_s": None,
            "total_latency_s": None,
            "metadata": meta,
            "transcript": [],
        }
 
    response = outcome["response"]
    # The agent generated free prose only on a settled information route.
    # Smalltalk replies and unterminated clarification questions are canned
    # strings, not retrieval-grounded answers — scoring them would be noise.
    generated_answer = (
        response.route == "information"
        and response.intent != "smalltalk"
        and outcome["final_pending"] is None
    )
    reason = _escalation_reason(agent, outcome)
 
    # Criterion 2. See metrics.py's module docstring for the full argument.
    #   True  -> the agent emitted an answer, which means it already passed the
    #            runtime is_grounded() check against the chunks the answer was
    #            actually generated from (post-retry).
    #   False -> the agent TRIED to answer, generated twice (once broadened),
    #            and could not ground it. That is a real groundedness failure and
    #            it belongs in the denominator, not filtered out of it.
    #   None  -> the agent never attempted a corpus answer (an action, or an
    #            escalation for some unrelated reason). Not applicable.
    if generated_answer:
        answer_grounded = True
    elif reason == "ungrounded_after_retry":
        answer_grounded = False
    else:
        answer_grounded = None
 
    return {
        "id": example["id"],
        "expected_route": example["expected_route"],
        "predicted_route": response.route,
        "detail": response.detail,
        "intent": response.intent,
        "intent_confidence": response.intent_confidence,
        "retrieval_confidence": response.retrieval_confidence,
        "escalation_id": response.escalation_id,
        "reason": reason,
        "turns": len(outcome["transcript"]),
        "answer_grounded": answer_grounded,
        # Conservative cross-check only — scored against FIRST-PASS chunks, so a
        # graph-broadened retry that rescued the answer is scored against the
        # wrong references. Never report this as Criterion 2.
        "grounded": _score_groundedness(embedder, response.detail, gold_retrieval) if generated_answer else None,
        "recall_hit": _recall_hit(gold_retrieval, meta.get("expected_source_file")),
        "answer_match": (
            _score_answer_match(embedder, response.detail, example["expected_answer"]) if generated_answer else None
        ),
        "first_turn_latency_s": outcome["first_turn_latency_s"],
        "total_latency_s": outcome["total_latency_s"],
        "metadata": meta,
        "transcript": outcome["transcript"],
    }
 
 
def score_baseline_example(baseline, embedder, gold_retrieval, example: dict) -> dict:
    """Runs the plain-RAG baseline on one example and scores it.
 
    The baseline always answers, so `grounded` and `answer_match` always apply.
    Its route is the constant "information" — see eval/baseline.py's docstring.
    """
    meta = example["metadata"]
    try:
        started = time.perf_counter()
        result = baseline.answer(example["input"])
        elapsed = time.perf_counter() - started
    except Exception as e:
        return {
            "id": example["id"],
            "expected_route": example["expected_route"],
            "predicted_route": "error",
            "detail": f"{type(e).__name__}: {e}",
            "grounded": None,
            "answer_grounded": None,
            "reason": None,
            "recall_hit": None,
            "answer_match": None,
            "first_turn_latency_s": None,
            "total_latency_s": None,
            "metadata": meta,
        }
 
    # The baseline never escalates, so EVERY question is an answer attempt and
    # `answer_grounded` is simply its post-hoc groundedness. It has no retry, so
    # its first-pass chunks are the chunks its answer was generated from — the
    # post-hoc check and a runtime check would agree exactly. This is what makes
    # the Criterion-2 comparison against the agent an apples-to-apples one.
    grounded = _score_groundedness(embedder, result.detail, result.retrieval_result)
 
    return {
        "id": example["id"],
        "expected_route": example["expected_route"],
        "predicted_route": result.route,
        "detail": result.detail,
        "reason": None,
        "grounded": grounded,
        "answer_grounded": grounded,
        "recall_hit": _recall_hit(gold_retrieval, meta.get("expected_source_file")),
        "answer_match": _score_answer_match(embedder, result.detail, example["expected_answer"]),
        "first_turn_latency_s": elapsed,
        "total_latency_s": elapsed,
        "metadata": meta,
    }
 
 
# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
 
 
def _fmt(agg: dict) -> str:
    """mean ± std, or 'n/a' for a metric that could not be computed."""
    if agg["mean"] is None:
        return "n/a"
    return f"{agg['mean']:.3f} ± {agg['std']:.3f}"
 
 
def _aggregate(per_seed_summaries: list) -> dict:
    """Collapses one system's per-seed summaries into mean ± std per metric."""
    return {
        "routing_accuracy": M.mean_std([s["routing_accuracy"]["value"] for s in per_seed_summaries]),
        "escalation_precision": M.mean_std([s["escalation"]["precision"] for s in per_seed_summaries]),
        "escalation_recall": M.mean_std([s["escalation"]["recall"] for s in per_seed_summaries]),
        "escalation_f1": M.mean_std([s["escalation"]["f1"] for s in per_seed_summaries]),
        "answer_attempt_groundedness": M.mean_std(
            [s["answer_attempt_groundedness"]["value"] for s in per_seed_summaries]
        ),
        "groundedness_rate": M.mean_std([s["groundedness_rate"]["value"] for s in per_seed_summaries]),
        "recall_at_k": M.mean_std([s["recall_at_k"]["value"] for s in per_seed_summaries]),
        "answer_match_rate": M.mean_std([s["answer_match_rate"]["value"] for s in per_seed_summaries]),
    }
 
 
def _pooled_latency(rows: list, field: str = "first_turn_latency_s") -> dict:
    """Latencies are POOLED across seeds, not averaged per seed then averaged.
 
    A percentile of per-seed percentiles is not a percentile of anything. Every
    individual request is one sample; 60 examples x 3 seeds gives 180 samples,
    and the p95 of those 180 is the real p95.
    """
    return M.latency_stats([r.get(field) for r in rows])
 
 
def _fmt_secs(value) -> str:
    return "n/a" if value is None else f"{value:.2f}s"
 
 
def print_report(
    agent_agg: dict,
    baseline_agg: dict,
    n_examples: int,
    seeds: list,
    pooled_agent_rows: list,
    pooled_baseline_rows: list,
) -> None:
    print("\n" + "=" * 78)
    print(f"  RESULTS — {n_examples} held-out examples × {len(seeds)} seeds {seeds}")
    print("=" * 78)
    print("\n  PRIMARY METRIC: routing accuracy (terminal route vs. hand-labeled expected_route)\n")
    print(f"    {'Metric':<30}{'Baseline (plain RAG)':>22}{'Full agent':>22}")
    print(f"    {'-' * 74}")
 
    def row(label, key):
        print(f"    {label:<30}{_fmt(baseline_agg[key]):>22}{_fmt(agent_agg[key]):>22}")
 
    row("routing accuracy", "routing_accuracy")
    print()
    row("escalation recall", "escalation_recall")
    row("escalation precision", "escalation_precision")
    row("escalation F1", "escalation_f1")
    print()
    row("groundedness (C2)", "answer_attempt_groundedness")
    row("  ^ cross-check only", "groundedness_rate")
    row("recall@k", "recall_at_k")
    row("answer match rate", "answer_match_rate")
 
    # ---- Latency (Criterion 3) --------------------------------------------
    agent_first = _pooled_latency(pooled_agent_rows, "first_turn_latency_s")
    agent_total = _pooled_latency(pooled_agent_rows, "total_latency_s")
    base_first = _pooled_latency(pooled_baseline_rows, "first_turn_latency_s") if pooled_baseline_rows else {}
 
    print(f"\n    {'Latency (pooled requests)':<30}{'Baseline':>22}{'Full agent':>22}")
    print(f"    {'-' * 74}")
    for label, key in (("p50", "p50"), ("p95", "p95"), ("max", "max")):
        print(
            f"    {'first turn ' + label:<30}"
            f"{_fmt_secs(base_first.get(key)):>22}{_fmt_secs(agent_first.get(key)):>22}"
        )
    under = agent_first.get("pct_under_target")
    under_str = "n/a" if under is None else f"{under:.1%}"
    print(f"    {'first turn < 3.0s':<30}{'':>22}{under_str:>22}")
    print(f"    {'full conversation p95':<30}{'':>22}{_fmt_secs(agent_total.get('p95')):>22}")
 
    # ---- Escalation reasons ----------------------------------------------
    reasons = M.escalation_reasons(pooled_agent_rows)
    print("\n  WHY THE AGENT ESCALATED (pooled across seeds)")
    print(f"    {'-' * 74}")
    if not reasons:
        print("    (no escalations recorded)")
    for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"    {count:>4}x  {reason}")
 
    print("\n  Notes:")
    print("    - C2 groundedness puts ungrounded_after_retry escalations back into the")
    print("      denominator. The 'cross-check only' row is the naive version, which is")
    print("      near-vacuous: the agent never emits an answer that failed its own check.")
    print("    - Baseline escalation precision is 'n/a', not 0.000: it never predicts")
    print("      escalate, so precision is 0/0 (undefined). Its recall IS 0.000 — it")
    print("      misses every example that should have been handed to a human.")
    print("    - recall@k is identical for both systems by construction: they share one")
    print("      HybridRetriever, and this measures first-pass retrieval.")
    print("    - Latency excludes the harness's own scoring embeds. It is measured on a")
    print("      warm process against a live Groq endpoint, so it includes real network")
    print("      time and will vary with your connection.")
 
    # ---- Success criteria -------------------------------------------------
    print("\n" + "=" * 78)
    print("  SUCCESS CRITERIA")
    print("=" * 78 + "\n")
    for name, target, observed, passed in M.check_success_criteria(agent_agg, agent_first):
        if passed is None:
            mark, shown = "?", "not measured"
        else:
            mark = "PASS" if passed else "FAIL"
            shown = f"{observed:.3f}" if target <= 1.0 else f"{observed:.2f}s"
        print(f"    [{mark:^4}]  {name:<40}  observed: {shown}")
 
    print("\n" + "=" * 78)
    print("  ERROR ANALYSIS — routing error rate by category × difficulty")
    print(f"  (pooled across all {len(seeds)} seeds, agent only)")
    print("=" * 78 + "\n")
    grid = M.error_grid(pooled_agent_rows, "category", "difficulty")
    print(M.format_error_grid(grid, "category", "difficulty"))
 
    print("\n  Worst cells (highest routing error rate, n >= 3):")
    worst = sorted(
        [(k, v) for k, v in grid.items() if v["n"] >= 3 and v["error_rate"] > 0],
        key=lambda kv: -kv[1]["error_rate"],
    )[:5]
    if not worst:
        print("    none — no cell with n>=3 had any routing error.")
    for (cat, diff), cell in worst:
        print(f"    {cat} × {diff}: {cell['error_rate']:.0%} ({cell['wrong']}/{cell['n']})")
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
 
 
def load_heldout(path: Path) -> list:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
 
 
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heldout", type=Path, default=Path("eval/heldout.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("eval/results.json"))
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N examples (smoke test).")
    parser.add_argument("--skip-baseline", action="store_true", help="Agent only — saves ~1 Groq call per example.")
    args = parser.parse_args()
 
    heldout = load_heldout(args.heldout)
    if args.limit:
        heldout = heldout[: args.limit]
    print(f"Loaded {len(heldout)} held-out examples from {args.heldout}.")
 
    agent, baseline, retriever, embedder = build_systems()
 
    # Retrieval is deterministic given the frozen corpus, so retrieve once per
    # example and reuse across every seed. This is the gold retrieval used for
    # recall@k and for the agent's groundedness references.
    print("Retrieving once per example (deterministic — reused across seeds)...")
    gold_retrievals = {ex["id"]: retriever.retrieve(ex["input"]) for ex in heldout}
 
    agent_summaries, baseline_summaries = [], []
    pooled_agent_rows, pooled_baseline_rows, per_seed_detail = [], [], {}
 
    for seed in args.seeds:
        set_all_seeds(seed)
        print(f"\n--- seed {seed} ---")
 
        agent_rows = [score_agent_example(agent, embedder, gold_retrievals[ex["id"]], ex) for ex in heldout]
        agent_summary = M.summarize(agent_rows)
        agent_summaries.append(agent_summary)
        pooled_agent_rows.extend(agent_rows)
        print(f"  agent    routing accuracy = {agent_summary['routing_accuracy']['value']:.4f}")
 
        baseline_rows = []
        if not args.skip_baseline:
            baseline_rows = [score_baseline_example(baseline, embedder, gold_retrievals[ex["id"]], ex) for ex in heldout]
            baseline_summary = M.summarize(baseline_rows)
            baseline_summaries.append(baseline_summary)
            pooled_baseline_rows.extend(baseline_rows)
            print(f"  baseline routing accuracy = {baseline_summary['routing_accuracy']['value']:.4f}")
 
        per_seed_detail[str(seed)] = {"agent": agent_rows, "baseline": baseline_rows}
 
    agent_agg = _aggregate(agent_summaries)
    baseline_agg = (
        _aggregate(baseline_summaries)
        if baseline_summaries
        else {k: {"mean": None, "std": None, "n_seeds": 0, "values": []} for k in agent_agg}
    )
 
    print_report(agent_agg, baseline_agg, len(heldout), args.seeds, pooled_agent_rows, pooled_baseline_rows)
 
    grid = M.error_grid(pooled_agent_rows, "category", "difficulty")
    agent_first_latency = _pooled_latency(pooled_agent_rows, "first_turn_latency_s")
    payload = {
        "primary_metric": "routing_accuracy",
        "run_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "n_examples": len(heldout),
        "seeds": args.seeds,
        "summary": {"agent": agent_agg, "baseline": baseline_agg},
        "latency": {
            "agent_first_turn": agent_first_latency,
            "agent_full_conversation": _pooled_latency(pooled_agent_rows, "total_latency_s"),
            "baseline": _pooled_latency(pooled_baseline_rows, "first_turn_latency_s") if pooled_baseline_rows else None,
        },
        "escalation_reasons": M.escalation_reasons(pooled_agent_rows),
        "success_criteria": [
            {"criterion": name, "target": target, "observed": observed, "passed": passed}
            for name, target, observed, passed in M.check_success_criteria(agent_agg, agent_first_latency)
        ],
        "error_grid": {f"{cat}|{diff}": cell for (cat, diff), cell in grid.items()},
        "per_seed": per_seed_detail,
    }
 
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
 
    # Versioned copy, per the guide's "version your results" instruction — the
    # unversioned results.json is always the latest, the dated one is history.
    stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M")
    versioned = args.output.parent / f"results-{stamp}.json"
    with versioned.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
 
    print(f"\nWrote {args.output} and {versioned}\n")
 
 
if __name__ == "__main__":
    main()