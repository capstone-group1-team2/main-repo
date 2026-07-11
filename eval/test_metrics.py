"""Unit tests for the evaluation harness's scoring logic and its terminal-route
driver. Run from the repo root:
 
    pytest eval/test_metrics.py -v
 
These tests deliberately require NO Redis, NO Weaviate, NO Neo4j and NO Groq
key. The scoring functions in eval/metrics.py are pure, and eval/run_eval.py
imports the stack lazily, so the trickiest logic in the harness — driving a
conversation to its terminal route — can be verified against a fake agent that
reproduces agent.py's real multi-turn behavior.
 
Why this matters: if `run_agent_to_terminal()` is wrong, every escalation
number in the final report is wrong, and it is wrong in the flattering
direction (an agent that escalates correctly would appear to never escalate).
That is exactly the kind of bug an eval harness must not have.
"""
 
from __future__ import annotations
 
from dataclasses import dataclass, field
 
from eval import metrics as M
from eval.run_eval import _escalation_reason, _reply_for_pending, run_agent_to_terminal
 
 
# ---------------------------------------------------------------------------
# Fakes: minimal stand-ins that reproduce the real objects' observable behavior
# ---------------------------------------------------------------------------
 
 
@dataclass
class FakeResponse:
    route: str
    intent: str = "check_refund_policy"
    detail: str = "some reply"
    intent_confidence: float = 0.9
    retrieval_confidence: float = 0.8
    escalation_id: str = None
 
 
@dataclass
class FakePacket:
    escalation_id: str
    reason: str
 
 
@dataclass
class FakeState:
    pending: str = None
    pending_context: dict = field(default_factory=dict)
 
 
class FakeSessionStore:
    def __init__(self):
        self.states = {}
 
    def get_state(self, session_id):
        return self.states.get(session_id, FakeState())
 
 
class FakeEscalationHandler:
    def __init__(self, packets=()):
        self._packets = list(packets)
 
    def list_escalations(self, slack_delivered=None):
        return self._packets
 
 
class ScriptedAgent:
    """Replays a fixed sequence of (response, pending_after, pending_context)
    triples, mimicking how the real Agent leaves state in Redis after each turn."""
 
    def __init__(self, script, packets=()):
        self.script = [s if len(s) == 3 else (*s, {}) for s in script]
        self.session_store = FakeSessionStore()
        self.escalation = FakeEscalationHandler(packets)
        self.sent = []
 
    def handle_message(self, query, session_id):
        self.sent.append(query)
        response, pending_after, pending_context = self.script.pop(0)
        self.session_store.states[session_id] = FakeState(
            pending=pending_after, pending_context=pending_context
        )
        return response
 
 
# ---------------------------------------------------------------------------
# Terminal-route driver — the behavior the whole report depends on
# ---------------------------------------------------------------------------
 
 
def test_confident_information_answer_terminates_on_first_turn():
    agent = ScriptedAgent([(FakeResponse(route="information"), None)])
    out = run_agent_to_terminal(agent, "How long does standard shipping take?")
    assert out["response"].route == "information"
    assert len(out["transcript"]) == 1
 
 
def test_action_terminates_on_first_turn():
    agent = ScriptedAgent([(FakeResponse(route="action"), None)])
    out = run_agent_to_terminal(agent, "Track order ORD-1002.")
    assert out["response"].route == "action"
    assert len(out["transcript"]) == 1
 
 
def test_unclear_message_escalates_only_on_the_second_turn():
    """This is the whole reason the driver exists. The agent's FIRST reply to an
    unclear message is a clarifying question with route='information'. Scoring
    that turn would mark an expected-escalate example wrong."""
    agent = ScriptedAgent(
        [
            (FakeResponse(route="information", detail="could you tell me more?"), "awaiting_clarification"),
            (FakeResponse(route="escalate"), "awaiting_contact"),
        ]
    )
    out = run_agent_to_terminal(agent, "asdkjh asd kjahsd")
 
    assert out["transcript"][0]["route"] == "information"  # would have been scored WRONG
    assert out["response"].route == "escalate"  # correct terminal route
    assert out["final_pending"] == "awaiting_contact"
    assert agent.sent == ["asdkjh asd kjahsd", "asdkjh asd kjahsd"]  # restated verbatim
 
 
def test_missing_slot_escalates_after_customer_cannot_supply_it():
    agent = ScriptedAgent(
        [
            (FakeResponse(route="action", detail="could you share your order number?"), "awaiting_slot"),
            (FakeResponse(route="escalate"), "awaiting_contact"),
        ]
    )
    out = run_agent_to_terminal(agent, "I want to cancel my order.")
    assert out["response"].route == "escalate"
    assert agent.sent[1] == "I don't have the order number."
 
 
def test_awaiting_contact_is_treated_as_terminal_not_pumped_further():
    """route is already 'escalate' at awaiting_contact; feeding it a fake email
    would only exercise escalation.py's storage path, not routing."""
    agent = ScriptedAgent([(FakeResponse(route="escalate"), "awaiting_contact")])
    out = run_agent_to_terminal(agent, "What's your stock ticker?")
    assert out["response"].route == "escalate"
    assert len(agent.sent) == 1  # never sent a second message
 
 
def test_latency_is_captured_per_turn_and_summed():
    agent = ScriptedAgent(
        [
            (FakeResponse(route="information"), "awaiting_clarification", {}),
            (FakeResponse(route="escalate"), "awaiting_contact", {"reason": "r"}),
        ]
    )
    out = run_agent_to_terminal(agent, "unclear")
    assert out["first_turn_latency_s"] >= 0
    assert out["total_latency_s"] >= out["first_turn_latency_s"]
    assert len(out["transcript"]) == 2
    assert all("latency_s" in t for t in out["transcript"])
 
 
# ---------------------------------------------------------------------------
# Escalation reason extraction — two distinct code paths
# ---------------------------------------------------------------------------
 
 
def test_reason_read_from_pending_context_when_contact_unknown():
    """No packet was written and Slack never fired; the reason lives in
    pending_context. Reading it costs zero Groq calls."""
    agent = ScriptedAgent(
        [(FakeResponse(route="escalate"), "awaiting_contact", {"reason": "ungrounded_after_retry"})]
    )
    out = run_agent_to_terminal(agent, "q")
    assert _escalation_reason(agent, out) == "ungrounded_after_retry"
 
 
def test_reason_read_from_store_when_escalation_was_finalized_immediately():
    """Contact was already known, so the packet exists in the sandboxed store."""
    agent = ScriptedAgent(
        [(FakeResponse(route="escalate", escalation_id="esc-1"), None, {})],
        packets=[FakePacket(escalation_id="esc-0", reason="wrong"), FakePacket("esc-1", "order_not_found")],
    )
    out = run_agent_to_terminal(agent, "cancel ORD-9999, my email is a@b.com")
    assert _escalation_reason(agent, out) == "order_not_found"
 
 
def test_reason_is_none_for_a_non_escalation():
    agent = ScriptedAgent([(FakeResponse(route="information"), None, {})])
    out = run_agent_to_terminal(agent, "how long is shipping?")
    assert _escalation_reason(agent, out) is None
 
 
def test_reply_for_pending_restates_query_verbatim_on_clarification():
    assert _reply_for_pending("awaiting_clarification", "original text") == "original text"
 
 
def test_reply_for_pending_rejects_unknown_state():
    try:
        _reply_for_pending("awaiting_something_new", "q")
    except ValueError:
        return
    raise AssertionError("expected ValueError for an unrecognized pending state")
 
 
# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
 
 
def _row(expected, predicted, **kw):
    return {
        "id": kw.get("id", "x"),
        "expected_route": expected,
        "predicted_route": predicted,
        "grounded": kw.get("grounded"),
        "answer_grounded": kw.get("answer_grounded"),
        "reason": kw.get("reason"),
        "recall_hit": kw.get("recall_hit"),
        "answer_match": kw.get("answer_match"),
        "first_turn_latency_s": kw.get("first_turn_latency_s"),
        "total_latency_s": kw.get("total_latency_s"),
        "metadata": kw.get("metadata", {"category": "CANCEL", "difficulty": "easy"}),
    }
 
 
def test_routing_accuracy():
    rows = [_row("information", "information"), _row("escalate", "information"), _row("action", "action")]
    assert M.routing_accuracy(rows)["value"] == 2 / 3
    assert M.routing_accuracy(rows)["n"] == 3
 
 
def test_routing_accuracy_empty_set_does_not_divide_by_zero():
    assert M.routing_accuracy([])["value"] == 0.0
 
 
def test_escalation_prf_normal_case():
    rows = [
        _row("escalate", "escalate"),  # TP
        _row("escalate", "information"),  # FN
        _row("information", "escalate"),  # FP
        _row("information", "information"),  # TN
    ]
    prf = M.escalation_prf(rows)
    assert prf["tp"] == 1 and prf["fp"] == 1 and prf["fn"] == 1
    assert prf["precision"] == 0.5
    assert prf["recall"] == 0.5
    assert abs(prf["f1"] - 0.5) < 1e-9
 
 
def test_baseline_that_never_escalates_has_undefined_precision_not_zero():
    """The plain-RAG baseline always answers. Reporting precision=0.0 would
    claim it escalated and got them all wrong; it never escalated at all."""
    rows = [_row("escalate", "information"), _row("information", "information")]
    prf = M.escalation_prf(rows)
    assert prf["precision"] is None  # 0/0, undefined
    assert prf["recall"] == 0.0  # it genuinely missed the one escalation
    assert prf["tp"] == 0 and prf["fp"] == 0
 
 
def test_none_values_are_excluded_from_denominators_not_counted_as_failures():
    rows = [
        _row("information", "information", grounded=True, recall_hit=True, answer_match=True),
        _row("action", "action", grounded=None, recall_hit=None, answer_match=None),
        _row("escalate", "escalate", grounded=None, recall_hit=None, answer_match=None),
    ]
    assert M.groundedness_rate(rows) == {"value": 1.0, "n": 1}
    assert M.recall_at_k(rows) == {"value": 1.0, "n": 1}
    assert M.answer_match_rate(rows) == {"value": 1.0, "n": 1}
 
 
def test_metric_with_no_applicable_examples_is_none_not_zero():
    rows = [_row("action", "action", grounded=None)]
    assert M.groundedness_rate(rows)["value"] is None
 
 
def test_error_grid_counts_and_rates():
    rows = [
        _row("information", "information", metadata={"category": "CANCEL", "difficulty": "easy"}),
        _row("information", "escalate", metadata={"category": "CANCEL", "difficulty": "easy"}),
        _row("escalate", "escalate", metadata={"category": "ADVERSARIAL", "difficulty": "hard"}),
    ]
    grid = M.error_grid(rows)
    assert grid[("CANCEL", "easy")] == {"correct": 1, "wrong": 1, "n": 2, "error_rate": 0.5}
    assert grid[("ADVERSARIAL", "hard")]["error_rate"] == 0.0
 
 
def test_error_grid_formats_without_crashing_on_sparse_cells():
    rows = [
        _row("information", "information", metadata={"category": "CANCEL", "difficulty": "easy"}),
        _row("escalate", "escalate", metadata={"category": "ADVERSARIAL", "difficulty": "hard"}),
    ]
    table = M.format_error_grid(M.error_grid(rows))
    assert "CANCEL" in table and "ADVERSARIAL" in table
    assert "-" in table  # the empty (CANCEL, hard) cell renders as a dash
 
 
# ---------------------------------------------------------------------------
# Criterion 2: the non-vacuous groundedness metric
# ---------------------------------------------------------------------------
 
 
def test_answer_attempt_groundedness_counts_ungrounded_escalations_in_denominator():
    """The whole point. 3 grounded answers + 1 ungrounded escalation = 0.75,
    NOT 1.00. The naive metric would see only the 3 answers and report perfect."""
    rows = [
        _row("information", "information", grounded=True, answer_grounded=True),
        _row("information", "information", grounded=True, answer_grounded=True),
        _row("information", "information", grounded=True, answer_grounded=True),
        _row("information", "escalate", answer_grounded=False, reason="ungrounded_after_retry"),
    ]
    assert M.answer_attempt_groundedness(rows)["value"] == 0.75
    assert M.answer_attempt_groundedness(rows)["n"] == 4
    # ...and the naive one, computed over emitted answers only, is vacuously perfect:
    assert M.groundedness_rate(rows)["value"] == 1.0
    assert M.groundedness_rate(rows)["n"] == 3
 
 
def test_answer_attempt_groundedness_excludes_unrelated_escalations():
    """An order_not_found escalation was never an attempt to answer from the
    corpus. It must not drag the groundedness metric down."""
    rows = [
        _row("information", "information", grounded=True, answer_grounded=True),
        _row("action", "escalate", answer_grounded=None, reason="order_not_found"),
        _row("action", "action", answer_grounded=None),
    ]
    assert M.answer_attempt_groundedness(rows) == {"value": 1.0, "n": 1, "grounded": 1}
 
 
def test_escalation_reasons_counts_only_escalated_rows():
    rows = [
        _row("information", "escalate", reason="ungrounded_after_retry"),
        _row("information", "escalate", reason="ungrounded_after_retry"),
        _row("action", "escalate", reason="order_not_found"),
        _row("information", "information", reason=None),
    ]
    assert M.escalation_reasons(rows) == {"ungrounded_after_retry": 2, "order_not_found": 1}
 
 
# ---------------------------------------------------------------------------
# Criterion 3: latency
# ---------------------------------------------------------------------------
 
 
def test_latency_percentiles_and_target_fraction():
    stats = M.latency_stats([1.0, 2.0, 3.0, 4.0], target_seconds=3.0)
    assert stats["p50"] == 2.5
    assert stats["max"] == 4.0
    assert stats["n"] == 4
    assert stats["pct_under_target"] == 0.5  # 1.0 and 2.0 are under 3.0
 
 
def test_latency_p95_exposes_a_slow_path_that_the_mean_hides():
    """The real shape of the agent's latency: a fast path (classify + generate)
    and a ~10% slow path (+ broaden_via_graph + generate again).
 
    The mean says 1.4s and everything looks fine. The p95 says 5.0s and the
    3-second budget is blown for one request in ten. Report the p95.
    """
    fast, slow = [1.0] * 90, [5.0] * 10
    stats = M.latency_stats(fast + slow, target_seconds=3.0)
    assert abs(stats["mean"] - 1.4) < 1e-9  # comfortably "passing"
    assert stats["p50"] == 1.0
    assert stats["p95"] == 5.0  # the truth
    assert stats["pct_under_target"] == 0.9
 
 
def test_latency_stats_on_empty_input_is_none_not_zero():
    assert M.latency_stats([])["p95"] is None
    assert M.latency_stats([None, None])["p50"] is None
 
 
# ---------------------------------------------------------------------------
# Success criteria
# ---------------------------------------------------------------------------
 
 
def _agg(routing, grounded):
    return {
        "routing_accuracy": {"mean": routing, "std": 0.01},
        "answer_attempt_groundedness": {"mean": grounded, "std": 0.01},
    }
 
 
def test_success_criteria_pass_and_fail():
    results = M.check_success_criteria(_agg(0.90, 0.95), {"p95": 2.4})
    assert [r[3] for r in results] == [True, True, True]
 
    results = M.check_success_criteria(_agg(0.84, 0.88), {"p95": 3.1})
    assert [r[3] for r in results] == [False, False, False]
 
 
def test_success_criteria_boundaries_are_inclusive_for_accuracy_exclusive_for_latency():
    # ">= 0.85" and ">= 0.90" are inclusive; "< 3.0s" is strict.
    results = M.check_success_criteria(_agg(0.85, 0.90), {"p95": 3.0})
    assert [r[3] for r in results] == [True, True, False]
 
 
def test_unmeasurable_criterion_is_none_not_failed():
    """A criterion we couldn't measure must not render as a red X."""
    results = M.check_success_criteria(_agg(0.9, 0.9), {"p95": None})
    assert results[2][3] is None
 
 
def test_mean_std_population_not_sample():
    agg = M.mean_std([0.8, 0.9, 1.0])
    assert abs(agg["mean"] - 0.9) < 1e-9
    assert abs(agg["std"] - 0.08164965809) < 1e-6  # population std, /n not /(n-1)
    assert agg["n_seeds"] == 3
 
 
def test_mean_std_drops_none_and_reports_none_when_nothing_left():
    assert M.mean_std([None, None])["mean"] is None
    assert M.mean_std([None, 0.5])["mean"] == 0.5
 