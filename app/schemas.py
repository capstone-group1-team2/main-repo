"""Every Pydantic request/response model, exactly per ARCHITECTURE.md §9.

Pulled forward into M4 (Owner C) because agent/agent.py needs a return type
now — M5 (Owner D, real app/ scope) imports these, it does not recreate
them. Pure data contracts only: no route handlers, no business logic.
"""

from typing import Literal, Optional

from pydantic import BaseModel


class ChatRequest(BaseModel):
    query: str
    session_id: str


class SessionState(BaseModel):
    session_id: str
    # "awaiting_clarification" added in M9: a one-shot human-worded
    # clarifying question asked before ever escalating on low
    # intent-confidence or low retrieval-relevance (see ANALYSIS.md's M9
    # entry). "awaiting_contact" was defined back in M4 but never actually
    # used until M9 — it's now a real one-shot round-trip asking for
    # contact info BEFORE the escalation packet is built/stored/posted to
    # Slack, replacing M5's same-turn cosmetic ask that could never be
    # answered before Slack already fired.
    pending: Optional[Literal["awaiting_slot", "awaiting_clarification", "awaiting_contact"]]
    pending_context: dict
    turn_count: int
    # M11: the most recently completed track_order/cancel_order action's
    # order_id/intent/status, kept for exactly one follow-up turn (with
    # natural chaining if that follow-up is itself another successful
    # action) so a direct reference ("my order"/"it") or a delivery
    # dispute right after a "delivered" result doesn't get treated as a
    # brand-new request missing its slot. {} means no remembered order.
    # Distinct from `pending_context`: this is a resolved fact kept around
    # for grounding, not an unresolved question being tracked.
    last_order_context: dict = {}


class AgentResponse(BaseModel):
    route: Literal["information", "action", "escalate"]
    intent: str
    intent_confidence: float
    retrieval_confidence: float
    detail: str  # the generated answer, or the tool result
    escalation_id: Optional[str]
    session_id: str


class EscalationPacket(BaseModel):
    escalation_id: str
    session_id: str
    contact: Optional[str]
    contact_captured: bool
    slack_delivered: bool
    query: str
    retrieved_context: list
    reason: str
    timestamp: str
    # M9: a plain-English summary of what the agent tried before
    # escalating (intent classified, whether a clarifying question was
    # asked and what the reply was, etc.) — replaces a bare reason enum
    # in the Slack notification. See ANALYSIS.md's M9 entry.
    attempted_summary: str = ""


class HeldoutExample(BaseModel):
    id: str
    input: str
    expected_answer: str
    expected_route: Literal["information", "action", "escalate"]
    metadata: dict  # category, difficulty, etc.
