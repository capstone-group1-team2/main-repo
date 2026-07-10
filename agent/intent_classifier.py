"""One Groq call does both intent classification AND slot extraction via
real function-calling (ARCHITECTURE.md §5, §8) — not regex, not two calls.

SUPPORT_INTENTS is the verified, exact set of 27 unique values in
data/bitext_customer_support.csv's `intent` column (confirmed by reading
the actual file, not guessed) — hardcoded here rather than read from the
19MB CSV at request time, since it's a fixed data contract, not runtime data.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import groq

from agent.llm_client import LLMClient

logger = logging.getLogger("agent.intent_classifier")

SUPPORT_INTENTS = [
    "cancel_order",
    "change_order",
    "change_shipping_address",
    "check_cancellation_fee",
    "check_invoice",
    "check_payment_methods",
    "check_refund_policy",
    "complaint",
    "contact_customer_service",
    "contact_human_agent",
    "create_account",
    "delete_account",
    "delivery_options",
    "delivery_period",
    "edit_account",
    "get_invoice",
    "get_refund",
    "newsletter_subscription",
    "payment_issue",
    "place_order",
    "recover_password",
    "registration_problems",
    "review",
    "set_up_shipping_address",
    "switch_account",
    "track_order",
    "track_refund",
]
assert len(SUPPORT_INTENTS) == 27, f"expected 27 intents, got {len(SUPPORT_INTENTS)}"

# Intents whose action requires an order_id to actually execute a tool call
# (agent.py's ACTION-intent branch, §8). Everything else routes to
# INFORMATION even if it happens to mention an order in passing.
ACTION_INTENTS = {"cancel_order", "track_order"}

_CLASSIFY_TOOL = {
    "type": "function",
    "function": {
        "name": "classify_support_request",
        "description": (
            "Classifies a customer support message by intent and extracts any "
            "structured slots explicitly mentioned in it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "enum": SUPPORT_INTENTS,
                    "description": "The single best-matching intent for this message.",
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence in this intent classification, from 0.0 to 1.0.",
                },
                "slots": {
                    "type": "object",
                    "properties": {
                        # No example values in these descriptions — a small
                        # model will parrot a literal example back as if it
                        # were the extracted value when the message doesn't
                        # actually contain one (see ERRORS.md #5). Describe
                        # the *shape*, never a concrete instance.
                        "order_id": {"type": "string", "description": "The customer's order identifier, copied verbatim from their message."},
                        "tracking_number": {"type": "string", "description": "A shipment tracking number, copied verbatim from their message."},
                        "email": {"type": "string", "description": "An email address, copied verbatim from their message."},
                        "phone": {"type": "string", "description": "A phone number, copied verbatim from their message."},
                        "item": {"type": "string", "description": "A product name, copied verbatim from their message."},
                        "new_shipping_address": {"type": "string", "description": "A shipping address, copied verbatim from their message."},
                        "account_field": {"type": "string", "description": "An account field name, copied verbatim from their message."},
                    },
                    "description": (
                        "Only include a slot if its value is explicitly present in the customer's message. "
                        "Never invent, guess, or fill in an example — omit the field entirely if it wasn't mentioned."
                    ),
                },
            },
            "required": ["intent", "confidence", "slots"],
        },
    },
}


@dataclass(frozen=True)
class IntentResult:
    intent: str
    intent_confidence: float
    slots: dict = field(default_factory=dict)


_SYSTEM_PROMPT = (
    "You classify customer support messages for Meridian Retail, a retail company. "
    "Always call classify_support_request exactly once with your best answer, even if unsure. "
    "You MUST include all three parameters in the call: intent (one value from the enum), "
    "confidence (a number from 0.0 to 1.0), and slots (an object — use {} if nothing applies). "
    "Slot values must be copied verbatim from the customer's message. If a slot (like an order "
    "number) is not explicitly present in their message, leave it out of slots entirely — never "
    "invent, guess, or use a placeholder/example value."
)

_UNKNOWN_RESULT = IntentResult(intent="unknown", intent_confidence=0.0, slots={})


def _grounded_slots(raw_slots: dict, query: str) -> dict:
    """Defense in depth against slot hallucination (see ERRORS.md #5): a
    small model can still occasionally invent a slot value — e.g. parroting
    a schema example, or filling in a plausible-looking order number that
    was never actually in the message — even after a well-worded prompt.
    Discards any slot whose value doesn't literally appear (case-insensitive)
    in the original query, since a genuine extraction should always be a
    verbatim substring of what the customer actually typed."""
    query_lower = query.lower()
    return {k: v for k, v in raw_slots.items() if v and str(v).lower() in query_lower}


def classify_intent_and_slots(llm: LLMClient, query: str) -> IntentResult:
    """A small (8B) model doesn't always conform to a forced tool call's
    schema on the first attempt (observed live: it can omit required fields
    or invent different parameter names entirely — Groq raises
    `BadRequestError` with code `tool_use_failed` when this happens). Retried
    once with an explicit reminder; if it fails twice, this degrades to a
    safe "unknown"/0.0-confidence result rather than crashing the turn —
    `agent.py` routes that straight to escalation, which is the correct
    outcome for "the model couldn't even classify this," not a bug to hide."""
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    for attempt in range(2):
        try:
            completion = llm.complete(
                messages=messages,
                tools=[_CLASSIFY_TOOL],
                tool_choice={"type": "function", "function": {"name": "classify_support_request"}},
            )
            break
        except groq.BadRequestError as e:
            logger.warning("Tool-call validation failed on attempt %d/2: %s", attempt + 1, e)
            if attempt == 1:
                return _UNKNOWN_RESULT
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous call was missing required parameters. Call classify_support_request "
                        "again with ALL THREE parameters: intent, confidence, and slots."
                    ),
                }
            )

    tool_calls = completion.choices[0].message.tool_calls
    if not tool_calls:
        return _UNKNOWN_RESULT

    try:
        args = json.loads(tool_calls[0].function.arguments)
    except (json.JSONDecodeError, TypeError):
        return _UNKNOWN_RESULT

    intent = args.get("intent", "")
    slots = _grounded_slots(args.get("slots") or {}, query)

    if intent not in SUPPORT_INTENTS:
        # Model returned something outside the enum — degrade to
        # low-confidence rather than crashing the turn on a malformed call.
        return IntentResult(intent=intent or "unknown", intent_confidence=0.0, slots=slots)

    confidence = max(0.0, min(1.0, float(args.get("confidence", 0.0))))
    return IntentResult(intent=intent, intent_confidence=confidence, slots=slots)
