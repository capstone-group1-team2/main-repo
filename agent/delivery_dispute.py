"""Cheap, pre-Groq-independent detection of "I didn't actually receive
this" language (M11 — ARCHITECTURE.md §8's routing flow had no way to
distinguish a plain track_order query from a customer disputing a
`delivered` status; both got the identical flat "This order has already
been delivered." reply with no escalation).

Unlike `agent/smalltalk.py`'s exact-whole-message match, dispute language
is normally embedded in a longer sentence ("but I didn't receive it !!"),
so this is a substring check, not a whole-message match. It's deliberately
narrow: `agent/agent.py` only calls this after confirming the message is
about the SAME order that was just reported `delivered` (see agent.py's
`_route_confident`) — this module has no context of its own about which
order is in play, it only recognizes the dispute language itself.
"""

from __future__ import annotations

_DISPUTE_PHRASES = [
    "didn't receive",
    "did not receive",
    "never received",
    "never got it",
    "never arrived",
    "haven't received",
    "have not received",
    "hasn't arrived",
    "has not arrived",
    "wasn't delivered",
    "was not delivered",
    "don't have it",
    "dont have it",
    "haven't got",
    "havent got",
]


def detect_delivery_dispute(query: str) -> bool:
    """True if `query` contains language disputing physical receipt of a
    package, anywhere in the message (substring, case-insensitive)."""
    normalized = query.lower()
    return any(phrase in normalized for phrase in _DISPUTE_PHRASES)
