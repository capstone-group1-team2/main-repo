"""Redis-backed per-session state (ARCHITECTURE.md §8, §9 schema).

15-minute TTL, refreshed on every write — an active session's state never
expires mid-conversation, and an abandoned session cleans itself up
automatically with no manual cleanup job. `session_id` is an anonymous
client-generated UUID (§8) — this store has no notion of user identity.
"""

from __future__ import annotations

import json

import redis

from app.config import REDIS_URL
from app.schemas import SessionState

_KEY_PREFIX = "session:"
TTL_SECONDS = 15 * 60


class SessionStateStore:
    def __init__(self, redis_url: str = REDIS_URL, client=None):
        """`client`, if given, is used as-is instead of constructing a real
        Redis connection — lets tests inject a fake without touching a live
        Redis or monkeypatching private attributes."""
        self._redis = client if client is not None else redis.Redis.from_url(redis_url, decode_responses=True)

    def _key(self, session_id: str) -> str:
        return f"{_KEY_PREFIX}{session_id}"

    def get_state(self, session_id: str) -> SessionState:
        raw = self._redis.get(self._key(session_id))
        if raw is None:
            return SessionState(session_id=session_id, pending=None, pending_context={}, turn_count=0)
        return SessionState(**json.loads(raw))

    def set_state(self, session_id: str, state: SessionState) -> None:
        """Persists `state` and refreshes the 15-minute TTL. Callers own
        incrementing `turn_count` themselves before calling this — this
        store just persists whatever SessionState it's handed."""
        self._redis.set(self._key(session_id), state.model_dump_json(), ex=TTL_SECONDS)

    def clear_pending(self, session_id: str) -> SessionState:
        """Drops any pending slot/contact question — used both when a
        pending flow resolves normally and when agent.py detects the
        customer changed the subject (§8's named failure mode)."""
        state = self.get_state(session_id)
        state.pending = None
        state.pending_context = {}
        self.set_state(session_id, state)
        return state
