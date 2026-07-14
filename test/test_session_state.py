from agent.session_state import SessionStateStore, TTL_SECONDS
from app.schemas import SessionState


class FakeRedis:
    """Minimal in-memory stand-in for redis.Redis — just enough of the API
    surface session_state.py uses, so tests don't need a live Redis."""

    def __init__(self):
        self.store = {}
        self.last_ex = {}

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value, ex=None):
        self.store[key] = value
        self.last_ex[key] = ex
        return True


def _store():
    return SessionStateStore(client=FakeRedis())


def test_get_state_defaults_when_never_seen():
    store = _store()
    state = store.get_state("session-1")
    assert state.session_id == "session-1"
    assert state.pending is None
    assert state.pending_context == {}
    assert state.turn_count == 0


def test_set_state_then_get_state_roundtrips():
    store = _store()
    state = SessionState(session_id="session-1", pending="awaiting_slot", pending_context={"a": 1}, turn_count=3)
    store.set_state("session-1", state)

    fetched = store.get_state("session-1")
    assert fetched.pending == "awaiting_slot"
    assert fetched.pending_context == {"a": 1}
    assert fetched.turn_count == 3


def test_set_state_refreshes_ttl_every_write():
    store = _store()
    state = SessionState(session_id="s", pending=None, pending_context={}, turn_count=1)
    store.set_state("s", state)
    assert store._redis.last_ex["session:s"] == TTL_SECONDS

    state.turn_count = 2
    store.set_state("s", state)
    assert store._redis.last_ex["session:s"] == TTL_SECONDS  # refreshed again, not decremented


def test_clear_pending_drops_pending_and_context_but_keeps_turn_count():
    store = _store()
    state = SessionState(session_id="s", pending="awaiting_slot", pending_context={"order_id": "ORD-1"}, turn_count=5)
    store.set_state("s", state)

    cleared = store.clear_pending("s")

    assert cleared.pending is None
    assert cleared.pending_context == {}
    assert cleared.turn_count == 5
    assert store.get_state("s").pending is None


def test_sessions_are_isolated_by_id():
    store = _store()
    store.set_state("a", SessionState(session_id="a", pending="awaiting_slot", pending_context={}, turn_count=1))
    store.set_state("b", SessionState(session_id="b", pending=None, pending_context={}, turn_count=9))

    assert store.get_state("a").pending == "awaiting_slot"
    assert store.get_state("b").turn_count == 9
