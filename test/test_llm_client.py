import httpx
import groq

from agent.llm_client import DAILY_CALL_COUNTER_KEY_PREFIX, LLMClient


class FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class FakeChoice:
    def __init__(self, message):
        self.message = message


class FakeChatCompletion:
    def __init__(self, content="ok"):
        self.choices = [FakeChoice(FakeMessage(content=content))]


class FakeRawResponse:
    def __init__(self, headers=None, content="ok"):
        self.headers = headers or {}
        self._completion = FakeChatCompletion(content)

    def parse(self):
        return self._completion


class FakeCompletionsWithRawResponse:
    def __init__(self, queue):
        self._queue = list(queue)
        self.received_models = []

    def create(self, **kwargs):
        self.received_models.append(kwargs["model"])
        item = self._queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeGroqClient:
    def __init__(self, queue):
        raw = FakeCompletionsWithRawResponse(queue)
        completions = type("C", (), {"with_raw_response": raw})()
        self.chat = type("Chat", (), {"completions": completions})()


class FakeRedis:
    def __init__(self):
        self.store = {}
        self.ttls = {}

    def incr(self, key):
        self.store[key] = self.store.get(key, 0) + 1
        return self.store[key]

    def expire(self, key, ttl):
        self.ttls[key] = ttl


def _client(queue, **kwargs):
    groq_client = FakeGroqClient(queue)
    return LLMClient(groq_client=groq_client, redis_client=FakeRedis(), retry_backoff_seconds=0, **kwargs), groq_client


def test_complete_returns_parsed_completion():
    client, _ = _client([FakeRawResponse(content="hello")])
    result = client.complete(messages=[{"role": "user", "content": "hi"}])
    assert result.choices[0].message.content == "hello"


def test_complete_increments_daily_call_counter():
    client, _ = _client([FakeRawResponse(), FakeRawResponse()])
    client.complete(messages=[{"role": "user", "content": "1"}])
    client.complete(messages=[{"role": "user", "content": "2"}])

    keys = [k for k in client._redis.store if k.startswith(DAILY_CALL_COUNTER_KEY_PREFIX)]
    assert len(keys) == 1
    assert client._redis.store[keys[0]] == 2
    assert client._redis.ttls[keys[0]] > 0  # TTL set on first increment of the day


def test_circuit_breaker_trips_below_threshold_and_forces_default_model():
    low_quota_headers = {"x-ratelimit-remaining-tokens": "5000", "x-ratelimit-limit-tokens": "100000"}  # 5%
    client, groq_client = _client(
        [FakeRawResponse(headers=low_quota_headers), FakeRawResponse()],
        default_model="llama-3.1-8b-instant",
        circuit_breaker_min_remaining_pct=0.10,
    )

    client.complete(messages=[{"role": "user", "content": "1"}], model="llama-3.3-70b-versatile")
    assert client.circuit_tripped is True

    client.complete(messages=[{"role": "user", "content": "2"}], model="llama-3.3-70b-versatile")

    raw = groq_client.chat.completions.with_raw_response
    assert raw.received_models == ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]


def test_circuit_breaker_does_not_trip_above_threshold():
    healthy_headers = {"x-ratelimit-remaining-tokens": "90000", "x-ratelimit-limit-tokens": "100000"}  # 90%
    client, _ = _client([FakeRawResponse(headers=healthy_headers)], circuit_breaker_min_remaining_pct=0.10)

    client.complete(messages=[{"role": "user", "content": "1"}], model="llama-3.3-70b-versatile")

    assert client.circuit_tripped is False


def test_circuit_breaker_logs_warning_exactly_once_across_multiple_tripped_calls(caplog):
    low_quota_headers = {"x-ratelimit-remaining-tokens": "1000", "x-ratelimit-limit-tokens": "100000"}
    client, _ = _client(
        [FakeRawResponse(headers=low_quota_headers) for _ in range(3)],
        circuit_breaker_min_remaining_pct=0.10,
    )

    with caplog.at_level("WARNING"):
        for i in range(3):
            client.complete(messages=[{"role": "user", "content": str(i)}])

    tripped_logs = [r for r in caplog.records if "TRIPPED" in r.message]
    assert len(tripped_logs) == 1


def test_retries_transient_error_then_succeeds():
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    error = groq.APIConnectionError(request=request)
    client, groq_client = _client([error, FakeRawResponse(content="recovered")], max_retries=3)

    result = client.complete(messages=[{"role": "user", "content": "hi"}])

    assert result.choices[0].message.content == "recovered"
    assert len(groq_client.chat.completions.with_raw_response.received_models) == 2


def test_raises_after_exhausting_retries():
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    errors = [groq.APIConnectionError(request=request) for _ in range(3)]
    client, _ = _client(errors, max_retries=3)

    try:
        client.complete(messages=[{"role": "user", "content": "hi"}])
        assert False, "expected APIConnectionError to be raised"
    except groq.APIConnectionError:
        pass


def test_daily_call_counter_increments_even_on_a_rejected_call():
    """Regression test for ERRORS.md #6: a non-retryable error (e.g. Groq
    rejecting a malformed tool call with BadRequestError) still represents a
    real HTTP round trip against the account's quota, so it must still be
    counted — not just clean successes."""
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(400, request=request, json={"error": {"message": "bad"}})
    error = groq.BadRequestError("bad", response=response, body=None)
    client, _ = _client([error])

    try:
        client.complete(messages=[{"role": "user", "content": "hi"}])
    except groq.BadRequestError:
        pass

    keys = [k for k in client._redis.store if k.startswith(DAILY_CALL_COUNTER_KEY_PREFIX)]
    assert len(keys) == 1
    assert client._redis.store[keys[0]] == 1
