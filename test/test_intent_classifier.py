import httpx
import groq

from agent.intent_classifier import classify_intent_and_slots


class FakeMessage:
    def __init__(self, tool_calls=None):
        self.tool_calls = tool_calls


class FakeToolCall:
    def __init__(self, arguments: str):
        self.function = type("F", (), {"arguments": arguments})()


class FakeChoice:
    def __init__(self, message):
        self.message = message


class FakeCompletion:
    def __init__(self, message):
        self.choices = [FakeChoice(message)]


def _bad_request_error():
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(400, request=request, json={"error": {"message": "tool_use_failed"}})
    return groq.BadRequestError("tool_use_failed", response=response, body=None)


class FakeLLM:
    def __init__(self, queue):
        self._queue = list(queue)
        self.calls = 0

    def complete(self, messages, tools=None, tool_choice=None, **kwargs):
        self.calls += 1
        item = self._queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _completion_with_args(args_json: str):
    return FakeCompletion(FakeMessage(tool_calls=[FakeToolCall(args_json)]))


def test_classify_succeeds_on_first_try():
    llm = FakeLLM([_completion_with_args('{"intent": "cancel_order", "confidence": 0.9, "slots": {"order_id": "ORD-1"}}')])
    result = classify_intent_and_slots(llm, "cancel my order ORD-1")
    assert result.intent == "cancel_order"
    assert result.intent_confidence == 0.9
    assert result.slots == {"order_id": "ORD-1"}
    assert llm.calls == 1


def test_classify_retries_once_after_bad_request_then_succeeds():
    llm = FakeLLM(
        [
            _bad_request_error(),
            _completion_with_args('{"intent": "track_order", "confidence": 0.8, "slots": {}}'),
        ]
    )
    result = classify_intent_and_slots(llm, "where is my order")
    assert result.intent == "track_order"
    assert llm.calls == 2


def test_classify_degrades_to_unknown_after_two_bad_requests():
    llm = FakeLLM([_bad_request_error(), _bad_request_error()])
    result = classify_intent_and_slots(llm, "gibberish")
    assert result.intent == "unknown"
    assert result.intent_confidence == 0.0
    assert llm.calls == 2  # never retried a third time


def test_classify_handles_empty_tool_calls():
    llm = FakeLLM([FakeCompletion(FakeMessage(tool_calls=None))])
    result = classify_intent_and_slots(llm, "hello")
    assert result.intent == "unknown"


def test_classify_handles_intent_outside_enum():
    llm = FakeLLM([_completion_with_args('{"intent": "made_up_intent", "confidence": 0.9, "slots": {}}')])
    result = classify_intent_and_slots(llm, "something")
    assert result.intent_confidence == 0.0


def test_hallucinated_slot_not_in_message_is_discarded():
    """Regression test for ERRORS.md #5: a small model parroted the schema's
    example order number (ORD-1001) even though the customer's message
    never mentioned any order number at all."""
    llm = FakeLLM(
        [_completion_with_args('{"intent": "cancel_order", "confidence": 0.9, "slots": {"order_id": "ORD-1001"}}')]
    )
    result = classify_intent_and_slots(llm, "How do I cancel my order before it ships?")
    assert result.slots == {}  # hallucinated value discarded — not present in the query


def test_genuine_slot_present_in_message_is_kept():
    llm = FakeLLM(
        [_completion_with_args('{"intent": "cancel_order", "confidence": 0.9, "slots": {"order_id": "ORD-1003"}}')]
    )
    result = classify_intent_and_slots(llm, "Please cancel order ORD-1003")
    assert result.slots == {"order_id": "ORD-1003"}


def test_slot_grounding_is_case_insensitive():
    llm = FakeLLM(
        [_completion_with_args('{"intent": "track_order", "confidence": 0.9, "slots": {"order_id": "ord-1002"}}')]
    )
    result = classify_intent_and_slots(llm, "where is ORD-1002")
    assert result.slots == {"order_id": "ord-1002"}
