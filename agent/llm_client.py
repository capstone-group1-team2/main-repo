"""Thin Groq wrapper: model tiering, retry, daily call counter, circuit
breaker (ARCHITECTURE.md §5, §8.1). Every Groq call in the whole codebase
goes through this one class.

Cross-ownership interface (ARCHITECTURE.md §15): the daily call counter's
Redis key format is `groq:daily_calls:{YYYY-MM-DD}` (UTC date), a plain
INCR-ed integer with its TTL set to expire at the next UTC midnight on the
first increment of the day. Owner D's M5 `/metrics` route reads this same
key format — see DAILY_CALL_COUNTER_KEY_PREFIX below, do not change it here
without updating that route too.
"""

from __future__ import annotations

import datetime
import logging

import time

import groq
import redis

from app.config import (
    GROQ_API_KEY,
    GROQ_CIRCUIT_BREAKER_MIN_REMAINING_PCT,
    GROQ_MAX_RETRIES,
    GROQ_MODEL_DEFAULT,
    GROQ_MODEL_UPGRADE,
    GROQ_RETRY_BACKOFF_SECONDS,
    REDIS_URL,
)

logger = logging.getLogger("agent.llm_client")

DAILY_CALL_COUNTER_KEY_PREFIX = "groq:daily_calls:"

# Fallback daily token limits (ARCHITECTURE.md §5's table) used only when
# Groq's response doesn't include an `x-ratelimit-limit-tokens` header.
_KNOWN_DAILY_TOKEN_LIMITS = {
    "llama-3.1-8b-instant": 500_000,
    "llama-3.3-70b-versatile": 100_000,
}

_RETRYABLE_EXCEPTIONS = (
    groq.APIConnectionError,
    groq.APITimeoutError,
    groq.InternalServerError,
    groq.RateLimitError,
)


class LLMClient:
    def __init__(
        self,
        api_key: str = GROQ_API_KEY,
        default_model: str = GROQ_MODEL_DEFAULT,
        upgrade_model: str = GROQ_MODEL_UPGRADE,
        redis_url: str = REDIS_URL,
        max_retries: int = GROQ_MAX_RETRIES,
        retry_backoff_seconds: float = GROQ_RETRY_BACKOFF_SECONDS,
        circuit_breaker_min_remaining_pct: float = GROQ_CIRCUIT_BREAKER_MIN_REMAINING_PCT,
        groq_client=None,
        redis_client=None,
    ):
        """`groq_client`/`redis_client`, if given, are used as-is — lets
        tests inject fakes without a real Groq API key or a live Redis."""
        self._client = groq_client if groq_client is not None else groq.Groq(api_key=api_key)
        self.default_model = default_model
        self.upgrade_model = upgrade_model
        self._redis = redis_client if redis_client is not None else redis.Redis.from_url(redis_url, decode_responses=True)
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.circuit_breaker_min_remaining_pct = circuit_breaker_min_remaining_pct
        self._circuit_tripped = False

    @property
    def circuit_tripped(self) -> bool:
        return self._circuit_tripped

    def complete(self, messages, model=None, tools=None, tool_choice=None, temperature: float = 0.0, max_tokens=None):
        """Returns the parsed ChatCompletion. `model` defaults to
        `default_model`; the circuit breaker can silently override any
        requested model back down to `default_model` once tripped."""
        requested_model = model or self.default_model
        effective_model = self.default_model if self._circuit_tripped else requested_model
        if effective_model != requested_model:
            logger.info("Circuit breaker active — using %s instead of requested %s.", effective_model, requested_model)

        kwargs = {"model": effective_model, "messages": messages, "temperature": temperature}
        if tools is not None:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        last_exception = None
        for attempt in range(1, self.max_retries + 1):
            # Counted before the request, not after a successful parse: this
            # is meant to track real outbound calls against Groq's quota
            # (§8.1's "burn through the daily budget" concern), so a call
            # that Groq rejects (e.g. a malformed tool call) still counts —
            # it still consumed a real HTTP round trip against the account's
            # rate limit, even though it produced no usable completion.
            self._increment_daily_call_counter()
            try:
                raw = self._client.chat.completions.with_raw_response.create(**kwargs)
                self._check_circuit_breaker(effective_model, raw.headers)
                return raw.parse()
            except _RETRYABLE_EXCEPTIONS as e:
                last_exception = e
                if attempt < self.max_retries:
                    logger.warning("Groq call failed (attempt %d/%d): %s — retrying.", attempt, self.max_retries, e)
                    time.sleep(self.retry_backoff_seconds * attempt)
                    continue
        raise last_exception

    def _check_circuit_breaker(self, model: str, headers) -> None:
        if self._circuit_tripped:
            return
        remaining = headers.get("x-ratelimit-remaining-tokens")
        limit = headers.get("x-ratelimit-limit-tokens") or _KNOWN_DAILY_TOKEN_LIMITS.get(model)
        if remaining is None or limit is None:
            return
        try:
            remaining, limit = float(remaining), float(limit)
        except (TypeError, ValueError):
            return
        if limit <= 0:
            return
        pct_remaining = remaining / limit
        if pct_remaining < self.circuit_breaker_min_remaining_pct:
            self._circuit_tripped = True
            logger.warning(
                "Circuit breaker TRIPPED for model=%s: %.1f%% tokens remaining (< %.0f%% threshold). "
                "Forcing all subsequent calls in this process to %s.",
                model, pct_remaining * 100, self.circuit_breaker_min_remaining_pct * 100, self.default_model,
            )

    def _increment_daily_call_counter(self) -> int:
        key = f"{DAILY_CALL_COUNTER_KEY_PREFIX}{datetime.datetime.now(datetime.timezone.utc).date().isoformat()}"
        count = self._redis.incr(key)
        if count == 1:
            now = datetime.datetime.now(datetime.timezone.utc)
            next_midnight = (now + datetime.timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            ttl_seconds = int((next_midnight - now).total_seconds())
            self._redis.expire(key, ttl_seconds)
        return count
