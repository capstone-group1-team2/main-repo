# Runbook

Operational guide for diagnosing problems once the system is running. For setup instructions see `README.md`; for design rationale see `ARCHITECTURE.md`.

## 1. First response — check health and metrics

```
GET /health     → {"status": "ok" | "degraded", "dependencies": {weaviate, neo4j, redis}}
GET /metrics    → Prometheus format: groq_daily_calls_total, escalations_total,
                   escalations_slack_undelivered_total
```

`/health` returning `503 degraded` tells you *which* dependency is down (`weaviate`/`neo4j`/`redis` each report `true`/`false` independently) — start there before looking at anything else.

## 2. Known failure modes

| Symptom | Likely cause | Where to look | Fix |
|---|---|---|---|
| `/health` shows `weaviate: false` or `neo4j: false` | Docker container down, or reachable but the *stores are out of sync* (one was reset independently) | `docker ps`; container logs | Restart the container. If content hashes match but data is missing, `python -m ingestion.ingest` self-heals automatically (see `ingestion/ingest.py`'s self-heal check) — no manual fix needed. |
| `/health` shows `redis: false` | Redis down or unreachable | `docker ps`; check `REDIS_URL` in `.env` | Restart Redis. Sessions are ephemeral (15-min TTL) — no data loss beyond in-flight conversations. |
| Agent silently answers with a stale/default model despite requesting the upgrade tier | Groq circuit breaker tripped — daily token quota fell below `GROQ_CIRCUIT_BREAKER_MIN_REMAINING_PCT` | JSON logs, `agent.llm_client` — look for `"Circuit breaker TRIPPED"` | Expected behavior under quota pressure, not a bug. Resets on process restart or once Groq's quota window rolls over. If it's tripping too early, revisit the threshold in `.env`. |
| `groq_daily_calls_total` climbing unexpectedly fast | Retry storms, or a bug causing repeated classification/generation calls per turn | JSON logs filtered to `agent.llm_client` | Check for exceptions triggering `_RETRYABLE_EXCEPTIONS` retries; each retry increments the counter even on failure, by design. |
| `escalations_slack_undelivered_total` > 0 | Slack webhook failing (bad URL, Slack outage, rate limit) | `GET /escalations?slack_delivered=false`; JSON logs, `agent.escalation` — `"Slack notification failed"` | Escalations are never lost — the packet is written to SQLite *before* the Slack attempt. Fix `SLACK_WEBHOOK_URL` if wrong, or manually notify the team from the `/escalations` list until Slack recovers. |
| A customer reports "the bot ignored me" / didn't escalate when it should have | Check the routing decision that was actually made | JSON logs — filter by `session_id` across `agent.intent_classifier`, `agent.escalation` | Compare `intent_confidence`/`retrieval_confidence` against `MIN_INTENT_CONFIDENCE`, `MIN_ESCALATION_DENSE_FLOOR`, `MIN_ESCALATION_BM25_FLOOR` in `.env`. See `eval/failure_cases.md` for documented misclassification patterns (e.g. a cancel-shaped request misclassified as `contact_customer_service` never reaches the escalation safety net). |
| Retrieval returns irrelevant chunks / agent escalates too often on legitimate questions | Corpus gap, or thresholds mis-calibrated for a phrasing style not in the held-out set | `eval/run_eval.py` error grid, `eval/failure_cases.md` | Check if the failing category is corpus-thin (e.g. `subscription.md` historically). Corpus fix > threshold fix — see failure_cases.md's SUBSCRIPTION analysis. |
| Ingestion run does nothing (`0 chunks processed`) on a real corpus edit | Content hash matched — the edit may not have actually changed the file, or you edited the wrong path | `ingestion/hash_store.json` | Confirm `CORPUS_DIR` in `.env` points at the file you edited. |

## 3. Where to look, in order

1. **JSON logs** (stdout, structured — see `logging_config.py`) filtered by `logger` field: `agent.llm_client`, `agent.escalation`, `ingestion.ingest`, `retrieval.hybrid_retriever`.
2. **`/health`** for dependency status.
3. **`/metrics`** for daily Groq usage and escalation counts.
4. **`GET /escalations`** for the actual escalation record tied to a session, including `attempted_summary` — a plain-English trace of what the agent tried before giving up.

## 4. Who to escalate to

| Area | Owner | Files |
|---|---|---|
| Ingestion, Neo4j graph | Owner A | `ingestion/` |
| Retrieval, groundedness | Owner B | `retrieval/`, `agent/groundedness.py` |
| Routing, intent, generation, sessions, tools | Owner C | `agent/agent.py`, `agent/intent_classifier.py`, `agent/generator.py`, `agent/session_state.py`, `agent/tools.py` |
| Escalation, API, frontend | Owner D | `agent/escalation.py`, `app/`, `frontend/` |

If the failure spans layers (e.g. a routing bug that only manifests because of a corpus gap), start with whoever owns the *first* component in the chain that misbehaved, per the logs.

## 5. Things that are expected behavior, not incidents

- Circuit breaker downgrading to the default model under quota pressure.
- Escalating on unintelligible or off-topic input (adversarial queries) — this is the safety net working, not a bug.
- Sessions disappearing after 15 minutes of inactivity — no login, by design.
- Empty Weaviate/Neo4j right after a fresh `docker-compose up` — run `python -m ingestion.ingest` once.
