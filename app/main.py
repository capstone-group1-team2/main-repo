"""FastAPI app: /chat /health /metrics /escalations (ARCHITECTURE.md §10).

Every shared resource (embedder, retriever, llm client, session store,
escalation handler, agent, and a shared Redis client for liveness/metrics)
is constructed exactly once in `lifespan()` and stored on `app.state` — no
route handler ever constructs its own client, per §10.2's wiring shape.
Route handlers are plain `def`, not `async def`: every dependency here
(Groq, Weaviate, Neo4j, Redis) uses a synchronous client, and FastAPI runs
sync route handlers in a thread pool automatically, so a slow call to any
of them doesn't block the event loop for other requests.
"""

from __future__ import annotations

import datetime
from contextlib import asynccontextmanager

import redis as redis_lib
from fastapi import FastAPI, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Gauge, generate_latest

import app.config as cfg
from agent.agent import Agent
from agent.escalation import EscalationHandler
from agent.llm_client import DAILY_CALL_COUNTER_KEY_PREFIX, LLMClient
from agent.session_state import SessionStateStore
from app.schemas import AgentResponse, ChatRequest, EscalationPacket
from ingestion.embedder import Embedder
from retrieval.graph_retriever import GraphRetriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.weaviate_retriever import WeaviateRetriever
from logging_config import configure_logging

configure_logging(cfg.LOG_LEVEL)

_METRICS_REGISTRY = CollectorRegistry()
_GROQ_DAILY_CALLS = Gauge(
    "groq_daily_calls_total",
    "Number of Groq API calls made today (UTC), from agent.llm_client's shared counter.",
    registry=_METRICS_REGISTRY,
)
_ESCALATIONS_TOTAL = Gauge(
    "escalations_total", "Total escalations ever recorded.", registry=_METRICS_REGISTRY
)
_ESCALATIONS_SLACK_FAILED = Gauge(
    "escalations_slack_undelivered_total",
    "Escalations whose Slack notification failed and may need manual resending.",
    registry=_METRICS_REGISTRY,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    embedder = Embedder(model_name=cfg.EMBEDDING_MODEL_NAME)
    retriever = HybridRetriever(
        vec=WeaviateRetriever(embedder, cfg.WEAVIATE_HOST, cfg.WEAVIATE_PORT, cfg.WEAVIATE_GRPC_PORT, cfg.WEAVIATE_COLLECTION_NAME),
        graph=GraphRetriever(cfg.NEO4J_URI, cfg.NEO4J_USERNAME, cfg.NEO4J_PASSWORD),
        top_k=cfg.RETRIEVAL_TOP_K,
    )
    llm = LLMClient(
        api_key=cfg.GROQ_API_KEY,
        default_model=cfg.GROQ_MODEL_DEFAULT,
        upgrade_model=cfg.GROQ_MODEL_UPGRADE,
        redis_url=cfg.REDIS_URL,
    )
    session_store = SessionStateStore(redis_url=cfg.REDIS_URL)
    escalation = EscalationHandler(store_path=cfg.ESCALATION_STORE_PATH, slack_webhook_url=cfg.SLACK_WEBHOOK_URL)

    app.state.agent = Agent(
        retriever=retriever, llm=llm, embedder=embedder, session_store=session_store, escalation=escalation
    )
    app.state.redis = redis_lib.Redis.from_url(cfg.REDIS_URL, decode_responses=True, socket_connect_timeout=2)

    yield

    retriever.close()
    app.state.redis.close()


app = FastAPI(title="Agentic Customer Support Assistant — backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/chat", response_model=AgentResponse)
def chat(request: ChatRequest) -> AgentResponse:
    return app.state.agent.handle_message(request.query, request.session_id)


@app.get("/health")
def health(response: Response):
    agent = app.state.agent
    checks = {
        "weaviate": agent.retriever.vec.is_ready(),
        "neo4j": agent.retriever.graph.is_reachable(),
        "redis": _check_redis(),
    }
    all_ok = all(checks.values())
    response.status_code = 200 if all_ok else 503
    return {"status": "ok" if all_ok else "degraded", "dependencies": checks}


def _check_redis() -> bool:
    try:
        return bool(app.state.redis.ping())
    except Exception:
        return False


@app.get("/metrics")
def metrics():
    today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    key = f"{DAILY_CALL_COUNTER_KEY_PREFIX}{today}"
    try:
        call_count = int(app.state.redis.get(key) or 0)
    except Exception:
        call_count = 0
    _GROQ_DAILY_CALLS.set(call_count)

    escalations = app.state.agent.escalation.list_escalations()
    _ESCALATIONS_TOTAL.set(len(escalations))
    _ESCALATIONS_SLACK_FAILED.set(sum(1 for e in escalations if not e.slack_delivered))

    return Response(generate_latest(_METRICS_REGISTRY), media_type=CONTENT_TYPE_LATEST)


@app.get("/escalations", response_model=list[EscalationPacket])
def list_escalations(slack_delivered: bool = Query(default=None)) -> list:
    return app.state.agent.escalation.list_escalations(slack_delivered=slack_delivered)
