# System Architecture
 
One-page view of how a request flows through the system, and how the corpus gets there in the first place. For full design rationale, thresholds, and ownership, see `ARCHITECTURE.md`.
 
## Request flow
 
```mermaid
flowchart TB
    User["User (browser)"] -->|"types a message"| FE["Next.js frontend<br/>single chat page"]
    FE -->|"POST /chat"| API["FastAPI backend<br/>app/main.py"]
 
    API --> Agent["Agent.handle_message()<br/>agent/agent.py"]
 
    Agent -->|"1. cheap, no LLM call"| Smalltalk["smalltalk.py<br/>greeting/thanks/farewell?"]
    Agent -->|"2. one Groq call"| Intent["intent_classifier.py<br/>intent + slots"]
    Agent -->|"per-turn state"| Redis[("Redis<br/>session_state.py<br/>15-min TTL")]
 
    Intent --> Route{"route decision"}
 
    Route -->|"information"| Retrieve["HybridRetriever.retrieve()<br/>retrieval/hybrid_retriever.py"]
    Retrieve --> Weaviate[("Weaviate<br/>dense + BM25 search")]
    Retrieve --> Neo4j1[("Neo4j<br/>related categories")]
    Retrieve --> Gen["generator.py<br/>Groq: answer from chunks"]
    Gen --> Ground{"groundedness.py<br/>supported by chunks?"}
    Ground -->|"no — retry once"| Broaden["broaden_via_graph()<br/>widen via Neo4j categories"]
    Broaden --> Gen
    Ground -->|"yes"| Answer["answer returned to user"]
 
    Route -->|"action"| Tools["tools.py<br/>cancel_order() / track_order()"]
    Tools --> MockOrders[("data/mock_orders.json<br/>simulated backend")]
 
    Route -->|"unclear / low confidence /<br/>still ungrounded / order not found"| Escalate["escalation.py<br/>capture contact → store → Slack"]
    Ground -->|"no — after retry"| Escalate
    Escalate --> SQLite[("SQLite<br/>escalations.db")]
    Escalate --> Slack["Slack webhook"]
 
    Agent --> LLM["llm_client.py<br/>Groq wrapper: retry,<br/>daily counter, circuit breaker"]
    Intent -.-> LLM
    Gen -.-> LLM
 
    API -->|"GET /health /metrics /escalations"| Ops["Ops endpoints"]
```
 
## Ingestion pipeline (offline — rerun when the corpus changes)
 
```mermaid
flowchart LR
    Corpus["data/corpus/*.md<br/>11 policy docs"] --> Hash{"hash_store.py<br/>content changed?"}
    Hash -->|"unchanged"| Skip["skip — no reprocessing"]
    Hash -->|"changed/new"| Chunk["chunker.py<br/>heading → sentence<br/>semantic chunking"]
 
    Chunk --> Embed["embedder.py<br/>bge-large-en-v1.5"]
    Embed --> WLoad["weaviate_loader.py<br/>upsert changed chunks"]
    WLoad --> Weaviate2[("Weaviate")]
 
    Chunk --> GBuild["graph_builder.py<br/>link chunk → category"]
    Seed["graph_seed.py<br/>CATEGORY_RELATIONS<br/>(the one hand-written input)"] --> GBuild
    GBuild --> Neo4j2[("Neo4j")]
```
 
## What each box actually is
 
| Layer | Component | Role |
|---|---|---|
| Frontend | Next.js chat page | Anonymous `session_id` (localStorage), sends `/chat`, renders route/confidence badges |
| Backend | FastAPI (`app/main.py`) | Wires every shared client once at startup; `/chat`, `/health`, `/metrics`, `/escalations` |
| Decision core | `agent/agent.py` | The only place that decides information vs. action vs. escalate |
| Retrieval | Weaviate + Neo4j + `hybrid_retriever.py` | Dense + BM25 fusion (RRF), graph-broadened retry on failed groundedness |
| LLM | Groq (via `llm_client.py`) | Intent+slot extraction, answer generation — one wrapper, retry + circuit breaker |
| Action | `tools.py` + mock order store | Real function calls against a clearly-fake backend |
| Escalation | `escalation.py` + SQLite + Slack | Captures contact, stores permanently, posts live |
| Session memory | Redis | Pending question state, last completed order (one-turn memory) |
 
 