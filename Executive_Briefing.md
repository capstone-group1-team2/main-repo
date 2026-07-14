# Executive Briefing — Meridian Assist

**Agentic Customer Support Assistant | AI.SPIRE Capstone, Group 1 Team 2**

---

## 1. Executive Summary

Meridian Assist is an AI agent that answers customer-support questions for a fictional online retailer, Meridian Retail, by retrieving and citing the company's own policy documents, performing real order actions (cancel, track), and escalating to a human the moment it cannot verify an answer. It is not a chatbot: it is a routed, self-checking, tool-using agent built on Retrieval-Augmented Generation (RAG), a hybrid vector/keyword retriever, a knowledge graph for topic broadening, and a hard groundedness gate that blocks fabricated answers before they reach a customer.

Against a frozen 60-example held-out test set, the system achieved **0.917 routing accuracy** (target ≥0.85) and **1.000 groundedness** (target ≥0.90), both clearing their bar with zero variance across three independent runs. Against a plain-RAG baseline with no routing, tools, or escalation, the gap is stark: baseline escalation recall is **0.000** — a conventional chatbot structurally cannot hand a hard case to a human — while the agent's is **0.625**. That single comparison is the business case for building an agent instead of deploying a generic chatbot.

## 2. The Problem & Stakeholder

For a support team at an online retailer buried in repetitive tickets, the same "where's my order?" and "what's your refund window?" questions flood in daily, while genuinely hard cases wait behind them in the same queue. Today, teams handle this in one of two unsatisfying ways: answering every ticket by hand, which does not scale, or deploying a generic chatbot that will — eventually, confidently — invent a policy it does not actually know, quietly damaging customer trust at the exact moment a customer needed a correct answer.

Meridian Assist is built around a narrower, more defensible premise than "automate support": an agent that knows the difference between a question it can answer, an action it can safely perform, and a moment where the right move is to step back and bring in a person.

## 3. What We Built

The system is an **agentic RAG pipeline**, not a single retrieval-and-generate loop. Every customer message is classified into one of 27 intents and routed to exactly one of three paths:

- **Information** — retrieve relevant policy chunks, generate an answer, and verify it against a groundedness check before it is ever shown to the customer. If the check fails, the system retries once by broadening retrieval through a knowledge graph of related policy categories; if it still fails, the message is escalated rather than answered.
- **Action** — a small, explicit set of real operations (cancel an order, track an order) executed against a backend order system, not simulated in text.
- **Escalate** — the system captures a way to contact the customer, writes a durable record, and notifies a human support channel, rather than guessing.

This routing logic, the self-verification step, and the escalation path are precisely what separate an agent from a chatbot, and they are the reason the evaluation compares the full system against a plain-RAG baseline that has none of them.

## 4. Architecture

**Data source.** An openly-licensed customer-support dataset (Bitext, ~27,000 rows) was curated into 11 policy documents covering accounts, orders, cancellations, refunds, payments, shipping, subscriptions, and related topics.

**Ingestion (offline).** Documents are split by semantic meaning into roughly 93 chunks, embedded with `bge-large-en-v1.5`, and stored in two systems: **Weaviate**, a vector database supporting both dense semantic search and BM25 keyword search, and **Neo4j**, a graph database recording which policy categories are conceptually related to one another. Ingestion is idempotent and self-healing: content is identified by a hash of its text, so re-running ingestion after a partial failure or an independently-reset database repairs only what is actually missing, without duplicating or re-processing anything untouched.

**Agent (per request).** A single Groq LLM call classifies intent and extracts any relevant details (such as an order number) from the customer's message. A hybrid retriever fuses dense and keyword search results using Reciprocal Rank Fusion. If the resulting answer fails a local, cost-free groundedness check — cosine similarity between the generated answer and the retrieved text, computed without any additional LLM call — the system consults the knowledge graph for related categories, searches those specifically, and tries once more before escalating. **Redis** holds short-lived session state so a clarifying follow-up question is understood in context.

**Serving.** A Next.js chat frontend serves customers; escalations are logged to SQLite and posted to a Slack channel for human follow-up. The full stack runs as a single Docker Compose deployment.

## 5. Evaluation Methodology

**Primary metric:** routing accuracy — did the agent's chosen path (information, action, or escalate) match a human-labeled expected path.

**Held-out set:** 60 hand-labeled examples, split 45 information / 7 action / 8 escalate, deliberately over-sampling thinner categories (CANCEL, SUBSCRIPTION) rather than mirroring the raw dataset's natural proportions, since those are the categories the action-routing and escalation logic depend on most. Five examples are hand-written adversarial questions — fluent but off-topic, or deliberately ambiguous — because the source dataset is too clean to naturally contain anything that should be escalated.

**Baseline:** a plain retrieval-and-generation pipeline with no intent routing, no tool-calling, no groundedness retry, and no escalation logic — establishing what a straightforward RAG system achieves before the team's routing, self-checking, and escalation additions.

**Stochastic handling:** generation and intent classification are not deterministic, so the full held-out set was run three times with independent seeds (42, 1337, 2024); Groq exposes no seed parameter, so these are genuinely independent samples of the model's real variance rather than an artificially pinned repeat. Retrieval is deterministic given the frozen corpus and was computed once and reused across seeds.

**Error analysis:** results were grouped across two dimensions — policy category and difficulty tier (easy/medium/hard) — to find where errors concentrate rather than only how many occur.

## 6. Results

| Metric | Baseline (plain RAG) | Meridian Assist |
|---|---|---|
| **Routing accuracy** (primary) | 0.750 | **0.917 ± 0.000** ✅ (target ≥0.85) |
| Escalation recall | 0.000 | **0.625 ± 0.000** |
| Escalation precision | n/a (never escalates) | **0.714 ± 0.000** |
| **Groundedness** (C2) | 0.967 | **1.000 ± 0.000** ✅ (target ≥0.90) |
| Recall@k (retrieval quality) | 0.956 | 0.956 *(identical — shared retriever)* |
| Answer match (diagnostic) | 0.822 ± 0.021 | 0.841 ± 0.010 |
| Latency, p95 (optional) | — | 23.7s ❌ (target <3s) |

All routing-decision metrics were **identical across all three random seeds** (standard deviation of 0.000) — the same five examples were misrouted every single run. This indicates the agent's discrete routing decisions are stable, not a favorable roll of the dice; only the exact wording of generated text varied run to run, which is exactly the kind of variance three-seed evaluation is designed to surface honestly rather than hide.

**On the 100% groundedness result:** this is a real, structurally-earned number, not an artifact to be embarrassed by, but it deserves one honest caveat. The agent only ever emits an information answer after that answer has already passed the groundedness check; anything that fails is retried once and, failing again, escalated instead of shown to the customer. Across all 180 example-runs in this evaluation, **zero** answers ever reached that final escalation-for-ungroundedness path — the safety net exists, is implemented, and is unit-tested, but was never actually needed to intervene in this run. The team validated this behavior further through targeted manual testing: six deliberately adversarial questions designed to force the model into inventing a specific fact (an exact dollar refund amount, a definite yes/no on an uncovered policy) were each met with an honest admission of the gap rather than a fabricated answer.

**On the latency failure:** this criterion is optional and the failure is a measurement-environment artifact rather than an architectural one. The embedding model runs on CPU on a development laptop rather than a GPU-backed service; the p50 latency of 14.6s versus p95 of 23.7s shows the expected split between a fast path and a slower graph-broadening retry path, and a warm, properly-hosted deployment would be materially faster.

## 7. Error Analysis

Grouped across category and difficulty, **24 of 28 populated cells were perfect (0% error).** All five routing errors concentrate in exactly four cells:

- **CANCEL × hard** (100%, 1 example) — a "cancel my order" request with no order number was misclassified into a different intent, which meant the ask-for-missing-slot-then-escalate path was never entered at all.
- **ADVERSARIAL × hard** (50%, 2 of 4) — two fluent but off-topic questions (a tax question, a labor-dispute question) were answered instead of escalated, because the model found tangentially related text and treated it as sufficient.
- **SUBSCRIPTION × easy and hard** (50% each) — two legitimate subscription questions were escalated because that part of the underlying corpus is genuinely too thin to ground an answer. This is arguably the *correct* behavior scored as an error against an optimistic held-out label, and the team logged it rather than editing the frozen test set.

One predicted failure mode was **tested and refuted**: an order-ID case-sensitivity mismatch was hypothesized to silently escalate valid cancellation requests, but all seven real-order action examples routed correctly in every seed. The underlying code risk still exists but did not manifest in this evaluation.

**Next-iteration hypothesis:** add an explicit topical out-of-scope gate before the information route commits to answering, because every non-corpus error was the agent answering something it should have refused — not a failure of retrieval (0.956) or groundedness (1.000), both of which were already strong.

## 8. Business Interpretation

The comparison that matters most to a support-operations stakeholder is not the routing-accuracy gap alone, but what sits behind it: a plain RAG chatbot **cannot escalate at all** by construction, while this system correctly identifies and hands off the majority of cases that genuinely need a human, at 0.625 recall against zero. In practice, that is the difference between a tool a support team can trust to sit in front of customers unsupervised on routine questions, and one that requires constant human oversight to catch its confident mistakes. The escalation path is not a fallback bolted on for safety theater — it durably records every handoff with the reason it occurred, which is itself a diagnostic asset: the categorized reasons collected here already point at concrete next investments (enriching thin corpus sections, adding an out-of-scope filter) rather than vague "the AI needs more training" guesses.

## 9. Limitations & Future Work

The evaluation corpus is derived from a conversational customer-service dataset rather than authored policy reference material, and this shows: the subscription-policy section is a single thin passage with almost no concrete facts, which is the direct cause of every subscription-related error observed. A related, unresolved artifact exists in the account-recovery section, which currently describes resetting a PIN rather than a password — a leftover inconsistency from the source data that the team has identified but not yet corrected. More broadly, the corpus has no mechanism to detect if two documents were ever added that quietly contradicted one another; the groundedness check verifies an answer is supported by at least one retrieved chunk, which is the right design for catching fabrication, but is structurally unable to catch two true-sounding chunks that disagree with each other.

On the operational side, several gaps were identified through direct code review that would need to be closed before any real deployment, foremost among them: the order-cancellation and order-tracking tools currently perform no ownership check against the customer making the request, and the escalation-log and metrics endpoints are unauthenticated. Neither is exploitable against the current mock order data, but both are architectural gaps, not implementation details, and are the team's top-priority items before this system could safely handle real customer data. The small language model powering classification occasionally proposes an intent label outside its allowed set, which the system catches and retries safely, but at a measurable token cost worth monitoring at scale. Finally, the measured latency reflects a CPU-bound development environment rather than the system's real architectural ceiling, and should be re-measured against a GPU-backed or managed embedding service before being used to make a production capacity decision.

The team's next concrete step, directly motivated by this evaluation's own error analysis rather than a general instinct to "improve the model," is a topical out-of-scope gate ahead of the information route — closing the one category of error the data actually shows, without touching the retrieval and groundedness machinery that is already performing well.
