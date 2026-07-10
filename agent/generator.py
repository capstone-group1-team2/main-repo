"""Groq answer generation from retrieved chunks (ARCHITECTURE.md §8) — an
actual synthesized answer, not a raw chunk dump or template concatenation.
"""

from __future__ import annotations

from agent.llm_client import LLMClient

_SYSTEM_PROMPT = (
    "You are a customer support assistant for Meridian Retail. Answer the "
    "customer's question using ONLY the information in the provided policy "
    "excerpts below. Be concise and direct. If the excerpts don't actually "
    "answer the question, say so plainly instead of guessing."
)


def generate_answer(llm: LLMClient, query: str, chunks: list, model: str = None) -> str:
    context = "\n\n".join(f"[{c.category}/{c.heading}]\n{c.text}" for c in chunks)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"Policy excerpts:\n{context}\n\nCustomer question: {query}"},
    ]
    completion = llm.complete(messages=messages, model=model, temperature=0.2)
    return completion.choices[0].message.content.strip()
