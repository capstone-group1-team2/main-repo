"""The baseline the full agent is measured against (Project Proposal,
"Baseline for comparison"; Module 12 guide §4, option 3).
 
This is the M5-M11 reference implementation: a plain retrieval-and-generation
pipeline. It has, deliberately and by definition:
 
  - the SAME hybrid retriever the agent uses (dense + BM25, RRF-fused)
  - the SAME generation call (agent/generator.py, same prompt, same model)
  - NO intent routing        -> it answers everything
  - NO tool calling          -> it cannot cancel or track an order
  - NO groundedness retry    -> one shot, no broaden_via_graph()
  - NO escalation logic      -> it never hands off to a human
 
That last group is the point. The baseline establishes what a straightforward
application of our RAG techniques achieves BEFORE the routing, self-checking,
and escalation work — so any lift we report is attributable to those additions
and not to retrieval or the LLM.
 
**On the baseline's "route".** The baseline has no router, so it cannot emit a
route. But it still has an implicit, unavoidable routing policy: it answers
every message from retrieved policy text. That is exactly equivalent to a
constant predictor that always outputs "information". We score it that way
rather than declaring routing accuracy inapplicable, because:
 
  1. It gives the primary metric a real scale, which is the entire purpose of
     a baseline (guide §4). "The agent scored 0.93" means nothing alone.
  2. It is the honest formalization of "what if we skipped the routing
     entirely?" — which is the question the baseline exists to answer.
  3. On THIS held-out set, always-"information" scores 0.75 (45 of 60), because
     the set is naturally information-heavy. That is a deliberately unflattering
     baseline for us to beat, not a strawman — see the guide's "cherry-picked
     baseline" pitfall. It also demonstrates precisely why routing accuracy
     alone is insufficient and escalation recall is reported alongside it: the
     baseline's escalation recall is 0.00, and no amount of majority-class
     accuracy hides that.
"""
 
from __future__ import annotations
 
from dataclasses import dataclass
 
from agent.generator import generate_answer
 
 
@dataclass(frozen=True)
class BaselineResult:
    """Mirrors the subset of `AgentResponse` the harness scores."""
 
    route: str  # always "information" — see the module docstring
    detail: str
    retrieval_result: object  # retrieval.hybrid_retriever.RetrievalResult
 
 
class RagBaseline:
    """Retrieve once, generate once, return the answer. No decisions made."""
 
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
 
    def answer(self, query: str) -> BaselineResult:
        # One retrieval. No graph broadening, no second pass, no confidence gate.
        retrieval_result = self.retriever.retrieve(query)
 
        # One generation. Whatever comes back is the final answer — the baseline
        # has no groundedness check, so it cannot know if this is supported by
        # the chunks, and no escalation path, so it could not act on that even
        # if it did.
        detail = generate_answer(self.llm, query, retrieval_result.chunks)
 
        return BaselineResult(route="information", detail=detail, retrieval_result=retrieval_result)
 