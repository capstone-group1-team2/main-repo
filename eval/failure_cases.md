# Failure Analysis
 
Rubric R5 / Module 12 guide §7. Five documented, **illustrative** failure cases
(not randomly chosen), grouped against the `category × difficulty` error grid
printed by `python -m eval.run_eval`, ending in one specific next-iteration
hypothesis.
 
**Status: from a real, completed run.** 60 held-out examples × 3 seeds
[42, 1337, 2024], run 2026-07-11. Routing accuracy 0.917 (mean, std=0.000
across all three seeds — the discrete routing decisions were identical every
run; only continuous scores like groundedness and answer-match varied, as
expected from a sampling LLM). Every claim below was checked directly against
`per_seed.<seed>.agent` rows in `results.json`, not assumed from the printed
summary.
 
Of the five original candidates: **two confirmed, one partially confirmed,
two refuted.** Both refuted candidates are kept below rather than deleted —
a prediction the run refutes is as useful as one it confirms, because it
means a code path we were worried about is actually sound. They are replaced
in the "five illustrative cases" list by the two real failures the error
grid surfaced instead (a misclassification bug and an adversarial miss).
 
---
 
## Candidate 1 — Order IDs may fail an exact-match lookup after slot extraction
 
- **Example:** `hold-0005` — "Please cancel order ORD-1001."
- **Expected route:** `action`
- **Predicted route:** `action` ✅ (all three seeds)
- **Verdict: REFUTED.**
**What the run actually showed.** All 7 action-route examples with real,
existing order IDs (`ORD-1001` through `ORD-1006`) routed to `action`
correctly, in every seed. The only `reason="order_not_found"` escalations in
the entire run were `hold-0021` ("Can you check ORD-9999?") and `hold-0022`
("Please cancel ORD-8888...") — both **deliberately nonexistent** order IDs,
written into the held-out set specifically to test "order genuinely doesn't
exist → escalate." Both escalated correctly, matching their expected label.
The casing mismatch this candidate predicted never manifested: the 8B model
preserved the customer's original casing (`ORD-1001`, not `ord-1001`) on
every real-order example this run.
 
**Why we're keeping the prediction in the write-up anyway.** The underlying
code path is still real — `_grounded_slots()` is still case-insensitive and
`tools._find_order()` is still case-sensitive — so the risk hasn't been
*fixed*, it just wasn't *triggered* this run. That's a meaningfully different
claim from "this isn't a bug," and it's worth saying precisely: the latent
risk remains, but it did not cost us any routing accuracy on this held-out
set. The cheap fix (case-normalize at the `tools._find_order()` boundary)
is still worth doing as a defensive measure, just not urgently motivated by
this run's evidence.
 
---
 
## Candidate 2 — Short factual questions may trip the BM25 escalation floor
 
- **Example:** `hold-0033` — "Do you take Apple Pay?"
- **Expected route:** `information`
- **Predicted route:** `information` ✅ (all three seeds)
- **Verdict: REFUTED**, and more broadly: **no `easy` cell shows a higher
  error rate than its `medium` counterpart anywhere in the grid.**
**What the run actually showed.** `hold-0033` routed correctly every seed.
More importantly, the counter-intuitive pattern this candidate would predict
— `easy` cells failing more than `medium` ones — does not appear anywhere.
Every `easy` cell across all 12 categories is either 0.00 or matches its
`medium` sibling, **except** SUBSCRIPTION, where `easy` and `hard` both sit
at 0.50 while `medium` is 0.00 (see Candidate 3 — a different mechanism,
corpus thinness, not query length). The BM25-floor-on-short-queries failure
mode this candidate described simply did not occur on this held-out set.
 
**Why the prediction was reasonable but didn't land.** `MIN_ESCALATION_BM25_FLOOR`
(4.0) is real and is still an absolute, length-sensitive threshold in the
code — that part of the reasoning holds. It just didn't bite any of our 27
`easy`-labeled information examples. That may say more about our held-out
set's specific phrasings (several of our "easy" questions still contain 2-3
distinctive corpus terms, e.g. "Apple Pay," "standard shipping") than about
the floor being safe in general — a held-out set with shorter, more generic
easy phrasings might still expose it. Worth flagging as a **held-out-set
limitation**, not a settled "this is fine."
 
---
 
## Candidate 3 — `subscription.md` is too thin to ground an answer
 
- **Examples:** `hold-0013` ("Will I stop getting emails immediately after I
  unsubscribe?"), `hold-0014` ("Can I unsubscribe without giving my email
  address?")
- **Expected route:** `information`
- **Predicted route:** `hold-0013` → `information` ✅ &nbsp;|&nbsp;
  `hold-0014` → `escalate` ❌ (all three seeds)
- **Verdict: PARTIALLY CONFIRMED**, and the mechanism was slightly different
  than predicted.
**What the run actually showed.** `hold-0014` escalated in every seed, but
with `reason="still_unclear_after_clarification_low_intent_confidence"` —
not `"ungrounded_after_retry"` as predicted. Its `intent` was classified as
`unknown`. So the failure occurred **one step earlier** than the candidate's
hypothesis: the classifier couldn't confidently assign an intent at all
(rather than confidently classifying it, retrieving, generating, and then
failing groundedness). `hold-0013` — the sibling example — routed correctly.
 
Zooming out: **zero examples anywhere in the entire 60×3 run carried
`reason="ungrounded_after_retry"`.** The generation→groundedness→retry→escalate
path this candidate specifically targeted was never exercised, on any
example, in any seed. That's a real, checkable fact (confirmed directly
against every row's `reason` field) and it matters for how you present
Criterion 2: your groundedness metric passed at 1.00, but it did so without
ever needing its safety net.
 
**The broader SUBSCRIPTION pattern (real, confirmed at the category level).**
SUBSCRIPTION is your single worst-performing category: 6 of 15 examples
wrong (both `easy` and `hard` cells at 50%, `medium` at 0%). The mechanism is
still corpus thinness, as predicted — `subscription.md` has exactly one `##`
section and almost no concrete facts, which is a genuine and confirmed
finding — it's just that the failure surfaces as a low-confidence
*classification*, not a failed *groundedness check*, more often than
predicted.
 
**This is arguably correct behavior scored as an error.** The agent
correctly declined to guess where the corpus doesn't support an answer. That
is a held-out-**label** issue (a human labeled these `information` assuming
the corpus covered them), not an agent bug. Per the freeze rule, we have not
edited `heldout.jsonl` — logging it here instead.
 
**Fix:** enrich `subscription.md` with real subscription-policy facts
(unsubscribe timing, whether an email is required to process the request).
This is a corpus fix, not a code fix, and based on the category-level
pattern it would likely recover most or all of the 6 SUBSCRIPTION errors at
once.
 
---
 
## Candidate 4 — `account.md` answers a *PIN* question, not a *password* one
 
- **Example:** `hold-0037` — "I forgot my password, how do I reset it?"
- **Expected route:** `information`
- **Predicted route:** `information` ✅ (all three seeds — routing was correct)
- **Verdict: REFUTED as a routing failure; the underlying corpus artifact is
  real but did not cost us the metric it was predicted to cost.**
**What the run actually showed.** `hold-0037` routed correctly in every seed,
and it does **not** appear anywhere in the mismatch list. ACCOUNT is a
perfect category across all three difficulty tiers (0.00 error rate, all 12
examples, all three seeds). We checked the specific failure signature this
candidate predicted — `recall_hit=true` **and** `answer_match=false` on the
same row — directly against the raw data, and it does not occur on this
example.
 
**Why the prediction was reasonable but didn't land.** The PIN/password
mismatch in `account.md`'s "How to Recover a Password" section is real — we
verified it by reading the corpus directly in an earlier pass of this
project, and it hasn't been fixed. It's a genuine unresolved Bitext artifact.
It simply didn't produce a *scoreable* failure on this example: either the
generator successfully paraphrased around the PIN/password inconsistency
closely enough to satisfy `answer_match`'s 0.60 cosine threshold, or the
threshold is more forgiving of this kind of near-miss than the candidate
assumed. We did not manually read the generated answer text for this
specific example as part of this pass — that would be the natural follow-up
if you want to confirm *why* it passed rather than just *that* it passed.
 
**This is a genuine limitation of the metric, not of the corpus check.**
`answer_match` is a cosine-similarity threshold, not an exact-fact check —
it can pass an answer that is directionally similar to the reference even if
it quietly substitutes the wrong noun. The corpus defect is still there and
still worth fixing; it just wasn't severe enough to move this particular
metric on this particular question, on this run.
 
---
 
## Real Failure A — Missing-slot request misclassified, so escalation never fires
 
*(Replaces a slot in the "five illustrative cases" — see the two refuted
candidates above.)*
 
- **Example:** `hold-0009` — "I want to cancel my order." (no order ID
  supplied)
- **Expected route:** `escalate`
- **Predicted route:** `information` ❌ (all three seeds)
- **Category × difficulty:** CANCEL × hard — the single worst cell in the
  grid, 100% error rate (3/3 across seeds).
**What happened.** The classifier assigned intent `contact_customer_service`
instead of `cancel_order`. `ACTION_INTENTS` is `{cancel_order, track_order}`
only, so this example never entered the ACTION branch's "ask for the missing
order number, then escalate if still not supplied" path (`agent.py`'s
`_ask_for_slot` / `awaiting_slot` flow). Instead it was treated as a plain
information request and answered from retrieval.
 
**Why this is the most instructive failure in the whole run.** It is not a
wrong-answer problem or a retrieval problem — recall@k and groundedness are
both fine on this example. It is a single upstream misclassification that
**silently removes the escalation option before the escalation logic ever
runs.** The safety net (ask-for-slot → escalate-if-missing) was never given
a chance to fire, because the router never considered this an action request
at all.
 
**Fix:** treat a cancel/track-shaped request with no resolvable order-related
slot as an escalation trigger regardless of the specific intent label the
classifier assigned it — i.e., a secondary keyword or embedding check for
"is this plausibly about an existing order" that doesn't depend entirely on
the primary intent classification being correct.
 
---
 
## Real Failure B — Fluent off-topic questions answered instead of escalated
 
*(Replaces the second slot — see the two refuted candidates above.)*
 
- **Examples:** `hold-0057` (a tax-implications-of-expensing question),
  `hold-0060` (a labor-dispute opinion question)
- **Expected route:** `escalate` (both)
- **Predicted route:** `information` ❌ (both, all three seeds)
- **Category × difficulty:** ADVERSARIAL × hard — 50% error rate (6/12
  across seeds; 2 of 4 distinct adversarial-hard examples wrong every seed).
**What happened.** `hold-0057` was classified `payment_issue` (plausible —
the question mentions "expensing" and "purchase"); `hold-0060` was
classified `complaint` (also plausible — it's phrased as a grievance).
Both are legitimate, in-scope intents. Retrieval then found loosely-related
text in `payment.md` / `feedback.md`, the generated answer cleared the
groundedness floor against that loosely-related text, and it was returned as
a confident answer to a question the corpus was never meant to address.
 
**Why this is exactly what the adversarial examples were built to find.**
These are two of the five hand-written adversarial questions in the
held-out set, deliberately fluent and superficially on-topic — precisely
because the raw Bitext dataset is too clean to naturally contain anything
that should escalate. The mechanism worked as designed: it caught 2 of 4
adversarial-hard cases failing, which is a real, quantified gap, not a
hypothetical one. (The other 2 adversarial-hard examples, plus all 3
adversarial-medium examples, escalated correctly — so the miss rate is
specific to certain phrasings, not universal.)
 
**Fix:** an explicit topical out-of-scope check before the information route
commits to answering — something that asks "is this actually a retail
support question" independent of whether retrieval happened to find
*something* similar enough to clear the groundedness floor. A relevance
floor on the *query* against the corpus's overall topic space, rather than
relying solely on the *answer*'s groundedness to the retrieved chunks.
 
---
 
## Error grid
 
Pooled across all 3 seeds (180 example-runs total: 60 examples × 3 seeds).
22 of 25 populated cells are perfect (0.00 error rate). All errors
concentrate in exactly three cells, all at the `hard` (or, for SUBSCRIPTION,
`easy`) edge of the difficulty spectrum:
 
```
category                  easy          medium            hard       ROW TOTAL
------------------------------------------------------------------------------
ACCOUNT             0.00 (0/3)      0.00 (0/6)      0.00 (0/3)     0.00 (0/12)
ADVERSARIAL                  -      0.00 (0/3)     0.50 (6/12)     0.40 (6/15)
CANCEL             0.00 (0/12)     0.00 (0/12)      1.00 (3/3)     0.11 (3/27)
CONTACT             0.00 (0/9)      0.00 (0/3)               -     0.00 (0/12)
DELIVERY            0.00 (0/6)      0.00 (0/6)               -     0.00 (0/12)
FEEDBACK            0.00 (0/9)      0.00 (0/3)               -     0.00 (0/12)
INVOICE             0.00 (0/6)      0.00 (0/3)      0.00 (0/3)     0.00 (0/12)
ORDER               0.00 (0/9)     0.00 (0/15)               -     0.00 (0/24)
PAYMENT             0.00 (0/9)      0.00 (0/3)               -     0.00 (0/12)
REFUND              0.00 (0/6)      0.00 (0/9)               -     0.00 (0/15)
SHIPPING            0.00 (0/6)      0.00 (0/6)               -     0.00 (0/12)
SUBSCRIPTION        0.50 (3/6)      0.00 (0/3)      0.50 (3/6)     0.40 (6/15)
 
Worst cells (n >= 3):
  CANCEL × hard:        100% (3/3)
  SUBSCRIPTION × easy:   50% (3/6)
  SUBSCRIPTION × hard:   50% (3/6)
  ADVERSARIAL × hard:    50% (6/12)
```
 
**What this shows.** The agent is flawless on every routine information and
action query across all 22 other populated cells — CANCEL, ORDER, PAYMENT,
REFUND, DELIVERY, SHIPPING, INVOICE, CONTACT, FEEDBACK, ACCOUNT all score
0.00 at every difficulty tier where they have examples. Every single error in
the entire 180-row run falls into one of exactly two failure types: (1) the
router failing to recognize it should escalate (CANCEL×hard's
misclassification, ADVERSARIAL×hard's off-topic-but-fluent misses), or (2)
the router correctly declining to answer from a corpus that doesn't actually
support the question (SUBSCRIPTION). Neither is a retrieval-quality or
generation-quality failure — recall@k (0.956) and groundedness (1.00) are
both excellent. The entire gap between "very good" and "excellent" on this
system lives at the escalation boundary, not in the core answer pipeline.
 
---
 
## Next-iteration hypothesis
 
> If we had another week, we would add an explicit topical-relevance /
> out-of-scope check before the information route commits to answering,
> because every routing error in this run that was not attributable to
> corpus thinness (SUBSCRIPTION) was the agent confidently answering a
> question it should have refused — the two adversarial misclassifications
> and the CANCEL-with-no-ID misclassification — while recall@k (0.956) and
> groundedness (1.00) show the retrieval and generation pipeline itself has
> no quality problem to fix. The gains left in this system are entirely at
> the escalation boundary, not in answer quality.
 
**Why this is the right hypothesis, and not the two candidates originally
templated for this section:**
 
- The BM25-floor hypothesis (Candidate 2's proposed fix) is **not supported
  by this run's evidence** — no `easy` cell underperformed its `medium`
  sibling, so a query-length-normalized BM25 score would not address
  anything this run actually surfaced.
- The order-ID-casing hypothesis (Candidate 1's proposed fix) is **also not
  supported** — every real-order-ID action example routed correctly this
  run, so a case-normalization fix, while still cheap and worth doing
  defensively, would not move any measured number.
- The topical-scope-gate hypothesis is supported by **three of the five
  errors directly** (CANCEL×hard, both ADVERSARIAL×hard misses) and is
  distinguishable from a simple "raise the escalation floor" fix, which
  would instead make the SUBSCRIPTION cells *worse* by escalating more
  legitimate thin-corpus questions. A topical-scope check and a
  retrieval-relevance floor are different mechanisms solving different
  problems, and conflating them would trade one category of error for
  another rather than reducing errors overall.
**Runner-up, if corpus work is prioritized over code work:** enrich
`subscription.md` with real subscription-policy facts. Based on the
category-level pattern (6 of 15 SUBSCRIPTION examples wrong, spanning both
`easy` and `hard`), this single corpus change would likely recover most or
all of the SUBSCRIPTION errors — a bigger single-change win than any code
fix in this list, but it is a data-authoring task rather than an engineering
one, which is why the topical-scope gate is presented as the primary
recommendation.
 