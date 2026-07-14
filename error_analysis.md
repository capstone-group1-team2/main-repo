## Error Analysis

Errors were grouped across **two dimensions**: policy **category** (12 values — ACCOUNT, ADVERSARIAL, CANCEL, CONTACT, DELIVERY, FEEDBACK, INVOICE, ORDER, PAYMENT, REFUND, SHIPPING, SUBSCRIPTION) and **difficulty tier** (easy / medium / hard), pooled across all 3 evaluation seeds.

**24 of 28 populated category × difficulty cells were perfect (0% error).** All 5 routing errors are concentrated in exactly 4 cells:

- **CANCEL × hard** — 100% (1 example, misclassified — skipped escalation)
- **ADVERSARIAL × hard** — 50% (2 of 4 off-topic questions answered instead of escalated)
- **SUBSCRIPTION × easy / hard** — 50% each (thin corpus, correctly over-escalates)

### 5 documented failure cases

1. **"Cancel my order" (no ID) misclassified as a different intent** — skipped the escalation path entirely
2. **A fluent tax-question (off-topic) was answered instead of escalated**
3. **A fluent labor-dispute question was answered instead of escalated**
4. **A legitimate subscription question (easy) was over-escalated** — the corpus is too thin to ground it
5. **A legitimate subscription question (hard) was also over-escalated** — same cause

*Also tested and refuted: a predicted order-ID casing bug — all 7 real-order actions routed correctly, so this hypothesis did not hold.*

### Next-iteration idea

> If we had another week, we would add a topical out-of-scope gate before answering, because every non-corpus error was the agent answering something it should have refused, not a failure of retrieval or generation quality (recall@k 0.956, groundedness 1.00 were both already strong).
