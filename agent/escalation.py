"""Real escalation handling (ARCHITECTURE.md §8's ESCALATE steps 1-4).

Storage: SQLite, not JSONL (see ANALYSIS.md's M5 entry for the full
reasoning) — the packet is written BEFORE the Slack attempt (step 3's
guaranteed-first ordering), then its `slack_delivered` column is updated
after, and `GET /escalations?slack_delivered=false` needs a real WHERE
filter. Both are awkward with an append-only file, trivial with SQLite.

M9 revision — `handle()` is replaced by two methods, since M5's
single-call design couldn't actually ask the customer for contact info
before Slack fired (the ask lived in agent.py's response text, constructed
*after* this module had already inserted the record and attempted Slack):
- `check_known_contact()`: a passive check agent.py calls BEFORE deciding
  whether a real contact-collection round-trip is needed. Only trusts
  `email`/`phone` — an M9 bug fix: the old check also trusted a bare
  `order_id` slot, which let a hallucinated-then-grounded gibberish
  substring (e.g. "asdf1234") get recorded as real contact info, since an
  order_id can survive the slot-grounding check by literal accident
  without being a genuine identifier at all.
- `finalize()`: the terminal step (build, store, attempt Slack) — called
  either immediately (contact already known) or after agent.py's new
  `awaiting_contact` round-trip completes, successfully or exhausted.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import re
import sqlite3
import uuid

import requests

from app.config import ESCALATION_STORE_PATH, SLACK_WEBHOOK_URL
from app.schemas import EscalationPacket

logger = logging.getLogger("agent.escalation")

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"\+?\d[\d\-\s()]{6,14}\d")

_SLACK_TIMEOUT_SECONDS = 5
_SLACK_QUERY_DISPLAY_MAX_CHARS = 200


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _truncate_for_display(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


class EscalationHandler:
    def __init__(self, store_path: str = ESCALATION_STORE_PATH, slack_webhook_url: str = SLACK_WEBHOOK_URL):
        self._store_path = store_path
        self._slack_webhook_url = slack_webhook_url
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        directory = os.path.dirname(self._store_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        return sqlite3.connect(self._store_path)

    def _init_db(self) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS escalations (
                    escalation_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    contact TEXT,
                    contact_captured INTEGER NOT NULL,
                    slack_delivered INTEGER NOT NULL,
                    query TEXT NOT NULL,
                    retrieved_context TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
                """
            )
            # M9: additive column on an existing local store — a plain
            # CREATE TABLE would leave old rows without it; ALTER TABLE
            # with a default backfills them transparently.
            existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(escalations)").fetchall()}
            if "attempted_summary" not in existing_columns:
                conn.execute("ALTER TABLE escalations ADD COLUMN attempted_summary TEXT NOT NULL DEFAULT ''")
            conn.commit()
        finally:
            conn.close()

    def check_known_contact(self, slots: dict, query: str) -> str:
        """Passive check — does NOT trigger a round trip. Only `email`/
        `phone` count (M9 fix — a bare `order_id` slot is not trustworthy
        contact info; see this module's docstring). Returns None if
        nothing usable is already known."""
        for key in ("email", "phone"):
            value = slots.get(key)
            if value:
                return value
        match = _EMAIL_RE.search(query) or _PHONE_RE.search(query)
        return match.group(0) if match else None

    def finalize(
        self,
        query: str,
        session_id: str,
        reason: str,
        retrieved_context: list,
        slots: dict,
        attempted_summary: str,
        explicit_contact: str = None,
    ) -> EscalationPacket:
        """§8 steps 1-4, terminal step. `explicit_contact` — when given —
        is trusted as-is (agent.py's `awaiting_contact` round-trip already
        confirmed it's a direct answer to "please share your email or
        order number," so an order_id IS acceptable there, unlike the
        passive `check_known_contact()` path). Otherwise falls back to the
        passive check."""
        contact = explicit_contact or self.check_known_contact(slots, query)

        packet = EscalationPacket(
            escalation_id=str(uuid.uuid4()),
            session_id=session_id,
            contact=contact,
            contact_captured=bool(contact),
            slack_delivered=False,
            query=query,
            retrieved_context=retrieved_context,
            reason=reason,
            timestamp=_utc_now_iso(),
            attempted_summary=attempted_summary,
        )

        self._insert(packet)  # ALWAYS FIRST, guaranteed, before Slack (step 3)

        slack_delivered = self._try_post_to_slack(packet)
        if slack_delivered:
            self._update_slack_delivered(packet.escalation_id, True)
            packet = packet.model_copy(update={"slack_delivered": True})

        return packet

    def list_escalations(self, slack_delivered: bool = None) -> list:
        conn = self._connect()
        try:
            if slack_delivered is None:
                rows = conn.execute("SELECT * FROM escalations ORDER BY timestamp DESC").fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM escalations WHERE slack_delivered = ? ORDER BY timestamp DESC",
                    (int(slack_delivered),),
                ).fetchall()
        finally:
            conn.close()
        return [self._row_to_packet(row) for row in rows]

    # ------------------------------------------------------------------

    def _try_post_to_slack(self, packet: EscalationPacket) -> bool:
        if not self._slack_webhook_url:
            logger.info("SLACK_WEBHOOK_URL not configured — skipping notification for escalation %s.", packet.escalation_id)
            return False
        try:
            response = requests.post(
                self._slack_webhook_url,
                json=self._format_slack_message(packet),
                timeout=_SLACK_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            return True
        except requests.RequestException as e:
            # A Slack outage must never break the customer-facing response —
            # the packet is already durably stored by the time we get here.
            logger.warning("Slack notification failed for escalation %s: %s", packet.escalation_id, e)
            return False

    def _format_slack_message(self, packet: EscalationPacket) -> dict:
        # M9/M10: Block Kit — a bold header line, a divider, then each
        # field as its own section (not a wall of concatenated text).
        # `text` is kept as a plain-text fallback for notifications/screen
        # readers that can't render blocks, per Slack's own guidance.
        contact_text = packet.contact if packet.contact else "Not captured"
        summary_text = packet.attempted_summary or packet.reason
        # Display-only truncation — the stored record (`packet.query`,
        # what's returned by list_escalations()/GET /escalations) is never
        # touched; this only shortens what's rendered in the Slack message
        # itself, so a very long customer message doesn't turn into a wall
        # of text there.
        query_display = _truncate_for_display(packet.query, _SLACK_QUERY_DISPLAY_MAX_CHARS)
        return {
            "text": f"Escalation {packet.escalation_id}: {query_display}",
            "blocks": [
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": f"🚨 Escalation {packet.escalation_id}", "emoji": True},
                },
                {"type": "divider"},
                {"type": "section", "text": {"type": "mrkdwn", "text": f"*Customer Query:*\n{query_display}"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": f"*What We Tried:*\n{summary_text}"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": f"*Contact:*\n{contact_text}"}},
            ],
        }

    def _insert(self, packet: EscalationPacket) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO escalations
                    (escalation_id, session_id, contact, contact_captured, slack_delivered,
                     query, retrieved_context, reason, timestamp, attempted_summary)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    packet.escalation_id,
                    packet.session_id,
                    packet.contact,
                    int(packet.contact_captured),
                    int(packet.slack_delivered),
                    packet.query,
                    json.dumps(packet.retrieved_context),
                    packet.reason,
                    packet.timestamp,
                    packet.attempted_summary,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def _update_slack_delivered(self, escalation_id: str, slack_delivered: bool) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE escalations SET slack_delivered = ? WHERE escalation_id = ?",
                (int(slack_delivered), escalation_id),
            )
            conn.commit()
        finally:
            conn.close()

    def _row_to_packet(self, row) -> EscalationPacket:
        (
            escalation_id, session_id, contact, contact_captured, slack_delivered,
            query, retrieved_context_json, reason, timestamp, attempted_summary,
        ) = row
        retrieved_context = json.loads(retrieved_context_json) if retrieved_context_json else []
        return EscalationPacket(
            escalation_id=escalation_id,
            session_id=session_id,
            contact=contact,
            contact_captured=bool(contact_captured),
            slack_delivered=bool(slack_delivered),
            query=query,
            retrieved_context=retrieved_context,
            reason=reason,
            timestamp=timestamp,
            attempted_summary=attempted_summary or "",
        )
