// Anonymous session_id generation (ARCHITECTURE.md §8, §11) — a
// client-generated UUID persisted in localStorage. NOT authentication;
// nobody is verified. It exists purely so the agent can remember what it
// just asked within one conversation, and so a page refresh continues the
// same conversation instead of starting a new one.

const SESSION_STORAGE_KEY = "meridian_support_session_id";

export function getOrCreateSessionId(): string {
  const existing = window.localStorage.getItem(SESSION_STORAGE_KEY);
  if (existing) {
    return existing;
  }
  const sessionId = crypto.randomUUID();
  window.localStorage.setItem(SESSION_STORAGE_KEY, sessionId);
  return sessionId;
}
