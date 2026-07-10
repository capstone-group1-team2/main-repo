"use client";

import { FormEvent, useEffect, useRef, useState } from "react";
import { AgentResponse, ChatApiError, sendMessage } from "@/lib/api";
import styles from "./ChatWindow.module.css";

interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "error";
  text: string;
  response?: AgentResponse;
}

const ROUTE_LABEL: Record<AgentResponse["route"], string> = {
  information: "Information",
  action: "Action taken",
  escalate: "Escalated to a human",
};

function formatPct(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function newId(): string {
  return crypto.randomUUID();
}

interface QuickReply {
  label: string;
  query: string;
  variant?: "escalate";
}

const QUICK_REPLIES: QuickReply[] = [
  { label: "Where's my order?", query: "Where's my order?" },
  { label: "How do I cancel an order?", query: "How do I cancel an order?" },
  { label: "What's your return policy?", query: "What's your return policy?" },
  {
    label: "Ask something off-topic (demo)",
    query: "What's your company's stock ticker symbol and current share price?",
    variant: "escalate",
  },
];

export default function ChatWindow({ sessionId }: { sessionId: string }) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  async function submitQuery(query: string) {
    if (!query || isLoading) return;

    setMessages((prev) => [...prev, { id: newId(), role: "user", text: query }]);
    setInputValue("");
    setIsLoading(true);

    try {
      const response = await sendMessage(query, sessionId);
      setMessages((prev) => [
        ...prev,
        { id: newId(), role: "assistant", text: response.detail, response },
      ]);
    } catch (err) {
      const message = err instanceof ChatApiError ? err.message : "Something went wrong. Please try again.";
      setMessages((prev) => [...prev, { id: newId(), role: "error", text: message }]);
    } finally {
      setIsLoading(false);
    }
  }

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    submitQuery(inputValue.trim());
  }

  function handleQuickReply(query: string) {
    // Behaves exactly like the user typing and sending it themselves —
    // same submitQuery() path, no special-cased backend behavior.
    submitQuery(query);
  }

  return (
    <div className={styles.card}>
      <header className={styles.header}>
        <div className={styles.headerAvatar} aria-hidden="true">
          MR
        </div>
        <div>
          <h1 className={styles.headerTitle}>Meridian Retail Support</h1>
          <p className={styles.headerSubtitle}>Usually replies in a few seconds</p>
        </div>
      </header>

      <div className={styles.transcript} role="log" aria-live="polite" aria-label="Conversation">
        {messages.length === 0 && (
          <div className={styles.emptyState}>
            <p className={styles.emptyStateTitle}>How can we help?</p>
            <p className={styles.emptyStateSubtitle}>
              Ask about an order, shipping, refunds, your account, or anything else.
            </p>
            <div className={styles.quickReplies}>
              {QUICK_REPLIES.map((reply) => (
                <button
                  key={reply.label}
                  type="button"
                  className={`${styles.quickReplyChip} ${reply.variant === "escalate" ? styles.quickReplyChipEscalate : ""}`}
                  onClick={() => handleQuickReply(reply.query)}
                  disabled={isLoading}
                >
                  {reply.label}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((message) => (
          <MessageBubble key={message.id} message={message} />
        ))}

        {isLoading && (
          <div className={`${styles.bubbleRow} ${styles.bubbleRowAssistant}`}>
            <div className={`${styles.bubble} ${styles.bubbleAssistant} ${styles.typingBubble}`} aria-live="polite" aria-label="Assistant is typing">
              <span className={styles.typingDot} />
              <span className={styles.typingDot} />
              <span className={styles.typingDot} />
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      <form className={styles.inputBar} onSubmit={handleSubmit}>
        <input
          ref={inputRef}
          className={styles.input}
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          placeholder="Type your message…"
          aria-label="Type your message"
          disabled={isLoading}
          autoComplete="off"
        />
        <button
          type="submit"
          className={styles.sendButton}
          disabled={isLoading || !inputValue.trim()}
          aria-label="Send message"
        >
          Send
        </button>
      </form>
    </div>
  );
}

function MessageBubble({ message }: { message: ChatMessage }) {
  if (message.role === "user") {
    return (
      <div className={`${styles.bubbleRow} ${styles.bubbleRowUser}`}>
        <div className={`${styles.bubble} ${styles.bubbleUser}`}>{message.text}</div>
      </div>
    );
  }

  if (message.role === "error") {
    return (
      <div className={`${styles.bubbleRow} ${styles.bubbleRowAssistant}`}>
        <div className={`${styles.bubble} ${styles.bubbleError}`} role="alert">
          {message.text}
        </div>
      </div>
    );
  }

  const response = message.response;
  const isEscalation = response?.route === "escalate";

  return (
    <div className={`${styles.bubbleRow} ${styles.bubbleRowAssistant}`}>
      <div className={`${styles.bubble} ${styles.bubbleAssistant} ${isEscalation ? styles.bubbleEscalated : ""}`}>
        {isEscalation && (
          <div className={styles.escalationNotice}>
            <span aria-hidden="true">🔔</span> A team member has been notified and will follow up.
          </div>
        )}
        <p className={styles.bubbleText}>{message.text}</p>
        {response && (
          <div className={styles.badgeRow}>
            <span className={`${styles.badge} ${styles[`badge_${response.route}`]}`}>
              {ROUTE_LABEL[response.route]}
            </span>
            <span className={styles.badgeMuted}>Intent: {response.intent}</span>
            <span className={styles.badgeMuted}>Confidence: {formatPct(response.intent_confidence)}</span>
            <span className={styles.badgeMuted}>Retrieval: {formatPct(response.retrieval_confidence)}</span>
          </div>
        )}
      </div>
    </div>
  );
}
