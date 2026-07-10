// Typed client for the FastAPI backend (ARCHITECTURE.md §11). Pure client
// fetch — no Next.js API routes, no server-side data fetching.

export type AgentRoute = "information" | "action" | "escalate";

export interface AgentResponse {
  route: AgentRoute;
  intent: string;
  intent_confidence: number;
  retrieval_confidence: number;
  detail: string;
  escalation_id: string | null;
  session_id: string;
}

export class ChatApiError extends Error {}

function getApiUrl(): string {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL;
  if (!apiUrl) {
    throw new ChatApiError(
      "NEXT_PUBLIC_API_URL is not set. Copy frontend/.env.local.example to frontend/.env.local."
    );
  }
  return apiUrl;
}

export async function sendMessage(query: string, sessionId: string): Promise<AgentResponse> {
  const apiUrl = getApiUrl();
  let response: Response;

  try {
    response = await fetch(`${apiUrl}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, session_id: sessionId }),
    });
  } catch {
    throw new ChatApiError("Could not reach the support backend. Check your connection and try again.");
  }

  if (!response.ok) {
    throw new ChatApiError(`The support backend returned an error (status ${response.status}).`);
  }

  return (await response.json()) as AgentResponse;
}
