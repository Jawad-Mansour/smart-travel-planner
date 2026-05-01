import { apiFetch, readApiError } from "./http";

export type Session = { id: string; title: string; created_at: string | null; updated_at: string | null };

export async function listSessions(): Promise<Session[]> {
  const res = await apiFetch("/api/sessions");
  if (!res.ok) throw new Error(await readApiError(res));
  return res.json();
}

export async function createSession(title?: string): Promise<Session> {
  const res = await apiFetch("/api/sessions", {
    method: "POST",
    body: JSON.stringify({ title: title ?? null }),
  });
  if (!res.ok) throw new Error(await readApiError(res));
  return res.json();
}

export type ChatMessage = {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  created_at: string | null;
  meta?: Record<string, unknown> | null;
};

export async function listMessages(sessionId: string): Promise<ChatMessage[]> {
  const res = await apiFetch(`/api/sessions/${sessionId}/messages`);
  if (!res.ok) throw new Error(await readApiError(res));
  return res.json();
}

export async function deleteSession(sessionId: string): Promise<void> {
  const res = await apiFetch(`/api/sessions/${sessionId}`, { method: "DELETE" });
  if (!res.ok) throw new Error(await readApiError(res));
}
