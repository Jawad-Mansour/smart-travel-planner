import { apiFetch, readApiError } from "./http";

export type ChatSseEvent =
  | { type: "session"; session_id: string }
  | { type: "meta"; needs_clarification: boolean; missing_fields: string[]; intent: unknown }
  | { type: "delta"; content: string }
  | { type: "segment"; content: string }
  | {
      type: "done";
      message_id: string;
      session_id: string;
      needs_clarification: boolean;
      missing_fields: string[];
      elapsed_seconds?: number;
      intent?: unknown;
      tool_results?: Record<string, unknown>;
      usage_parts?: unknown[];
      webhook_status?: string;
      query_embedding_preview?: number[];
    }
  | { type: "error"; detail: string };

function parseSseBlock(block: string): ChatSseEvent | null {
  const line = block.split("\n").find((l) => l.startsWith("data:"));
  if (!line) return null;
  const json = line.slice(5).trim();
  if (!json) return null;
  try {
    return JSON.parse(json) as ChatSseEvent;
  } catch {
    return null;
  }
}

export async function streamChat(
  message: string,
  sessionId: string | null,
  contextPatch: Record<string, unknown> | null,
  onEvent: (ev: ChatSseEvent) => void,
  opts?: { signal?: AbortSignal }
): Promise<void> {
  const res = await apiFetch("/api/chat/stream", {
    method: "POST",
    body: JSON.stringify({
      message,
      session_id: sessionId,
      context_patch: contextPatch,
    }),
    signal: opts?.signal,
  });
  if (res.status === 401) {
    throw new Error("UNAUTHORIZED");
  }
  if (!res.ok) {
    throw new Error(await readApiError(res));
  }
  if (!res.body) {
    throw new Error("Chat request failed");
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const parts = buf.split("\n\n");
      buf = parts.pop() || "";
      for (const p of parts) {
        const ev = parseSseBlock(p);
        if (ev) onEvent(ev);
      }
    }
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") {
      throw new Error("ABORTED");
    }
    const m = e instanceof Error ? e.message : String(e);
    throw new Error(m || "Network error while streaming");
  }
  if (buf.trim()) {
    const ev = parseSseBlock(buf);
    if (ev) onEvent(ev);
  }
}
