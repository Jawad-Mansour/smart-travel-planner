import { Paperclip, Plane, Share2, Square, Trash2 } from "lucide-react";
import {
  type KeyboardEvent,
  useCallback,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { streamChat, type ChatSseEvent } from "../api/chat";
import type { ChatMessage } from "../api/sessions";
import type { StreamAnalysis } from "../lib/analysisFormatter";
import { splitTravelMarkdown } from "../lib/splitTravelMarkdown";
import { InlineClarification } from "./InlineClarification";
import { LoadingEarth } from "./LoadingEarth";
import { MarkdownMessage } from "./MarkdownMessage";
import { TypingIndicator } from "./TypingIndicator";

const PLACEHOLDER_ID = "__stp_loading__";
const MAX_INPUT = 2000;

const QUICK_CHIPS: { icon: string; text: string; prompt: string }[] = [
  { icon: "🏔️", text: "Adventure trip", prompt: "Plan a 10-day adventure trip with hiking and outdoor activities." },
  { icon: "🏖️", text: "Beach & relaxation", prompt: "I want a relaxing beach getaway with warm weather for about a week." },
  { icon: "🏛️", text: "Culture & history", prompt: "Suggest destinations rich in museums, history, and food for 8–12 days." },
  { icon: "💰", text: "Budget travel", prompt: "Budget-conscious trip under $1500 total with affordable destinations." },
  { icon: "👨‍👩‍👧‍👦", text: "Family vacation", prompt: "Family-friendly trip with kids for summer, mix of activities and downtime." },
  { icon: "🗓️", text: "Last minute trip", prompt: "Last-minute flexible trip in the next few weeks — surprise me with options." },
];

function expandForDisplay(messages: ChatMessage[]): ChatMessage[] {
  const out: ChatMessage[] = [];
  for (const m of messages) {
    if (m.role !== "assistant") {
      out.push(m);
      continue;
    }
    const parts = splitTravelMarkdown(m.content);
    if (!parts) {
      out.push(m);
      continue;
    }
    parts.forEach((content, idx) => {
      out.push({
        ...m,
        id: `${m.id}~${idx}`,
        content,
        meta: { ...m.meta, ui_shard: idx },
      });
    });
  }
  return out;
}

function counterClass(len: number): string {
  if (len > MAX_INPUT - 80) return "text-red-600 font-semibold";
  if (len > MAX_INPUT * 0.75) return "text-amber-600 font-medium";
  return "text-slate-500";
}

export function ChatPanel({
  sessionId,
  initialMessages,
  chatTitle,
  onSessionResolved,
  onStreamAnalysis,
  mobileHeader,
  onClearThread,
}: {
  sessionId: string | null;
  initialMessages: ChatMessage[];
  chatTitle: string;
  onSessionResolved: (id: string) => void;
  onStreamAnalysis?: (payload: StreamAnalysis) => void;
  mobileHeader?: ReactNode;
  onClearThread?: () => void;
}) {
  const [messages, setMessages] = useState<ChatMessage[]>(initialMessages);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [streaming, setStreaming] = useState("");
  const [segmentMode, setSegmentMode] = useState(false);
  const [timingBanner, setTimingBanner] = useState<string | null>(null);
  const [clarifyBlock, setClarifyBlock] = useState<{ fields: string[]; userText: string } | null>(null);
  const [sendPulse, setSendPulse] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const taRef = useRef<HTMLTextAreaElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  const displayMessages = expandForDisplay(messages);

  useEffect(() => {
    setMessages(initialMessages);
  }, [sessionId, initialMessages]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streaming, loading, segmentMode, clarifyBlock]);

  useEffect(() => {
    const el = taRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 132)}px`;
  }, [input]);

  const stopStream = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
  }, []);

  const runStream = async (text: string, patch: Record<string, unknown> | null) => {
    const trimmed = text.trim();
    if (!trimmed || loading) return;
    setLoading(true);
    setStreaming("");
    setSegmentMode(false);
    setClarifyBlock(null);
    setTimingBanner(null);
    abortRef.current?.abort();
    abortRef.current = new AbortController();

    const userMsg: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: trimmed,
      created_at: new Date().toISOString(),
    };
    const placeholder: ChatMessage = {
      id: PLACEHOLDER_ID,
      role: "assistant",
      content: "_Analyzing your preferences…_",
      created_at: new Date().toISOString(),
      meta: { shimmer: true },
    };

    if (!patch) {
      setMessages((m) => [...m, userMsg, placeholder]);
    } else {
      setMessages((m) => [...m.filter((x) => x.id !== PLACEHOLDER_ID), placeholder]);
    }

    setInput("");
    let resolvedSession = sessionId;
    let assistantJoined = "";
    let segmentCount = 0;

    try {
      await streamChat(
        trimmed,
        resolvedSession,
        patch,
        (ev: ChatSseEvent) => {
          if (ev.type === "session") {
            resolvedSession = ev.session_id;
            onSessionResolved(ev.session_id);
          } else if (ev.type === "delta") {
            assistantJoined += ev.content;
            setMessages((m) => m.filter((x) => x.id !== PLACEHOLDER_ID));
            setStreaming(assistantJoined);
          } else if (ev.type === "segment") {
            segmentCount += 1;
            setSegmentMode(true);
            const mid = crypto.randomUUID();
            setMessages((m) => [
              ...m.filter((x) => x.id !== PLACEHOLDER_ID),
              {
                id: mid,
                role: "assistant",
                content: ev.content,
                created_at: new Date().toISOString(),
              },
            ]);
            setStreaming("");
          } else if (ev.type === "done") {
            setSegmentMode(false);
            const tools =
              ev.tool_results && typeof ev.tool_results === "object" ? ev.tool_results : {};
            const intentRaw = ev.intent;
            onStreamAnalysis?.({
              elapsed_seconds: ev.elapsed_seconds,
              intent: intentRaw && typeof intentRaw === "object" ? (intentRaw as Record<string, unknown>) : null,
              tool_results: tools,
              webhook_status: ev.webhook_status,
              usage_parts: ev.usage_parts as unknown[] | undefined,
              query_embedding_preview: ev.query_embedding_preview,
            });

            if (!segmentCount && assistantJoined) {
              const aid = crypto.randomUUID();
              setMessages((m) => [
                ...m.filter((x) => x.id !== PLACEHOLDER_ID),
                {
                  id: aid,
                  role: "assistant",
                  content: assistantJoined,
                  created_at: new Date().toISOString(),
                },
              ]);
              setStreaming("");
            }

            if (ev.elapsed_seconds != null) {
              setTimingBanner(`✨ Response generated in ${ev.elapsed_seconds}s`);
            }

            if (ev.needs_clarification && ev.missing_fields?.length) {
              setClarifyBlock({ fields: ev.missing_fields, userText: trimmed });
            }
          } else if (ev.type === "error") {
            throw new Error(ev.detail);
          }
        },
        { signal: abortRef.current.signal }
      );

      setMessages((m) => m.filter((x) => x.id !== PLACEHOLDER_ID));
    } catch (e) {
      const msg = (e as Error).message;
      if (msg === "UNAUTHORIZED") {
        window.location.href = "/login";
        return;
      }
      if (msg === "ABORTED") {
        setMessages((m) => [
          ...m.filter((x) => x.id !== PLACEHOLDER_ID),
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content: "_Generation stopped._",
            created_at: new Date().toISOString(),
          },
        ]);
        setStreaming("");
        return;
      }
      setMessages((m) => [
        ...m.filter((x) => x.id !== PLACEHOLDER_ID),
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: `**Something went wrong**\n\n${msg}`,
          created_at: new Date().toISOString(),
        },
      ]);
      setStreaming("");
    } finally {
      setLoading(false);
      abortRef.current = null;
    }
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void runStream(input, null);
    }
  };

  const handleShare = async () => {
    const text = displayMessages.map((m) => `**${m.role}**\n${m.content}`).join("\n\n---\n\n");
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      /* ignore */
    }
  };

  const showTailTyping =
    loading && segmentMode && !streaming && !messages.some((m) => m.id === PLACEHOLDER_ID);

  return (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col bg-[#f9fafb]">
      <header className="flex shrink-0 items-center gap-3 border-b border-slate-200/90 bg-white px-3 py-3 shadow-sm sm:px-5">
        {mobileHeader}
        <div className="min-w-0 flex-1">
          <h1 className="truncate text-base font-semibold tracking-tight text-slate-900 sm:text-lg">
            {chatTitle || "Trip planning"}
          </h1>
          <p className="truncate text-xs text-slate-500">Markdown replies · live tools behind the scenes</p>
        </div>
        <div className="flex shrink-0 items-center gap-1.5">
          <button
            type="button"
            className="hidden rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs font-medium text-slate-700 shadow-sm transition hover:bg-slate-50 sm:inline-flex sm:items-center sm:gap-1.5"
            onClick={() => void handleShare()}
          >
            <Share2 className="h-3.5 w-3.5" />
            Share
          </button>
          <button
            type="button"
            className="inline-flex items-center gap-1.5 rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs font-medium text-slate-700 shadow-sm transition hover:bg-red-50 hover:text-red-800 hover:border-red-200"
            onClick={() => {
              if (!onClearThread) return;
              if (window.confirm("Start fresh in a new chat? Your current session stays in the sidebar.")) {
                onClearThread();
              }
            }}
          >
            <Trash2 className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Clear</span>
          </button>
        </div>
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto px-3 py-4 sm:px-6">
        <div className="mx-auto max-w-3xl space-y-4">
          {displayMessages.length === 0 && !streaming && !loading ? (
            <div className="flex flex-col items-center justify-center px-4 py-16 text-center">
              <div className="mb-6 flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-emerald-500/15 to-teal-500/20 ring-1 ring-emerald-200/60">
                <Plane className="h-8 w-8 text-emerald-700" />
              </div>
              <h2 className="text-xl font-semibold text-slate-900 sm:text-2xl">
                Where should we plan next?
              </h2>
              <p className="mt-2 max-w-md text-sm text-slate-600">
                Describe budget, timing, and vibe — the agent blends retrieval, ML routing, and live pricing hints.
              </p>
            </div>
          ) : null}

          {displayMessages.map((msg) => (
            <div key={msg.id} className="msg-enter">
              <div className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                <div
                  className={`max-w-[min(100%,36rem)] rounded-2xl px-4 py-3 text-sm shadow-sm transition ${
                    msg.role === "user"
                      ? "rounded-br-md bg-[#0a7c7c] text-white shadow-emerald-900/10"
                      : msg.meta?.shimmer
                        ? "border border-slate-100 bg-white text-slate-600 italic"
                        : "border border-slate-100/90 bg-white text-slate-800 shadow-md"
                  }`}
                >
                  {msg.role === "assistant" ? (
                    <MarkdownMessage content={msg.content} />
                  ) : (
                    <p className="whitespace-pre-wrap">{msg.content}</p>
                  )}
                </div>
              </div>
            </div>
          ))}

          {streaming ? (
            <div className="msg-enter flex justify-start">
              <div className="max-w-[min(100%,36rem)] rounded-2xl border border-slate-100 bg-white px-4 py-3 text-sm shadow-md">
                <MarkdownMessage content={streaming} />
              </div>
            </div>
          ) : null}

          {clarifyBlock && !loading ? (
            <div className="msg-enter">
              <InlineClarification
                fields={clarifyBlock.fields}
                disabled={loading}
                onSubmit={(patch) => {
                  const t = clarifyBlock.userText;
                  setClarifyBlock(null);
                  void runStream(t, patch);
                }}
              />
            </div>
          ) : null}

          {loading && !streaming && !segmentMode && messages.some((m) => m.id === PLACEHOLDER_ID) ? (
            <div className="flex justify-start">
              <div className="flex items-center gap-3 rounded-2xl border border-slate-100 bg-white px-4 py-2 shadow-sm">
                <LoadingEarth compact />
                <TypingIndicator />
              </div>
            </div>
          ) : null}

          {showTailTyping ? (
            <div className="flex justify-start pl-1">
              <div className="rounded-2xl border border-slate-100 bg-white px-4 py-2 shadow-sm">
                <TypingIndicator />
              </div>
            </div>
          ) : null}

          {timingBanner && !loading ? (
            <div className="flex justify-center py-1">
              <span className="rounded-full bg-emerald-50 px-3 py-1 text-[11px] font-medium text-emerald-800 ring-1 ring-emerald-100">
                {timingBanner}
              </span>
            </div>
          ) : null}

          <div ref={bottomRef} />
        </div>
      </div>

      <div className="shrink-0 border-t border-slate-200/90 bg-white/95 px-3 py-3 backdrop-blur-md sm:px-6">
        <div className="mx-auto max-w-3xl space-y-2">
          <div className="flex flex-wrap justify-center gap-2 pb-1">
            {QUICK_CHIPS.map((c) => (
              <button
                key={c.text}
                type="button"
                disabled={loading}
                className="inline-flex items-center gap-1 rounded-full border border-slate-200 bg-[#f9fafb] px-3 py-1.5 text-xs font-medium text-slate-700 transition hover:border-emerald-300 hover:bg-emerald-50 disabled:opacity-40"
                onClick={() => setInput(c.prompt)}
              >
                <span>{c.icon}</span>
                {c.text}
              </button>
            ))}
          </div>

          <form
            className="relative rounded-[28px] border border-slate-200/90 bg-[#f9fafb] p-1.5 shadow-inner ring-emerald-500/0 transition focus-within:border-emerald-300/80 focus-within:ring-2 focus-within:ring-emerald-200/60"
            onSubmit={(e) => {
              e.preventDefault();
              setSendPulse(true);
              window.setTimeout(() => setSendPulse(false), 420);
              void runStream(input, null);
            }}
          >
            <textarea
              ref={taRef}
              className="max-h-[132px] min-h-[48px] w-full resize-none rounded-[22px] border-0 bg-transparent px-4 py-3 text-sm text-slate-900 outline-none placeholder:text-slate-400"
              placeholder="Describe your trip — budget, dates, interests, anything..."
              value={input}
              maxLength={MAX_INPUT}
              rows={1}
              disabled={loading}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKeyDown}
            />
            <div className="flex items-center justify-between gap-2 px-2 pb-1 pt-0">
              <button
                type="button"
                disabled
                title="Attachments coming soon"
                className="flex h-11 w-11 items-center justify-center rounded-full text-slate-400 opacity-50"
              >
                <Paperclip className="h-5 w-5" />
              </button>
              <div className="flex items-center gap-2">
                {loading ? (
                  <button
                    type="button"
                    onClick={() => stopStream()}
                    className="flex h-11 items-center gap-2 rounded-full bg-red-600 px-4 text-xs font-semibold text-white shadow-sm transition hover:bg-red-700"
                  >
                    <Square className="h-3.5 w-3.5 fill-current" />
                    Stop
                  </button>
                ) : (
                  <button
                    type="submit"
                    disabled={!input.trim()}
                    className={`flex h-11 w-11 items-center justify-center rounded-full bg-emerald-600 text-white shadow-md transition hover:bg-emerald-700 disabled:cursor-not-allowed disabled:bg-slate-300 ${sendPulse ? "scale-95" : "hover:scale-105"}`}
                    aria-label="Send"
                  >
                    <Plane className={`h-5 w-5 -rotate-45 transition-transform ${sendPulse ? "-translate-x-0.5 translate-y-0.5" : ""}`} />
                  </button>
                )}
              </div>
            </div>
          </form>
          <div className="flex items-center justify-between px-1 text-[11px] text-slate-500">
            <span>Enter to send · Shift+Enter for new line</span>
            <span className={counterClass(input.length)}>
              {input.length} / {MAX_INPUT}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
