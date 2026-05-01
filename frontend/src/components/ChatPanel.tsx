import {
  CalendarClock,
  Landmark,
  Mountain,
  Paperclip,
  Plane,
  Share2,
  Square,
  SunMedium,
  Trash2,
  Users,
  Wallet,
  type LucideIcon,
} from "lucide-react";
import {
  type Dispatch,
  type KeyboardEvent,
  type ReactNode,
  type SetStateAction,
  useCallback,
  useEffect,
  useRef,
  useState,
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
/** Composer grows from a single-line feel up to this cap (px). */
const COMPOSER_MAX_HEIGHT = 180;

const TRIP_STARTERS: { Icon: LucideIcon; label: string; prompt: string }[] = [
  {
    Icon: Mountain,
    label: "Adventure trip",
    prompt: "Plan a 10-day adventure trip with hiking and outdoor activities.",
  },
  {
    Icon: SunMedium,
    label: "Beach & relaxation",
    prompt: "I want a relaxing beach getaway with warm weather for about a week.",
  },
  {
    Icon: Landmark,
    label: "Culture & history",
    prompt: "Suggest destinations rich in museums, history, and food for 8–12 days.",
  },
  {
    Icon: Wallet,
    label: "Budget travel",
    prompt: "Budget-conscious trip under $1500 total with affordable destinations.",
  },
  {
    Icon: Users,
    label: "Family vacation",
    prompt: "Family-friendly trip with kids for summer, mix of activities and downtime.",
  },
  {
    Icon: CalendarClock,
    label: "Last minute trip",
    prompt: "Last-minute flexible trip in the next few weeks — surprise me with options.",
  },
];

function TripStarterPills({
  onSelect,
  disabled,
  variant = "hero",
}: {
  onSelect: (prompt: string) => void;
  disabled: boolean;
  /** hero: centered landing / compact: slim row above composer */
  variant?: "hero" | "compact";
}) {
  const isCompact = variant === "compact";
  return (
    <div className="flex flex-wrap items-center justify-center gap-2">
      {TRIP_STARTERS.map(({ Icon, label, prompt }) => (
        <button
          key={label}
          type="button"
          disabled={disabled}
          className={`inline-flex max-w-[calc(100vw-2rem)] items-center gap-2 rounded-full border border-slate-200 bg-white font-normal text-slate-800 shadow-sm transition hover:border-slate-300 hover:bg-slate-50/95 active:scale-[0.99] disabled:opacity-40 ${
            isCompact ? "px-2.5 py-1.5 text-xs" : "px-3.5 py-2 text-sm"
          }`}
          onClick={() => onSelect(prompt)}
        >
          <Icon
            className={`shrink-0 text-slate-500 ${isCompact ? "h-3 w-3" : "h-3.5 w-3.5"}`}
            strokeWidth={1.75}
            aria-hidden
          />
          <span className="truncate sm:whitespace-nowrap">{label}</span>
        </button>
      ))}
    </div>
  );
}

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
  messages,
  setMessages,
  chatTitle,
  onSessionResolved,
  onStreamAnalysis,
  mobileHeader,
  onClearThread,
}: {
  sessionId: string | null;
  messages: ChatMessage[];
  setMessages: Dispatch<SetStateAction<ChatMessage[]>>;
  chatTitle: string;
  onSessionResolved: (id: string) => void;
  onStreamAnalysis?: (payload: StreamAnalysis) => void;
  mobileHeader?: ReactNode;
  onClearThread?: () => void;
}) {
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [streaming, setStreaming] = useState("");
  const [segmentMode, setSegmentMode] = useState(false);
  const [timingBanner, setTimingBanner] = useState<string | null>(null);
  const [clarifyBlock, setClarifyBlock] = useState<{ fields: string[]; userText: string } | null>(null);
  const [sendPulse, setSendPulse] = useState(false);
  /** After a full assistant reply, hide quick chips until the user asks to show them again. */
  const [showQuickChips, setShowQuickChips] = useState(true);
  const bottomRef = useRef<HTMLDivElement>(null);
  const taRef = useRef<HTMLTextAreaElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  const displayMessages = expandForDisplay(messages);

  useEffect(() => {
    setShowQuickChips(true);
  }, [sessionId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streaming, loading, segmentMode, clarifyBlock]);

  useEffect(() => {
    const el = taRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, COMPOSER_MAX_HEIGHT)}px`;
  }, [input]);

  const stopStream = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
  }, []);

  const applyStarterPrompt = useCallback((prompt: string) => {
    setInput(prompt);
    requestAnimationFrame(() => {
      const el = taRef.current;
      if (!el) return;
      el.focus();
      const len = prompt.length;
      el.setSelectionRange(len, len);
    });
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

            if (segmentCount > 0 || assistantJoined.trim()) {
              setShowQuickChips(false);
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

  const showBetweenSegmentSearch =
    loading && segmentMode && !streaming && !messages.some((m) => m.id === PLACEHOLDER_ID);

  const isEmptyThread = displayMessages.length === 0 && !streaming && !loading;

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
          {isEmptyThread ? (
            <div className="flex flex-col items-center justify-center px-4 pb-8 pt-10 text-center sm:pt-14">
              <div className="mb-5 flex h-14 w-14 items-center justify-center overflow-hidden rounded-2xl bg-gradient-to-br from-emerald-500/12 to-teal-500/18 ring-1 ring-emerald-200/50">
                <img src="/plane.svg" alt="" className="h-11 w-11 object-contain" />
              </div>
              <h2 className="text-xl font-semibold tracking-tight text-slate-900 sm:text-2xl">
                Where should we plan <span className="text-emerald-700">next</span>?
              </h2>
              <p className="mt-2 max-w-md text-sm leading-relaxed text-slate-600">
                Add budget, dates, and interests below — or tap a starter to fill the box.
              </p>
              {showQuickChips ? (
                <div className="mt-8 w-full max-w-lg">
                  <TripStarterPills variant="hero" disabled={loading} onSelect={applyStarterPrompt} />
                  <button
                    type="button"
                    className="mt-4 text-xs font-medium text-slate-400 transition hover:text-slate-600"
                    onClick={() => setShowQuickChips(false)}
                  >
                    Hide suggestions
                  </button>
                </div>
              ) : (
                <button
                  type="button"
                  className="mt-6 text-sm font-medium text-emerald-800 underline-offset-2 hover:underline"
                  onClick={() => setShowQuickChips(true)}
                >
                  Show trip ideas
                </button>
              )}
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

          {showBetweenSegmentSearch ? (
            <div className="msg-enter flex justify-start pl-1">
              <div className="flex max-w-[min(100%,36rem)] items-center gap-3 rounded-2xl border border-emerald-100/90 bg-gradient-to-r from-white to-emerald-50/40 px-4 py-3 shadow-sm">
                <LoadingEarth compact />
                <div className="min-w-0">
                  <p className="text-xs font-semibold text-emerald-900">Searching destinations…</p>
                  <p className="text-[11px] text-slate-500">Preparing the next part of your plan</p>
                  <div className="-ml-1">
                    <TypingIndicator />
                  </div>
                </div>
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

      <div className="shrink-0 border-t border-slate-200/90 bg-white/95 px-3 py-2 backdrop-blur-md sm:px-6">
        <div className="mx-auto max-w-3xl space-y-2">
          {showQuickChips && !isEmptyThread ? (
            <div className="space-y-1">
              <div className="flex items-center justify-end px-0.5">
                <button
                  type="button"
                  className="text-[10px] font-medium text-slate-400 transition hover:text-slate-600"
                  onClick={() => setShowQuickChips(false)}
                >
                  Hide ideas
                </button>
              </div>
              <TripStarterPills variant="compact" disabled={loading} onSelect={applyStarterPrompt} />
            </div>
          ) : !showQuickChips ? (
            <div className="flex justify-center py-0.5">
              <button
                type="button"
                disabled={loading}
                className="text-[11px] font-medium text-slate-500 underline-offset-2 hover:text-emerald-800 hover:underline disabled:opacity-40"
                onClick={() => setShowQuickChips(true)}
              >
                Show trip ideas
              </button>
            </div>
          ) : null}

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
              style={{ maxHeight: COMPOSER_MAX_HEIGHT }}
              className="min-h-[40px] w-full resize-none rounded-[22px] border-0 bg-transparent px-4 py-2.5 text-sm leading-snug text-slate-900 outline-none placeholder:text-slate-400"
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
                    <Plane
                      className={`h-5 w-5 -rotate-45 transition-transform ${sendPulse ? "-translate-x-0.5 translate-y-0.5" : ""}`}
                      strokeWidth={2}
                    />
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
