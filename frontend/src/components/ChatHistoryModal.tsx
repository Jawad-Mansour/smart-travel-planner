import { Clock, Search, Trash2, X } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import type { Session } from "../api/sessions";

function timeAgo(iso: string | null): string {
  if (!iso) return "";
  const t = new Date(iso).getTime();
  if (Number.isNaN(t)) return "";
  const s = Math.floor((Date.now() - t) / 1000);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

export function ChatHistoryModal({
  open,
  onClose,
  sessions,
  sessionsLoading = false,
  activeId,
  onPickSession,
  onDeleteSession,
}: {
  open: boolean;
  onClose: () => void;
  sessions: Session[];
  sessionsLoading?: boolean;
  activeId: string | null;
  onPickSession: (id: string) => void;
  onDeleteSession: (id: string) => void;
}) {
  const [q, setQ] = useState("");

  useEffect(() => {
    if (!open) setQ("");
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  const filtered = useMemo(() => {
    const needle = q.trim().toLowerCase();
    const list = [...sessions].sort((a, b) => {
      const ta = a.updated_at ? new Date(a.updated_at).getTime() : 0;
      const tb = b.updated_at ? new Date(b.updated_at).getTime() : 0;
      return tb - ta;
    });
    if (!needle) return list;
    return list.filter((s) => (s.title || "Trip").toLowerCase().includes(needle));
  }, [sessions, q]);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-[60] flex items-end justify-center sm:items-center sm:p-4" role="dialog" aria-modal="true" aria-labelledby="stp-history-title">
      <button type="button" className="absolute inset-0 bg-slate-900/45 backdrop-blur-[2px]" aria-label="Close history" onClick={onClose} />
      <div className="relative flex max-h-[min(85vh,640px)] w-full max-w-lg flex-col rounded-t-2xl border border-slate-200/90 bg-white shadow-2xl sm:rounded-2xl">
        <header className="flex shrink-0 items-center gap-3 border-b border-slate-100 px-4 py-3 sm:px-5">
          <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-emerald-500 to-teal-600 text-white shadow-md">
            <Clock className="h-5 w-5" />
          </span>
          <div className="min-w-0 flex-1">
            <h2 id="stp-history-title" className="text-base font-semibold text-slate-900">
              Your trips
            </h2>
            <p className="text-xs text-slate-500">Open any chat — newest first</p>
          </div>
          <button type="button" className="btn-icon-muted h-10 w-10 shrink-0" aria-label="Close" onClick={onClose}>
            <X className="h-5 w-5" />
          </button>
        </header>

        <div className="shrink-0 border-b border-slate-100 px-4 py-2 sm:px-5">
          <div className="relative">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
            <input
              type="search"
              className="w-full rounded-xl border border-slate-200 bg-slate-50 py-2.5 pl-10 pr-3 text-sm text-slate-900 outline-none ring-emerald-200/80 transition focus:bg-white focus:ring-2"
              placeholder="Search by title…"
              value={q}
              onChange={(e) => setQ(e.target.value)}
            />
          </div>
        </div>

        <ul className="min-h-0 flex-1 overflow-y-auto px-2 py-2 sm:px-3">
          {sessionsLoading && sessions.length === 0 ? (
            <li className="space-y-3 px-3 py-8">
              {[0, 1, 2, 3].map((i) => (
                <div key={i} className="animate-pulse rounded-xl border border-slate-100 bg-slate-50 px-4 py-3">
                  <div className="h-4 max-w-[14rem] rounded bg-slate-200" style={{ width: `${58 + (i % 3) * 12}%` }} />
                  <div className="mt-2 h-3 w-16 rounded bg-slate-100" />
                </div>
              ))}
            </li>
          ) : filtered.length === 0 ? (
            <li className="px-3 py-10 text-center text-sm text-slate-500">
              {sessions.length === 0 ? "No trips yet — start with New chat." : "No trips match your search."}
            </li>
          ) : (
            filtered.map((s) => {
              const active = s.id === activeId;
              return (
                <li key={s.id} className="group border-b border-slate-50 last:border-0">
                  <div className="flex items-stretch gap-1 rounded-xl py-1.5 pr-1 transition hover:bg-slate-50">
                    <button
                      type="button"
                      className={`min-w-0 flex-1 rounded-lg px-3 py-2.5 text-left transition ${
                        active ? "bg-emerald-50 ring-1 ring-emerald-100" : ""
                      }`}
                      onClick={() => {
                        onPickSession(s.id);
                        onClose();
                      }}
                    >
                      <div className={`truncate text-sm font-medium ${active ? "text-emerald-950" : "text-slate-900"}`}>
                        {s.title?.trim() || "Trip"}
                      </div>
                      {s.updated_at ? (
                        <div className="mt-0.5 text-[11px] text-slate-500">{timeAgo(s.updated_at)}</div>
                      ) : null}
                    </button>
                    <button
                      type="button"
                      className="flex w-10 shrink-0 items-center justify-center rounded-lg text-slate-400 opacity-70 transition hover:bg-red-50 hover:text-red-600 group-hover:opacity-100"
                      aria-label="Delete trip"
                      onClick={(e) => {
                        e.stopPropagation();
                        onDeleteSession(s.id);
                      }}
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </div>
                </li>
              );
            })
          )}
        </ul>
      </div>
    </div>
  );
}
