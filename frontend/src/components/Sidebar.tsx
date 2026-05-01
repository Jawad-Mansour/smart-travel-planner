import {
  ChevronRight,
  LayoutGrid,
  LogOut,
  Mail,
  MessageCirclePlus,
  PanelLeftClose,
  PanelLeftOpen,
  Radar,
  RefreshCw,
  Sparkles,
  Trash2,
  X,
} from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import type { Session } from "../api/sessions";
import type { User } from "../context/AuthContext";
import { apiFetch } from "../api/http";
import { AppLogo } from "./AppLogo";
import type { InfoView } from "./InfoWorkspace";

function timeAgo(iso: string | null): string {
  if (!iso) return "";
  const t = new Date(iso).getTime();
  if (Number.isNaN(t)) return "";
  const s = Math.floor((Date.now() - t) / 1000);
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

function title30(t: string): string {
  const s = t.trim() || "Trip";
  return s.length <= 30 ? s : `${s.slice(0, 27)}…`;
}

const NAV: { id: Exclude<InfoView, "chat">; label: string; icon: typeof Sparkles }[] = [
  { id: "system-info", label: "How it works & data", icon: Sparkles },
  { id: "analysis", label: "Response analysis", icon: Radar },
  { id: "webhook-setup", label: "Webhook setup", icon: Mail },
];

export function Sidebar({
  sessions,
  activeId,
  mainView,
  mobileOpen,
  onMobileOpenChange,
  onSelect,
  onNewChat,
  onLogoClick,
  onNavigate,
  onDeleteSession,
  user,
  logout,
}: {
  sessions: Session[];
  activeId: string | null;
  mainView: InfoView;
  mobileOpen: boolean;
  onMobileOpenChange: (open: boolean) => void;
  onSelect: (id: string) => void;
  onNewChat: () => void;
  onLogoClick: () => void;
  onNavigate: (view: InfoView) => void;
  onDeleteSession: (id: string) => void;
  user: User;
  logout: () => void;
}) {
  const [collapsed, setCollapsed] = useState(() => localStorage.getItem("stp_sidebar_collapsed") === "1");
  const [healthOk, setHealthOk] = useState<boolean | null>(null);
  const [healthMs, setHealthMs] = useState<number | null>(null);
  const [healthOpen, setHealthOpen] = useState(false);

  useEffect(() => {
    localStorage.setItem("stp_sidebar_collapsed", collapsed ? "1" : "0");
  }, [collapsed]);

  const pingHealth = useCallback(async () => {
    const t0 = performance.now();
    try {
      const res = await apiFetch("/health", { auth: false });
      setHealthOk(res.ok);
      setHealthMs(Math.round(performance.now() - t0));
    } catch {
      setHealthOk(false);
      setHealthMs(null);
    }
  }, []);

  useEffect(() => {
    void pingHealth();
    const id = window.setInterval(() => void pingHealth(), 60000);
    return () => window.clearInterval(id);
  }, [pingHealth]);

  const narrow = collapsed;

  return (
    <aside
      className={`fixed inset-y-0 left-0 z-50 flex h-full shrink-0 flex-col border-r border-slate-200/90 bg-white shadow-[4px_0_24px_rgba(15,23,42,0.04)] transition-[width,transform] duration-200 ease-out lg:static lg:z-0 ${
        mobileOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0"
      } ${narrow ? "w-[72px] lg:w-[72px]" : "w-[min(100vw,280px)] lg:w-[280px]"}`}
    >
      <div
        className={`flex items-start gap-2 border-b border-slate-100 ${narrow ? "justify-center px-1 py-3" : "px-4 pb-4 pt-12"}`}
      >
        <button
          type="button"
          className="btn-icon-muted shrink-0 lg:hidden"
          aria-label="Close menu"
          onClick={() => onMobileOpenChange(false)}
        >
          <X className="h-5 w-5" />
        </button>
        {!narrow ? (
          <div className="min-w-0 flex-1">
            <AppLogo onClick={onLogoClick} />
          </div>
        ) : (
          <button
            type="button"
            onClick={onLogoClick}
            className="flex h-10 w-10 items-center justify-center rounded-xl bg-slate-50 ring-1 ring-slate-200/80"
            title="New chat"
          >
            <img src="/plane.svg" alt="" className="h-6 w-6 object-contain" />
          </button>
        )}
        <button
          type="button"
          className="btn-icon-muted hidden lg:flex"
          title={narrow ? "Expand sidebar" : "Collapse sidebar"}
          onClick={() => setCollapsed(!collapsed)}
        >
          {narrow ? <PanelLeftOpen className="h-5 w-5" /> : <PanelLeftClose className="h-5 w-5" />}
        </button>
      </div>

      <div className={narrow ? "px-1.5 pt-2" : "px-3 pt-3"}>
        <button
          type="button"
          onClick={() => {
            onNewChat();
            onMobileOpenChange(false);
          }}
          className={`flex w-full items-center justify-center gap-2 rounded-2xl bg-gradient-to-r from-emerald-600 to-teal-600 py-3 text-sm font-semibold text-white shadow-md transition hover:brightness-105 active:scale-[0.99] ${narrow ? "px-0" : ""}`}
        >
          <MessageCirclePlus className="h-4 w-4 shrink-0" />
          {!narrow ? <span>New chat</span> : null}
        </button>
      </div>

      <div className={`my-3 h-px bg-slate-100 ${narrow ? "mx-2" : "mx-3"}`} />

      <nav className={`space-y-0.5 ${narrow ? "px-1.5" : "px-2"}`}>
        <button
          type="button"
          onClick={() => {
            onNavigate("chat");
            onMobileOpenChange(false);
          }}
          title="Chat & history"
          className={`flex w-full items-center gap-2 rounded-xl py-2.5 text-left text-sm font-medium transition ${
            narrow ? "justify-center px-0" : "px-3"
          } ${
            mainView === "chat"
              ? "bg-emerald-50 text-emerald-900 ring-1 ring-emerald-100"
              : "text-slate-700 hover:bg-slate-50"
          }`}
        >
          <LayoutGrid className="h-4 w-4 shrink-0 opacity-80" />
          {!narrow ? <span>Chat & history</span> : null}
        </button>
        {NAV.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            type="button"
            title={label}
            onClick={() => {
              onNavigate(id);
              onMobileOpenChange(false);
            }}
            className={`flex w-full items-center gap-2 rounded-xl py-2.5 text-left text-sm font-medium transition ${
              narrow ? "justify-center px-0" : "px-3"
            } ${
              mainView === id ? "bg-sky-50 text-sky-900 ring-1 ring-sky-100" : "text-slate-700 hover:bg-slate-50"
            }`}
          >
            <Icon className="h-4 w-4 shrink-0 opacity-75" />
            {!narrow ? <span>{label}</span> : null}
          </button>
        ))}
      </nav>

      <div className={`my-3 h-px bg-slate-100 ${narrow ? "mx-2" : "mx-3"}`} />

      <div className="min-h-0 flex-1 overflow-y-auto">
        {!narrow ? (
          <div className="px-3 pb-2 pt-1 text-[10px] font-semibold uppercase tracking-wider text-slate-400">
            Recent trips
          </div>
        ) : null}
        <ul className={`space-y-0.5 ${narrow ? "px-1" : "px-2"}`}>
          {sessions.map((s) => (
            <li key={s.id} className="group relative">
              <button
                type="button"
                onClick={() => {
                  onSelect(s.id);
                  onNavigate("chat");
                  onMobileOpenChange(false);
                }}
                title={s.title}
                className={`flex w-full flex-col rounded-xl py-2 text-left transition ${
                  narrow ? "items-center px-0" : "px-3 pr-10"
                } ${
                  s.id === activeId && mainView === "chat"
                    ? "bg-slate-50 font-medium text-slate-900 ring-1 ring-slate-200/80"
                    : "text-slate-700 hover:bg-slate-50/90"
                }`}
              >
                {!narrow ? (
                  <>
                    <span className="truncate text-sm">{title30(s.title)}</span>
                    {s.updated_at ? (
                      <span className="mt-0.5 flex items-center gap-2 text-[10px] text-slate-400">
                        <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-sky-400" />
                        {timeAgo(s.updated_at)}
                      </span>
                    ) : null}
                  </>
                ) : (
                  <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-slate-100 text-[10px] font-bold text-slate-600">
                    {(s.title || "?").slice(0, 1).toUpperCase()}
                  </span>
                )}
              </button>
              {!narrow ? (
                <button
                  type="button"
                  className="absolute right-2 top-1/2 flex h-8 w-8 -translate-y-1/2 items-center justify-center rounded-lg text-slate-400 opacity-0 transition hover:bg-red-50 hover:text-red-600 group-hover:opacity-100"
                  aria-label="Delete chat"
                  onClick={(e) => {
                    e.stopPropagation();
                    onDeleteSession(s.id);
                  }}
                >
                  <Trash2 className="h-4 w-4" />
                </button>
              ) : null}
            </li>
          ))}
        </ul>
      </div>

      <div className={`mt-auto border-t border-slate-100 bg-slate-50/80 ${narrow ? "px-1 py-2" : "px-3 py-3"}`}>
        {!narrow ? (
          <div className="mb-3">
            <button
              type="button"
              onClick={() => {
                void pingHealth();
                setHealthOpen((v) => !v);
              }}
              className="flex w-full items-center gap-2 rounded-xl bg-white px-2 py-2 text-left text-xs shadow-sm ring-1 ring-slate-200/80 transition hover:bg-slate-50"
            >
              <span
                className={`h-2 w-2 shrink-0 rounded-full ${healthOk ? "animate-pulse bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.7)]" : "bg-amber-500"}`}
              />
              <div className="min-w-0 flex-1">
                <div className="font-semibold text-slate-800">
                  {healthOk ? "System healthy" : healthOk === false ? "API issue" : "Checking…"}
                </div>
                {healthMs != null ? (
                  <div className="text-[10px] text-slate-500">Ping ~{healthMs}ms · tap for details</div>
                ) : (
                  <div className="text-[10px] text-slate-500">Tap for details</div>
                )}
              </div>
              <ChevronRight className={`h-4 w-4 shrink-0 text-slate-400 transition ${healthOpen ? "rotate-90" : ""}`} />
            </button>
          </div>
        ) : (
          <button
            type="button"
            title="Health"
            onClick={() => void pingHealth()}
            className="mx-auto mb-2 flex h-10 w-10 items-center justify-center rounded-xl bg-white shadow-sm ring-1 ring-slate-200/80"
          >
            <span
              className={`h-2.5 w-2.5 rounded-full ${healthOk ? "bg-emerald-500" : "bg-amber-500"}`}
            />
          </button>
        )}

        {healthOpen && !narrow ? (
          <div className="mb-3 rounded-xl border border-slate-200 bg-white px-3 py-2 text-[11px] text-slate-600 shadow-sm">
            <p>
              <strong className="text-slate-800">Health</strong> — pings <code className="rounded bg-slate-100 px-1">/health</code>{" "}
              every minute. Webhooks run async; failures are logged server-side.
            </p>
          </div>
        ) : null}

        {!narrow ? (
          <div className="mb-2 rounded-xl bg-white px-3 py-2 ring-1 ring-slate-200/80">
            <div className="text-xs font-semibold text-slate-900">
              {user.full_name?.trim() ? `Welcome, ${user.full_name.split(" ")[0]}` : "Welcome"}
            </div>
            <div className="truncate text-[10px] text-slate-500">{user.email}</div>
          </div>
        ) : null}

        <div className={`flex items-center gap-2 ${narrow ? "flex-col" : "justify-between"}`}>
          <button
            type="button"
            onClick={() => {
              onNewChat();
              onMobileOpenChange(false);
            }}
            className={`inline-flex items-center justify-center gap-1 rounded-xl border border-slate-200 bg-white text-xs font-medium text-slate-700 shadow-sm transition hover:bg-slate-50 ${narrow ? "h-10 w-10" : "flex-1 px-2 py-2"}`}
            title="Refresh session"
          >
            <RefreshCw className="h-4 w-4" />
            {!narrow ? <span>New</span> : null}
          </button>
          <button
            type="button"
            onClick={() => {
              logout();
              window.location.href = "/login";
            }}
            className={`inline-flex items-center justify-center gap-1 rounded-xl border border-slate-200 bg-white text-xs font-medium text-slate-800 shadow-sm transition hover:bg-slate-50 ${narrow ? "h-10 w-10" : "flex-1 px-2 py-2"}`}
          >
            <LogOut className="h-4 w-4" />
            {!narrow ? <span>Log out</span> : null}
          </button>
        </div>
        {!narrow ? (
          <div className="mt-2 text-center text-[10px] text-slate-400">UI v0.1.0</div>
        ) : (
          <div className="mt-2 text-center text-[10px] text-slate-400">v0.1</div>
        )}
      </div>
    </aside>
  );
}
