import { useCallback, useEffect, useState } from "react";
import { Navigate } from "react-router-dom";
import {
  createSession,
  deleteSession,
  listMessages,
  listSessions,
  type ChatMessage,
  type Session,
} from "../api/sessions";
import { ChatPanel } from "../components/ChatPanel";
import { InfoWorkspace, type InfoView } from "../components/InfoWorkspace";
import { Sidebar } from "../components/Sidebar";
import { SidebarMobileTrigger } from "../components/SidebarMobileTrigger";
import { WelcomeModal } from "../components/WelcomeModal";
import { useAuth } from "../context/AuthContext";
import { useToast } from "../context/ToastContext";
import type { StreamAnalysis } from "../lib/analysisFormatter";

export function ChatPage() {
  const { user, loading, logout, refreshAccess } = useAuth();
  const { toast } = useToast();
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [welcomeOpen, setWelcomeOpen] = useState(false);
  const [mainView, setMainView] = useState<InfoView>("chat");
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [lastAnalysis, setLastAnalysis] = useState<StreamAnalysis | null>(null);

  const loadSessions = useCallback(async () => {
    try {
      const s = await listSessions();
      setSessions(s);
    } catch {
      const ok = await refreshAccess();
      if (ok) {
        setSessions(await listSessions());
      } else {
        logout();
      }
    }
  }, [logout, refreshAccess]);

  useEffect(() => {
    if (!user) return;
    void loadSessions();
    setWelcomeOpen(!user.onboarding_completed);
  }, [user, loadSessions]);

  useEffect(() => {
    if (!activeId) {
      setMessages([]);
      return;
    }
    void (async () => {
      try {
        setMessages(await listMessages(activeId));
      } catch {
        setMessages([]);
      }
    })();
  }, [activeId]);

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center text-slate-500">Loading…</div>
    );
  }
  if (!user) {
    return <Navigate to="/login" replace />;
  }

  const handleNewChat = async () => {
    const s = await createSession("New trip");
    setSessions((prev) => [s, ...prev]);
    setActiveId(s.id);
    setMessages([]);
    setMainView("chat");
  };

  const handleDeleteSession = async (id: string) => {
    if (!window.confirm("Delete this chat permanently?")) return;
    try {
      await deleteSession(id);
      toast("Chat deleted", "success");
      if (activeId === id) {
        setActiveId(null);
        setMessages([]);
      }
      await loadSessions();
    } catch (e) {
      toast((e as Error).message, "error");
    }
  };

  const chatTitle = sessions.find((s) => s.id === activeId)?.title ?? "New trip";

  return (
    <div className="relative flex h-full overflow-hidden bg-stp-app">
      <WelcomeModal open={welcomeOpen} onClose={() => setWelcomeOpen(false)} />

      <div
        className={`fixed inset-0 z-40 bg-slate-900/35 backdrop-blur-[1px] transition-opacity lg:hidden ${
          mobileMenuOpen ? "opacity-100" : "pointer-events-none opacity-0"
        }`}
        aria-hidden={!mobileMenuOpen}
        onClick={() => setMobileMenuOpen(false)}
      />

      <Sidebar
        sessions={sessions}
        activeId={activeId}
        mainView={mainView}
        mobileOpen={mobileMenuOpen}
        onMobileOpenChange={setMobileMenuOpen}
        onSelect={(id) => setActiveId(id)}
        onNewChat={() => void handleNewChat()}
        onLogoClick={() => void handleNewChat()}
        onNavigate={(v) => setMainView(v)}
        onDeleteSession={(id) => void handleDeleteSession(id)}
        user={user}
        logout={logout}
      />

      <div className="flex min-h-0 min-w-0 flex-1 flex-col lg:ml-0">
        {mainView === "chat" ? (
          <ChatPanel
            sessionId={activeId}
            initialMessages={messages}
            chatTitle={chatTitle}
            mobileHeader={<SidebarMobileTrigger onClick={() => setMobileMenuOpen(true)} />}
            onSessionResolved={(id) => {
              setActiveId(id);
              void loadSessions();
            }}
            onStreamAnalysis={setLastAnalysis}
            onClearThread={() => void handleNewChat()}
          />
        ) : (
          <div className="flex min-h-0 flex-1 flex-col pt-2 lg:pt-0">
            <div className="flex shrink-0 items-center gap-2 border-b border-slate-200/80 bg-white/90 px-4 py-2 backdrop-blur-md lg:hidden">
              <SidebarMobileTrigger onClick={() => setMobileMenuOpen(true)} />
              <span className="text-sm font-semibold text-slate-800">Information</span>
            </div>
            <InfoWorkspace view={mainView} analysis={lastAnalysis} />
          </div>
        )}
      </div>
    </div>
  );
}
