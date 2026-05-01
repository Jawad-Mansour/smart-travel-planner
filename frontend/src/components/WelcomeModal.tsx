import { Sparkles } from "lucide-react";
import { apiFetch } from "../api/http";
import { useAuth } from "../context/AuthContext";

export function WelcomeModal({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { setUser, user } = useAuth();

  if (!open) return null;

  const dismiss = async () => {
    try {
      await apiFetch("/api/auth/me/onboarding", {
        method: "PATCH",
        body: JSON.stringify({ onboarding_completed: true }),
      });
      if (user) setUser({ ...user, onboarding_completed: true });
    } catch {
      /* non-fatal */
    }
    onClose();
  };

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-slate-900/45 p-4 backdrop-blur-md">
      <div className="max-h-[90vh] w-full max-w-lg overflow-y-auto rounded-3xl border border-white/50 bg-white/90 p-8 shadow-2xl shadow-indigo-900/15 backdrop-blur-xl">
        <div className="flex items-start gap-3">
          <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl bg-gradient-to-br from-indigo-500 to-purple-600 text-white shadow-lg">
            <Sparkles className="h-5 w-5" />
          </span>
          <div>
            <h2 className="text-xl font-bold tracking-tight text-slate-900">Welcome to Smart Travel Planner</h2>
            <p className="mt-2 text-sm leading-relaxed text-slate-600">
              I help you find destinations based on <strong>budget</strong>, <strong>travel dates</strong>, and{" "}
              <strong>interests</strong> — then assemble weather, flight estimates, and FX context around a clear
              recommendation.
            </p>
          </div>
        </div>

        <ul className="mt-6 space-y-3 rounded-2xl bg-indigo-50/60 p-4 text-sm text-slate-700 ring-1 ring-indigo-100/80">
          <li className="flex gap-2">
            <span className="font-semibold text-indigo-800">RAG</span>
            <span>Grounding from Wikivoyage-derived destination guides (retrieval + citations in tooling).</span>
          </li>
          <li className="flex gap-2">
            <span className="font-semibold text-indigo-800">ML</span>
            <span>Travel-style classifier to bias destinations toward how you like to travel.</span>
          </li>
          <li className="flex gap-2">
            <span className="font-semibold text-indigo-800">Live APIs</span>
            <span>Weather, flight estimates, and exchange rates when keys are configured.</span>
          </li>
        </ul>

        <div className="mt-8 flex flex-col gap-2 sm:flex-row sm:justify-end">
          <button type="button" onClick={dismiss} className="btn-secondary order-2 sm:order-1">
            Don&apos;t show again
          </button>
          <button type="button" onClick={dismiss} className="btn-primary order-1 sm:order-2">
            Get started
          </button>
        </div>
      </div>
    </div>
  );
}
