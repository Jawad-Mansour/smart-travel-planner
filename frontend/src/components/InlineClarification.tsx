import { useEffect, useMemo, useState } from "react";

const LABELS: Record<string, string> = {
  duration: "Trip length (days)",
  budget: "Total budget (USD)",
  activities: "Key interests (comma-separated)",
  preferred_month: "Preferred month or season",
};

export function InlineClarification({
  fields,
  disabled,
  onSubmit,
}: {
  fields: string[];
  disabled?: boolean;
  onSubmit: (patch: Record<string, unknown>) => void;
}) {
  const [duration, setDuration] = useState("");
  const [budget, setBudget] = useState("");
  const [activities, setActivities] = useState("");
  const [month, setMonth] = useState("");

  useEffect(() => {
    setDuration("");
    setBudget("");
    setActivities("");
    setMonth("");
  }, [fields.join("|")]);

  const ordered = useMemo(() => {
    const order = ["budget", "duration", "activities", "preferred_month"];
    return order.filter((k) => fields.includes(k));
  }, [fields]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const patch: Record<string, unknown> = {};
    if (fields.includes("duration") && duration.trim()) patch.duration_days = Number(duration);
    if (fields.includes("budget") && budget.trim()) patch.budget_usd = Number(budget);
    if (fields.includes("activities") && activities.trim()) {
      patch.activities = activities
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean);
    }
    if (fields.includes("preferred_month") && month.trim()) patch.timing_or_season = month.trim();
    if (Object.keys(patch).length === 0) return;
    onSubmit(patch);
  };

  if (!ordered.length) return null;

  return (
    <form
      onSubmit={handleSubmit}
      className="mt-3 rounded-2xl border border-emerald-200/80 bg-gradient-to-br from-white to-emerald-50/40 p-4 shadow-sm ring-1 ring-emerald-100/60"
    >
      <p className="text-xs font-semibold uppercase tracking-wide text-emerald-800/90">
        Quick details
      </p>
      <p className="mt-1 text-sm text-slate-600">
        Add what you know — each field fades in so it stays easy to scan.
      </p>
      <div className="mt-4 space-y-4">
        {ordered.map((key, i) => (
          <label
            key={key}
            className="block text-sm opacity-0 animate-[stpFadeUp_0.45s_ease-out_forwards]"
            style={{ animationDelay: `${120 + i * 140}ms` }}
          >
            <span className="font-medium text-slate-800">{LABELS[key]}</span>
            {key === "duration" && (
              <input
                className="mt-1.5 w-full rounded-xl border border-slate-200 bg-white px-3 py-2.5 text-sm outline-none ring-emerald-500/0 transition focus:border-emerald-300 focus:ring-2 focus:ring-emerald-200"
                type="number"
                min={1}
                value={duration}
                disabled={disabled}
                onChange={(e) => setDuration(e.target.value)}
              />
            )}
            {key === "budget" && (
              <input
                className="mt-1.5 w-full rounded-xl border border-slate-200 bg-white px-3 py-2.5 text-sm outline-none ring-emerald-500/0 transition focus:border-emerald-300 focus:ring-2 focus:ring-emerald-200"
                type="number"
                min={0}
                value={budget}
                disabled={disabled}
                onChange={(e) => setBudget(e.target.value)}
              />
            )}
            {key === "activities" && (
              <input
                className="mt-1.5 w-full rounded-xl border border-slate-200 bg-white px-3 py-2.5 text-sm outline-none ring-emerald-500/0 transition focus:border-emerald-300 focus:ring-2 focus:ring-emerald-200"
                placeholder="e.g. hiking, culture, food"
                value={activities}
                disabled={disabled}
                onChange={(e) => setActivities(e.target.value)}
              />
            )}
            {key === "preferred_month" && (
              <input
                className="mt-1.5 w-full rounded-xl border border-slate-200 bg-white px-3 py-2.5 text-sm outline-none ring-emerald-500/0 transition focus:border-emerald-300 focus:ring-2 focus:ring-emerald-200"
                placeholder="e.g. October 2026"
                value={month}
                disabled={disabled}
                onChange={(e) => setMonth(e.target.value)}
              />
            )}
          </label>
        ))}
      </div>
      <div className="mt-4 flex justify-end">
        <button
          type="submit"
          disabled={disabled}
          className="rounded-full bg-emerald-600 px-5 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-emerald-700 disabled:cursor-not-allowed disabled:opacity-40"
        >
          Submit details
        </button>
      </div>
    </form>
  );
}
