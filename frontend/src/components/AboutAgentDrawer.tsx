import { useState } from "react";

export function AboutAgentDrawer() {
  const [open, setOpen] = useState(false);
  return (
    <div className="border-t border-slate-200/80 pt-3">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="flex w-full items-center justify-between rounded-lg px-2 py-2 text-left text-sm font-medium text-slate-700 hover:bg-white/60"
      >
        About the agent
        <span className="text-slate-400">{open ? "−" : "+"}</span>
      </button>
      {open && (
        <div className="mt-2 space-y-2 rounded-lg bg-white/70 p-3 text-xs leading-relaxed text-slate-600">
          <p>
            <strong>RAG</strong> retrieves top passages from embedded destination guides (pgvector +
            MiniLM embeddings).
          </p>
          <p>
            <strong>ML</strong> ranks destinations using your inferred travel style and numeric
            features from the training dataset.
          </p>
          <p>
            <strong>Live APIs</strong> add weather (Open-Meteo / OpenWeather path in backend),
            flight estimates (Amadeus or mock), and cached FX rates.
          </p>
        </div>
      )}
    </div>
  );
}
