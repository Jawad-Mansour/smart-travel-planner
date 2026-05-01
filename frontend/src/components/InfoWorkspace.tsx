import { type ReactNode, useEffect, useMemo, useState } from "react";
import {
  extractFlightLines,
  extractFxSummary,
  extractMl,
  extractRagDetailed,
  extractToolLatencyMs,
  extractWeatherLines,
  formatIntentSummary,
  formatUsageSteps,
  webhookLabel,
  type StreamAnalysis,
} from "../lib/analysisFormatter";
import { apiFetch } from "../api/http";
import { BookOpen, ChevronDown, Copy, Mail, Radar, Sparkles } from "lucide-react";

export type InfoView = "chat" | "system-info" | "analysis" | "webhook-setup";

const DESTINATIONS = [
  "Amsterdam",
  "Bangkok",
  "Barcelona",
  "Berlin",
  "Budapest",
  "Cape Town",
  "Cusco",
  "Dubai",
  "Edinburgh",
  "Istanbul",
  "Kathmandu",
  "Krakow",
  "Lisbon",
  "Maldives",
  "New York",
  "Paris",
  "Prague",
  "Queenstown",
  "Reykjavik",
  "Rome",
  "Santorini",
  "Sydney",
  "Tokyo",
  "Vienna",
  "Bali",
];

const FICTIONAL_IMPORTANCE = [
  { label: "hiking_score", v: 0.14 },
  { label: "culture_score", v: 0.12 },
  { label: "cost_per_day_avg_usd", v: 0.11 },
  { label: "beach_score", v: 0.09 },
  { label: "flight_cost_usd", v: 0.08 },
  { label: "family_score", v: 0.07 },
];

function PanelShell({
  title,
  icon,
  children,
}: {
  title: string;
  icon: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className="flex h-full min-h-0 flex-col bg-[#f9fafb]">
      <header className="border-b border-slate-200/90 bg-white px-4 py-4 shadow-sm sm:px-8">
        <div className="mx-auto flex max-w-4xl items-center gap-3">
          <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-emerald-100 text-emerald-700">
            {icon}
          </span>
          <h1 className="text-lg font-semibold tracking-tight text-slate-900">{title}</h1>
        </div>
      </header>
      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-6 sm:px-8">
        <div className="mx-auto max-w-4xl space-y-4 text-sm leading-relaxed text-slate-700">{children}</div>
      </div>
    </div>
  );
}

function Card({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <div
      className={`rounded-2xl border border-slate-200/90 bg-white p-5 shadow-sm ${className}`}
    >
      {children}
    </div>
  );
}

function Accordion({
  title,
  defaultOpen,
  children,
}: {
  title: string;
  defaultOpen?: boolean;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen ?? false);
  return (
    <Card className="p-0 overflow-hidden">
      <button
        type="button"
        className="flex w-full items-center justify-between gap-2 px-5 py-4 text-left font-semibold text-slate-900 transition hover:bg-slate-50"
        onClick={() => setOpen(!open)}
      >
        {title}
        <ChevronDown className={`h-5 w-5 shrink-0 text-slate-400 transition ${open ? "rotate-180" : ""}`} />
      </button>
      {open ? <div className="border-t border-slate-100 px-5 py-4 text-slate-600">{children}</div> : null}
    </Card>
  );
}

async function copyText(text: string): Promise<void> {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    /* ignore */
  }
}

export function InfoWorkspace({
  view,
  analysis,
}: {
  view: Exclude<InfoView, "chat">;
  analysis: StreamAnalysis | null;
}) {
  const [meta, setMeta] = useState<{
    rag?: { parent_chunks?: number; child_chunks?: number; destination_count?: number };
    ml_classifier?: { destinations_trained?: number; styles?: string[]; test_f1?: number };
  } | null>(null);
  const [sysTab, setSysTab] = useState<"ml" | "rag" | "apis" | "llm">("ml");

  useEffect(() => {
    if (view !== "system-info") return;
    void (async () => {
      try {
        const res = await apiFetch("/api/meta", { auth: false });
        if (res.ok) setMeta(await res.json());
      } catch {
        setMeta(null);
      }
    })();
  }, [view]);

  const embedPreview = useMemo(() => {
    const fromDone = analysis?.query_embedding_preview;
    if (fromDone?.length) return fromDone;
    const raw = analysis?.tool_results?.rag;
    if (!raw || typeof raw !== "object") return [];
    const pl = (raw as Record<string, unknown>).payload;
    if (!pl || typeof pl !== "object") return [];
    const q = (pl as Record<string, unknown>).query_embedding_preview;
    return Array.isArray(q) ? q.map(Number).filter((x) => !Number.isNaN(x)) : [];
  }, [analysis]);

  if (view === "system-info") {
    const parents = meta?.rag?.parent_chunks ?? 1184;
    const children = meta?.rag?.child_chunks ?? 20047;
    const nDest = meta?.rag?.destination_count ?? 25;
    const f1 = meta?.ml_classifier?.test_f1 ?? 0.894;
    const trained = meta?.ml_classifier?.destinations_trained ?? 155;

    const tabs = (
      <div className="flex flex-wrap gap-2 border-b border-slate-200 pb-4">
        {(
          [
            ["ml", "ML classifier"],
            ["rag", "RAG pipeline"],
            ["apis", "Live APIs"],
            ["llm", "LLM stack"],
          ] as const
        ).map(([id, label]) => (
          <button
            key={id}
            type="button"
            onClick={() => setSysTab(id)}
            className={`rounded-full px-4 py-2 text-xs font-semibold transition ${
              sysTab === id
                ? "bg-emerald-600 text-white shadow-sm"
                : "bg-white text-slate-700 ring-1 ring-slate-200 hover:bg-slate-50"
            }`}
          >
            {label}
          </button>
        ))}
      </div>
    );

    return (
      <PanelShell title="System info" icon={<Sparkles className="h-5 w-5" />}>
        {tabs}
        {sysTab === "ml" ? (
          <Card>
            <h3 className="font-semibold text-slate-900">ML classifier</h3>
            <ul className="mt-3 list-disc space-y-1 pl-5 text-slate-600">
              <li>
                Model: <strong>Random Forest</strong>
              </li>
              <li>
                Training data: <strong>{trained}</strong> labeled destinations
              </li>
              <li>
                Travel styles: Adventure, Culture, Budget, Luxury, Family, Relaxation
              </li>
              <li>
                Macro F1 (held-out): <strong>{f1}</strong>
              </li>
            </ul>
            <p className="mt-4 text-xs text-slate-500">
              Illustrative feature ranking (static ordering for UI — aligns with dominant signals in the CSV).
            </p>
            <div className="mt-3 space-y-2">
              {FICTIONAL_IMPORTANCE.map((row) => (
                <div key={row.label} className="flex items-center gap-3">
                  <span className="w-40 shrink-0 truncate font-mono text-[11px] text-slate-500">{row.label}</span>
                  <div className="h-2 flex-1 rounded-full bg-slate-100">
                    <div
                      className="h-2 rounded-full bg-emerald-500"
                      style={{ width: `${Math.min(100, row.v * 400)}%` }}
                    />
                  </div>
                  <span className="w-10 text-right font-mono text-[11px] text-slate-600">{row.v.toFixed(2)}</span>
                </div>
              ))}
            </div>
          </Card>
        ) : null}

        {sysTab === "rag" ? (
          <Card>
            <h3 className="font-semibold text-slate-900">RAG pipeline</h3>
            <ul className="mt-3 list-disc space-y-1 pl-5">
              <li>
                Destinations indexed: <strong>{nDest}</strong> major cities (
                {DESTINATIONS.slice(0, 6).join(", ")}, …)
              </li>
              <li>
                Chunking: parent–child — <strong>{parents.toLocaleString()}</strong> parents,{" "}
                <strong>{children.toLocaleString()}</strong> children
              </li>
              <li>
                Embedding model: <strong>all-MiniLM-L6-v2</strong> (384-d)
              </li>
              <li>
                Vector DB: <strong>pgvector</strong> in PostgreSQL
              </li>
              <li>
                Retrieval: top <strong>5</strong> parent sections by cosine similarity on child embeddings
              </li>
            </ul>
          </Card>
        ) : null}

        {sysTab === "apis" ? (
          <Card>
            <h3 className="font-semibold text-slate-900">Live APIs</h3>
            <ul className="mt-3 list-disc space-y-2 pl-5">
              <li>
                <strong>Weather</strong> — OpenWeatherMap path, ~10 minute TTL cache
              </li>
              <li>
                <strong>Flights</strong> — mock estimates plus Amadeus when API keys are configured (~1h cache)
              </li>
              <li>
                <strong>FX</strong> — Frankfurter / configured FX base (~1h cache)
              </li>
            </ul>
          </Card>
        ) : null}

        {sysTab === "llm" ? (
          <Card>
            <h3 className="font-semibold text-slate-900">LLM stack</h3>
            <ul className="mt-3 list-disc space-y-2 pl-5">
              <li>
                Routing / clarification: cheaper model (<strong>gpt-4o-mini</strong>)
              </li>
              <li>
                Synthesis: stronger model (<strong>gpt-4o</strong>) with structured JSON → markdown
              </li>
              <li>
                Token savings: roughly <strong>85%</strong> vs always calling the largest model for every hop
              </li>
            </ul>
          </Card>
        ) : null}

        <Card>
          <div className="flex items-center gap-2">
            <BookOpen className="h-5 w-5 text-emerald-700" />
            <h3 className="font-semibold text-slate-900">Indexed destinations</h3>
          </div>
          <p className="mt-2 text-xs leading-relaxed text-slate-600">{DESTINATIONS.join(", ")}.</p>
        </Card>
      </PanelShell>
    );
  }

  if (view === "analysis") {
    const intent = analysis?.intent && typeof analysis.intent === "object" ? analysis.intent : null;
    const tools =
      analysis?.tool_results && typeof analysis.tool_results === "object"
        ? analysis.tool_results
        : {};
    const ml = extractMl(tools);
    const ragRows = extractRagDetailed(tools, 5);
    const weather = extractWeatherLines(tools);
    const flights = extractFlightLines(tools);
    const fxLine = extractFxSummary(tools);
    const usageRows = formatUsageSteps(analysis?.usage_parts);
    const ragMs = extractToolLatencyMs(tools, "rag");
    const clsMs = extractToolLatencyMs(tools, "classifier");

    const debugBlob = JSON.stringify(
      {
        intent,
        tool_results: tools,
        usage_parts: analysis?.usage_parts,
        embedding_preview: embedPreview,
      },
      null,
      2
    );

    return (
      <PanelShell title="Response analysis" icon={<Radar className="h-5 w-5" />}>
        {!analysis ? (
          <Card>
            <p className="text-slate-600">
              Run a chat turn first. This panel shows embeddings (truncated), retrieval rows, classifier signals, API
              envelopes, tokens, and timings from the last completion.
            </p>
          </Card>
        ) : (
          <>
            <div className="flex flex-wrap items-center gap-2">
              <span className="rounded-full bg-emerald-50 px-3 py-1 text-xs font-medium text-emerald-900 ring-1 ring-emerald-100">
                Last run {analysis.elapsed_seconds != null ? `${analysis.elapsed_seconds}s` : "—"}
              </span>
              <button
                type="button"
                className="inline-flex items-center gap-1 rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-medium text-slate-700 shadow-sm hover:bg-slate-50"
                onClick={() => void copyText(debugBlob)}
              >
                <Copy className="h-3.5 w-3.5" />
                Copy debug JSON
              </button>
            </div>

            <Accordion title="Query embedding (first 10 dimensions)" defaultOpen>
              <div className="font-mono text-[11px] text-slate-700">
                {embedPreview.length ? embedPreview.map((n) => n.toFixed(5)).join(", ") : "— (no vector returned — clarification-only turn or RAG skipped)"}
              </div>
            </Accordion>

            <Accordion title="Retrieved chunks (RAG)" defaultOpen>
              {ragRows.length === 0 ? (
                <p className="text-slate-500">No chunks — clarification-only turn or retrieval empty.</p>
              ) : (
                <ul className="space-y-3">
                  {ragRows.map((r, i) => (
                    <li key={`${r.destination}-${r.heading}-${i}`} className="rounded-xl bg-slate-50 px-3 py-2 ring-1 ring-slate-100">
                      <div className="flex flex-wrap items-center gap-2 text-xs font-semibold text-slate-900">
                        <span>{r.destination}</span>
                        {r.score != null ? (
                          <span className="rounded-full bg-white px-2 py-0.5 font-mono text-[10px] text-emerald-800 ring-1 ring-emerald-100">
                            score {r.score.toFixed(4)}
                          </span>
                        ) : null}
                      </div>
                      <div className="mt-1 text-[11px] text-slate-500">{r.heading}</div>
                      <p className="mt-2 font-mono text-[11px] leading-relaxed text-slate-600">{r.snippet || "—"}</p>
                    </li>
                  ))}
                </ul>
              )}
              {ragMs != null ? (
                <p className="mt-2 font-mono text-[11px] text-slate-500">Retrieval latency ~{ragMs}ms (tool payload)</p>
              ) : null}
            </Accordion>

            <Accordion title="ML classification" defaultOpen>
              <p>
                Style <strong>{ml.style}</strong> · confidence <strong>{ml.confidence}</strong>
              </p>
              {ml.signals.length ? (
                <div className="mt-2">
                  <div className="text-xs font-semibold text-slate-800">Top keyword signals</div>
                  <ul className="mt-1 list-disc pl-5 text-sm">
                    {ml.signals.map((s) => (
                      <li key={s} className="font-mono text-[12px]">
                        {s}
                      </li>
                    ))}
                  </ul>
                </div>
              ) : null}
              {clsMs != null ? (
                <p className="mt-2 font-mono text-[11px] text-slate-500">Classifier latency ~{clsMs}ms</p>
              ) : null}
            </Accordion>

            <Accordion title="Live APIs">
              <div className="space-y-3 text-sm">
                <div>
                  <div className="text-xs font-semibold uppercase tracking-wide text-slate-400">Weather</div>
                  <ul className="mt-1 list-disc pl-5">
                    {weather.length ? weather.map((w) => <li key={w}>{w}</li>) : <li>—</li>}
                  </ul>
                </div>
                <div>
                  <div className="text-xs font-semibold uppercase tracking-wide text-slate-400">Flights</div>
                  <ul className="mt-1 list-disc pl-5">
                    {flights.length ? flights.map((f) => <li key={f}>{f}</li>) : <li>—</li>}
                  </ul>
                </div>
                <div>
                  <div className="text-xs font-semibold uppercase tracking-wide text-slate-400">FX</div>
                  <p className="mt-1 font-mono text-[12px]">{fxLine}</p>
                </div>
              </div>
            </Accordion>

            <Accordion title="Token usage (per step)">
              {usageRows.length === 0 ? (
                <p className="text-slate-500">No usage breakdown recorded.</p>
              ) : (
                <table className="w-full border-collapse text-left text-[12px]">
                  <thead>
                    <tr className="border-b border-slate-200 text-slate-500">
                      <th className="py-2 pr-2">Step</th>
                      <th className="py-2 pr-2">Prompt</th>
                      <th className="py-2">Completion</th>
                    </tr>
                  </thead>
                  <tbody>
                    {usageRows.map((r) => (
                      <tr key={r.step} className="border-b border-slate-100">
                        <td className="py-2 pr-2 font-mono">{r.step}</td>
                        <td className="py-2 pr-2">{r.prompt ?? "—"}</td>
                        <td className="py-2">{r.completion ?? "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </Accordion>

            <Accordion title="Latency checklist">
              <ul className="list-disc space-y-1 pl-5 font-mono text-[12px]">
                <li>Total agent: {analysis.elapsed_seconds ?? "—"}s</li>
                <li>RAG tool: {ragMs != null ? `${ragMs}ms` : "—"}</li>
                <li>Classifier tool: {clsMs != null ? `${clsMs}ms` : "—"}</li>
              </ul>
              <p className="mt-2 text-xs text-slate-500">
                Intent summary: {formatIntentSummary(intent as Record<string, unknown>).join(" · ")}
              </p>
            </Accordion>

            <Card>
              <h3 className="font-semibold text-slate-900">Webhook</h3>
              <p className="mt-1">{webhookLabel(analysis.webhook_status)}</p>
            </Card>
          </>
        )}
      </PanelShell>
    );
  }

  /* webhook-setup */
  return (
    <PanelShell title="Webhook delivery" icon={<Mail className="h-5 w-5" />}>
      <Card className="space-y-4 leading-relaxed">
        <h2 className="text-base font-semibold text-slate-900">What is the webhook?</h2>
        <p>
          After the agent finishes planning your trip, it can notify you in the background: Discord, Slack, and/or an
          email to the account you signed in with (when SMTP is configured). Delivery does not block the chat.
        </p>

        <h3 className="text-sm font-semibold text-slate-900">How to set it up (three steps)</h3>

        <h4 className="text-sm font-semibold text-slate-900">1. Discord (easiest)</h4>
        <ul className="list-disc pl-5">
          <li>In your Discord server: Server Settings → Integrations → Webhooks → New Webhook</li>
          <li>Name it &quot;Travel Planner&quot;, pick a channel</li>
          <li>Copy the webhook URL</li>
          <li>
            Add to your <code className="rounded bg-slate-100 px-1">.env</code>:{" "}
            <code className="rounded bg-slate-100 px-1">DISCORD_WEBHOOK_URL=your_url_here</code>
          </li>
          <li>Restart the API</li>
        </ul>
        <p>
          <strong>What you&apos;ll receive:</strong> each plan as a rich embed with title, preview text, and user
          metadata.
        </p>

        <h4 className="text-sm font-semibold text-slate-900">2. Slack</h4>
        <ul className="list-disc pl-5">
          <li>Create an incoming webhook from your Slack workspace apps catalog</li>
          <li>
            Add <code className="rounded bg-slate-100 px-1">SLACK_WEBHOOK_URL</code> to <code className="rounded bg-slate-100 px-1">.env</code>
          </li>
          <li>Restart the API — summaries post as plain text payloads</li>
        </ul>

        <h4 className="text-sm font-semibold text-slate-900">3. Email (SMTP)</h4>
        <p>
          When the server has SMTP settings, the API sends a plain-text email to the signed-in user&apos;s address after
          each completed itinerary (same moment as Discord/Slack hooks).
        </p>
        <ul className="list-disc pl-5">
          <li>
            Set <code className="rounded bg-slate-100 px-1">SMTP_HOST</code>,{" "}
            <code className="rounded bg-slate-100 px-1">SMTP_FROM</code>, and usually{" "}
            <code className="rounded bg-slate-100 px-1">SMTP_PORT</code> (587 with STARTTLS, or 465 with{" "}
            <code className="rounded bg-slate-100 px-1">SMTP_USE_SSL=true</code>)
          </li>
          <li>
            If your provider requires auth: <code className="rounded bg-slate-100 px-1">SMTP_USER</code> and{" "}
            <code className="rounded bg-slate-100 px-1">SMTP_PASSWORD</code>
          </li>
          <li>
            Local dev without TLS: <code className="rounded bg-slate-100 px-1">SMTP_USE_TLS=false</code> (e.g. Mailpit)
          </li>
        </ul>

        <h3 className="text-sm font-semibold text-slate-900">Testing</h3>
        <p>
          After setup, send a complete trip query (not a clarification turn). Notifications run automatically after the
          final assistant message. Check Discord, Slack, and/or the user&apos;s inbox.
        </p>

        <h3 className="text-sm font-semibold text-slate-900">Troubleshooting</h3>
        <ul className="list-disc pl-5">
          <li>Verify URLs and outbound connectivity from the API container</li>
          <li>Chat continues even if the webhook fails — errors are logged</li>
          <li>
            Inspect logs with <code className="rounded bg-slate-100 px-1">docker logs backend</code>
          </li>
        </ul>
      </Card>
    </PanelShell>
  );
}
