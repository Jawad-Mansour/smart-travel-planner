/** Normalize SSE ``tool_results`` / intent for the “Current Response Analysis” panel. */

export type StreamAnalysis = {
  elapsed_seconds?: number;
  intent: Record<string, unknown> | null;
  tool_results: Record<string, unknown>;
  webhook_status?: string;
  usage_parts?: unknown[];
  query_embedding_preview?: number[];
};

export type RagChunkLine = { destination: string; heading: string };

export type RagDetailRow = {
  destination: string;
  heading: string;
  score?: number;
  snippet: string;
  source?: string;
};

function num(v: unknown): number | undefined {
  if (typeof v === "number" && !Number.isNaN(v)) return v;
  if (typeof v === "string" && v.trim() && !Number.isNaN(Number(v))) return Number(v);
  return undefined;
}

function str(v: unknown): string | undefined {
  if (typeof v === "string" && v.trim()) return v;
  return undefined;
}

export function formatIntentSummary(intent: Record<string, unknown> | null): string[] {
  if (!intent || typeof intent !== "object") return ["—"];
  const parts: string[] = [];
  const d = num(intent.duration_days);
  const b = num(intent.budget_usd);
  const month = str(intent.timing_or_season);
  const acts = Array.isArray(intent.activities) ? intent.activities.map(String).filter(Boolean) : [];
  const style = str(intent.traveler_style);
  const temp = str(intent.temperature_preference);
  if (d) parts.push(`${d} days`);
  if (b != null) parts.push(`$${Math.round(b)} budget`);
  if (month) parts.push(month);
  if (acts.length) parts.push(acts.slice(0, 4).join(", "));
  if (style) parts.push(style);
  if (temp) parts.push(`${temp} weather`);
  return parts.length ? parts : ["—"];
}

export function extractMl(tool_results: Record<string, unknown>): {
  style: string;
  confidence: string;
  signals: string[];
} {
  const raw = tool_results.classifier;
  if (!raw || typeof raw !== "object") return { style: "—", confidence: "—", signals: [] };
  const env = raw as Record<string, unknown>;
  const payload = env.payload && typeof env.payload === "object" ? (env.payload as Record<string, unknown>) : null;
  if (!payload) return { style: "—", confidence: "—", signals: [] };
  const style = str(payload.travel_style) ?? "—";
  const c = num(payload.confidence);
  const confidence = c != null ? `${Math.round(c * 100)}%` : "—";
  const sig = payload.signal_features;
  const signals = Array.isArray(sig) ? sig.map(String).filter(Boolean).slice(0, 6) : [];
  return { style, confidence, signals };
}

export function extractRagChunks(tool_results: Record<string, unknown>): RagChunkLine[] {
  const raw = tool_results.rag;
  if (!raw || typeof raw !== "object") return [];
  const env = raw as Record<string, unknown>;
  const payload = env.payload && typeof env.payload === "object" ? (env.payload as Record<string, unknown>) : null;
  const chunks = payload?.chunks;
  if (!Array.isArray(chunks)) return [];
  const out: RagChunkLine[] = [];
  for (const row of chunks) {
    if (!row || typeof row !== "object") continue;
    const r = row as Record<string, unknown>;
    const destination = str(r.destination) ?? "Unknown";
    const heading = str(r.heading) ?? "Section";
    out.push({ destination, heading });
  }
  return out.slice(0, 12);
}

export function extractRagDetailed(tool_results: Record<string, unknown>, limit = 5): RagDetailRow[] {
  const raw = tool_results.rag;
  if (!raw || typeof raw !== "object") return [];
  const env = raw as Record<string, unknown>;
  const payload = env.payload && typeof env.payload === "object" ? (env.payload as Record<string, unknown>) : null;
  const chunks = payload?.chunks;
  if (!Array.isArray(chunks)) return [];
  const out: RagDetailRow[] = [];
  for (const row of chunks) {
    if (!row || typeof row !== "object") continue;
    const r = row as Record<string, unknown>;
    const destination = str(r.destination) ?? "Unknown";
    const heading = str(r.heading) ?? "Section";
    const content = str(r.content) ?? "";
    const snippet = content.length > 220 ? `${content.slice(0, 220)}…` : content;
    const score = num(r.retrieval_score);
    const source = str(r.source_url);
    out.push({ destination, heading, snippet, score, source });
    if (out.length >= limit) break;
  }
  return out;
}

export function extractToolLatencyMs(tool_results: Record<string, unknown>, key: string): number | undefined {
  const raw = tool_results[key];
  if (!raw || typeof raw !== "object") return undefined;
  const ms = num((raw as Record<string, unknown>).duration_ms);
  const pl = (raw as Record<string, unknown>).payload;
  if (ms != null) return Math.round(ms);
  if (pl && typeof pl === "object") {
    const inner = num((pl as Record<string, unknown>).duration_ms);
    if (inner != null) return Math.round(inner);
  }
  return undefined;
}

export function extractFxSummary(tool_results: Record<string, unknown>): string {
  const raw = tool_results.fx;
  if (!raw || typeof raw !== "object") return "—";
  const env = raw as Record<string, unknown>;
  const ok = env.ok === true;
  const payload = env.payload && typeof env.payload === "object" ? (env.payload as Record<string, unknown>) : null;
  const base = payload ? str(payload.base_code) : undefined;
  const rates = payload?.rates && typeof payload.rates === "object" ? payload.rates : null;
  const sample = rates && typeof rates === "object" ? Object.keys(rates as object).slice(0, 4).join(", ") : "";
  return ok ? `FX ok${base ? ` · base ${base}` : ""}${sample ? ` · samples: ${sample}` : ""}` : "FX unavailable";
}

export function formatUsageSteps(usage: unknown[] | undefined): { step: string; prompt?: number; completion?: number }[] {
  if (!Array.isArray(usage)) return [];
  const rows: { step: string; prompt?: number; completion?: number }[] = [];
  for (const item of usage) {
    if (!item || typeof item !== "object") continue;
    const u = item as Record<string, unknown>;
    const step = str(u.step) ?? "step";
    rows.push({
      step,
      prompt: num(u.prompt_tokens),
      completion: num(u.completion_tokens),
    });
  }
  return rows;
}

export function extractWeatherLines(tool_results: Record<string, unknown>): string[] {
  const w = tool_results.weather;
  if (!Array.isArray(w)) return [];
  const lines: string[] = [];
  for (const item of w) {
    if (!item || typeof item !== "object") continue;
    const o = item as Record<string, unknown>;
    const city = str(o.city) ?? "?";
    const env = o.envelope;
    let summary = "";
    if (env && typeof env === "object") {
      const e = env as Record<string, unknown>;
      const pl = e.payload && typeof e.payload === "object" ? (e.payload as Record<string, unknown>) : null;
      const forecast = pl?.forecast && typeof pl.forecast === "object" ? (pl.forecast as Record<string, unknown>) : null;
      const tmin = forecast ? num(forecast.temp_min ?? forecast.min_temp_c) : undefined;
      const tmax = forecast ? num(forecast.temp_max ?? forecast.max_temp_c) : undefined;
      const cond = forecast ? str(forecast.description ?? forecast.summary ?? forecast.conditions) : undefined;
      if (tmin != null && tmax != null) summary = `${tmin}–${tmax}°C`;
      else if (tmax != null) summary = `${tmax}°C`;
      if (cond) summary = summary ? `${summary}, ${cond}` : cond;
    }
    lines.push(summary ? `${city}: ${summary}` : `${city}: (see agent reply)`);
  }
  return lines;
}

export function extractFlightLines(tool_results: Record<string, unknown>): string[] {
  const f = tool_results.flights;
  if (!Array.isArray(f)) return [];
  const lines: string[] = [];
  for (const item of f) {
    if (!item || typeof item !== "object") continue;
    const o = item as Record<string, unknown>;
    const payload = o.payload && typeof o.payload === "object" ? (o.payload as Record<string, unknown>) : null;
    const est =
      payload?.estimate && typeof payload.estimate === "object"
        ? (payload.estimate as Record<string, unknown>)
        : null;
    if (!est) continue;
    const orig = str(est.origin_display);
    const dest = str(est.destination_display);
    const p = num(est.round_trip_price_usd_estimate);
    const route = orig && dest ? `${orig}→${dest}` : dest ?? "Flight";
    lines.push(p != null ? `${route}: ~$${Math.round(p)}` : route);
  }
  return lines;
}

export function webhookLabel(status: string | undefined): string {
  switch (status) {
    case "queued":
      return "Queued (Discord, Slack, and/or email after reply)";
    case "not_configured":
      return "Not configured";
    case "not_sent":
      return "Not sent (clarification or empty reply)";
    case "skipped_clarification":
      return "Skipped (awaiting details)";
    default:
      return status ?? "—";
  }
}
