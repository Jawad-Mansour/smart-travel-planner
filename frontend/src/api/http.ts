const base = "";

/** Parse FastAPI ``{ detail: ... }`` or plain text from a failed response. */
export async function readApiError(res: Response): Promise<string> {
  const text = await res.text();
  if (!text.trim()) return `Request failed (${res.status})`;
  try {
    const j = JSON.parse(text) as { detail?: unknown };
    if (typeof j.detail === "string") return j.detail;
    if (Array.isArray(j.detail)) {
      const msgs = j.detail
        .map((item) => {
          if (item && typeof item === "object" && "msg" in item) {
            return String((item as { msg: unknown }).msg);
          }
          return "";
        })
        .filter(Boolean);
      if (msgs.length) return msgs.join("; ");
    }
  } catch {
    return text.length > 300 ? `${text.slice(0, 300)}…` : text;
  }
  return `Request failed (${res.status})`;
}

export type ApiFetchOptions = RequestInit & {
  auth?: boolean;
  token?: string | null;
};

export async function apiFetch(path: string, opts: ApiFetchOptions = {}): Promise<Response> {
  const { auth = true, token, ...init } = opts;
  const headers = new Headers(init.headers);
  if (!headers.has("Content-Type") && init.body && typeof init.body === "string") {
    headers.set("Content-Type", "application/json");
  }
  if (auth) {
    const t = token ?? localStorage.getItem("stp_access");
    if (t) headers.set("Authorization", `Bearer ${t}`);
  }
  return fetch(`${base}${path}`, { ...init, headers });
}
