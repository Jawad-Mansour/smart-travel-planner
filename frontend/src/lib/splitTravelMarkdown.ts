/** Mirrors backend ``split_travel_answer_segments`` for hydrating DB messages into chat bubbles. */

export function splitTravelMarkdown(answer: string): string[] | null {
  const text = (answer || "").trim();
  if (!text || !text.includes("### 1.")) return null;

  const recMatch = text.match(/\n## My Recommendation\s*\n/);
  let main = text;
  let tail = "";
  if (recMatch && recMatch.index !== undefined) {
    main = text.slice(0, recMatch.index).trimEnd();
    tail = text.slice(recMatch.index).trim();
  }

  const parts = main.split(/\n---\n\n/);
  if (parts.length < 2) return null;

  const intro = parts[0].trim();
  const destParts = parts.slice(1).map((p) => p.trim()).filter(Boolean);
  if (!destParts.length) return null;

  const segments: string[] = [];
  segments.push(`${intro}\n\n${destParts[0]}`.trim());
  for (let i = 1; i < destParts.length; i++) segments.push(destParts[i]);
  if (tail) segments.push(tail);
  return segments.length >= 2 ? segments : null;
}
