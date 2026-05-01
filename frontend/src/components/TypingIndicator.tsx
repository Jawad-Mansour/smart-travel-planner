export function TypingIndicator() {
  return (
    <div className="flex items-center gap-1 px-1 py-2" aria-hidden>
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="inline-block h-2 w-2 rounded-full bg-slate-400/90 animate-bounce"
          style={{ animationDelay: `${i * 160}ms`, animationDuration: "0.9s" }}
        />
      ))}
    </div>
  );
}
