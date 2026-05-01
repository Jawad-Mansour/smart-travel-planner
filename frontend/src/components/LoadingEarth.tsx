export function LoadingEarth({ compact }: { compact?: boolean }) {
  if (compact) {
    return (
      <div className="relative h-7 w-7 shrink-0" aria-hidden>
        <svg
          className="h-7 w-7 animate-[spin_12s_linear_infinite] text-emerald-600"
          viewBox="0 0 100 100"
          fill="none"
        >
          <circle cx="50" cy="50" r="38" stroke="currentColor" strokeWidth="3" opacity="0.25" />
          <path
            d="M50 12 A38 38 0 0 1 88 50"
            stroke="currentColor"
            strokeWidth="3"
            strokeLinecap="round"
            opacity="0.9"
          />
        </svg>
        <span className="absolute inset-0 m-auto flex h-4 w-4 items-center justify-center text-[10px]">
          ✈️
        </span>
      </div>
    );
  }
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-8" aria-hidden>
      <div className="relative h-16 w-16">
        <div className="absolute inset-0 rounded-full bg-emerald-500/15 ring-2 ring-emerald-400/40 animate-pulse" />
        <svg
          className="absolute inset-0 m-auto h-[85%] w-[85%] animate-[spin_14s_linear_infinite] text-emerald-600"
          viewBox="0 0 100 100"
          fill="none"
        >
          <ellipse cx="50" cy="50" rx="40" ry="40" stroke="currentColor" strokeWidth="1.5" opacity="0.35" />
          <path d="M12 50h76M50 18v64" stroke="currentColor" strokeWidth="0.75" strokeDasharray="4 6" opacity="0.4" />
        </svg>
        <span className="absolute bottom-1 left-1/2 -translate-x-1/2 text-lg">✈️</span>
      </div>
    </div>
  );
}
