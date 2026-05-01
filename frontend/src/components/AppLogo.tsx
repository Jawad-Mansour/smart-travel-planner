export function AppLogo({ onClick, className = "" }: { onClick?: () => void; className?: string }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`group flex w-full items-center gap-3 rounded-2xl text-left transition hover:bg-emerald-50/80 focus:outline-none focus:ring-2 focus:ring-emerald-300/60 ${className}`}
      title="New chat"
    >
      <div className="relative shrink-0">
        <img
          src="/plane.svg"
          alt=""
          width={40}
          height={40}
          className="glowing-logo h-10 w-10 rounded-2xl object-contain"
        />
      </div>
      <div className="min-w-0 leading-tight">
        <div className="truncate text-sm font-bold tracking-tight text-slate-900">Smart Travel Planner</div>
        <div className="truncate text-xs text-slate-500">RAG · ML · Live data</div>
      </div>
    </button>
  );
}
