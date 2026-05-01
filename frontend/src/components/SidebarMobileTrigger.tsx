import { Menu } from "lucide-react";

export function SidebarMobileTrigger({ onClick }: { onClick: () => void }) {
  return (
    <button type="button" className="btn-icon-muted shrink-0 lg:hidden" aria-label="Open menu" onClick={onClick}>
      <Menu className="h-5 w-5" />
    </button>
  );
}
