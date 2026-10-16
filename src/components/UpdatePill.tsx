import type { UpdateState } from "../api/types";
import { updateCell } from "../updater";
import { Badge } from "@/components/glass/badge";
import { Button as GlassButton } from "@/components/glass/button";
import { cx } from "./cx";

export interface UpdatePillProps {
  /** From `useUpdater().state`. While idle the pill is not drawn at all. */
  state: UpdateState;
  /** An export is running: a ready update can't restart yet and says so on hover. */
  exportRunning?: boolean;
  /** The pill was pressed while it has an action: download when available, restart when ready. Use `useUpdater().activate`. */
  onActivate?: () => void;
  className?: string;
}

/**
 * The update cell of the header (`docs/11-ui.md` §Photos). Idle: nothing. "Update available" is a
 * button that downloads; "Downloading 42%" is a status; "Update ready" (a green square) is a
 * button that restarts, disabled with the tooltip "Restarts after the export finishes" while an
 * export runs; "Update failed" is a status with the reason on hover. The labels, actions and
 * tooltips come from `updateCell` in `updater.ts`.
 */
export function UpdatePill({ state, exportRunning = false, onActivate, className }: UpdatePillProps) {
  const cell = updateCell(state, exportRunning);
  if (!cell) return null;
  const content = (
    <>
      <span className="ym-update-pill__square" aria-hidden="true" />
      {cell.label}
    </>
  );
  const common = {
    "data-phase": cell.phase,
    title: cell.title ?? undefined,
  };
  return (
    <div className={cx("ym-update-pill", className)} data-phase={cell.phase}>
      {cell.action === null ? (
        <Badge variant="secondary" role="status" {...common} className="h-control-xs px-pad-sm">
          {content}
        </Badge>
      ) : (
        <GlassButton
          {...common}
          type="button"
          variant="glass"
          size="sm"
          aria-disabled={cell.disabled || undefined}
          onClick={() => {
            if (!cell.disabled) onActivate?.();
          }}
        >
          {content}
        </GlassButton>
      )}
    </div>
  );
}
