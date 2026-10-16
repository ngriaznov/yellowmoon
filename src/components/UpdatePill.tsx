import type { UpdateState } from "../api/types";
import { updateCell } from "../updater";
import { Badge } from "@/components/glass/badge";
import { Button } from "./Button";
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
  return (
    <div className={cx("ym-update-pill", className)} data-phase={cell.phase}>
      {cell.action === null ? (
        // A status has no pointer-events rule, so the browser's own title works on it.
        <Badge variant="secondary" role="status" data-phase={cell.phase} title={cell.title ?? undefined} className="h-control-xs px-pad-sm">
          {content}
        </Badge>
      ) : (
        // The app's Button, not glasscn's: a locked glass button takes no pointer events, so a native `title` on it would never show.
        // The Button puts its tooltip on the span around it, which still takes the hover ("Restarts after the export finishes").
        <Button
          variant="header"
          data-phase={cell.phase}
          disabled={cell.disabled}
          title={cell.title ?? undefined}
          onClick={() => onActivate?.()}
        >
          {content}
        </Button>
      )}
    </div>
  );
}
