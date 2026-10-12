import type { ReactNode } from "react";
import { KeyHint, MoonMark, UpdatePill } from "../components";
import { useAppState } from "../state/context";
import { useUpdater } from "../updater";

export interface HeaderProps {
  /** Folder cell, large line: "Yellowmoon" or the folder name. */
  title: string;
  /** Folder cell, small caps line: "Drop ARW files to start", "2026-10-04 · 12 photos". */
  subline: string;
  /** `h1` makes the title the screen's heading (Photos: the folder name). Start has its own `h1`, so it keeps a plain span. */
  titleAs?: "span" | "h1";
  /** The buttons and status text between the folder cell and the right cell. */
  children?: ReactNode;
  /** The export cell. */
  export: ReactNode;
}

/**
 * The header of Start and Photos (`docs/11-ui.md` §Photos, header cells): the folder cell, an
 * area for actions, the update cell and the export cell. The update cell is hidden while idle.
 * The folder cell keeps 84 px of left padding for the traffic lights.
 */
export function Header({ title, subline, titleAs: Title = "span", children, export: exportCell }: HeaderProps) {
  // "Update ready" can't restart the app while an export is running (`docs/12` §In the app).
  const exportRunning = useAppState((s) => s.exportRun?.phase === "running");
  const update = useUpdater(exportRunning);
  return (
    <header className="ym-header">
      <div className="ym-brand">
        <Title className="ym-brand-title">
          <MoonMark />
          {/* One line: a long folder name ends in an ellipsis, and the whole name is the tooltip. */}
          <span className="ym-brand-name" title={title}>
            {title}
          </span>
        </Title>
        {/* Drawn in capitals by CSS, as in the mockup; the source text stays readable. */}
        <span className="ym-brand-sub">{subline}</span>
      </div>
      <nav aria-label="Actions" className="ym-header__nav">
        {children}
      </nav>
      <div className="ym-header__right">
        <UpdatePill state={update.state} exportRunning={exportRunning} onActivate={update.activate} />
        {exportCell}
      </div>
    </header>
  );
}

export interface ExportCellProps {
  /** "Export", or "Export 3" with three photos picked. */
  label: string;
  disabled?: boolean;
  /** Why the cell is disabled, as a tooltip. */
  title?: string;
  /** True while the Export panel is open. */
  current?: boolean;
  /** Progress of a running export, 0–100: a yellow strip along the foot of the cell. */
  progress?: number;
  onClick?: () => void;
}

/** The "Export ⌘E" cell at the right end of the header. */
export function ExportCell({ label, disabled = false, title, current = false, progress, onClick }: ExportCellProps) {
  return (
    <button
      type="button"
      className="ym-export"
      aria-disabled={disabled || undefined}
      aria-current={current || undefined}
      title={title}
      onClick={() => {
        if (!disabled) onClick?.();
      }}
    >
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="square" aria-hidden="true">
        <path d="M12 3v12" />
        <path d="M6.5 9.5L12 15l5.5-5.5" />
        <path d="M4 20h16" />
      </svg>
      {label}
      <KeyHint>⌘E</KeyHint>
      {progress !== undefined ? (
        <span className="ym-export__strip" aria-hidden="true">
          <span style={{ width: `${progress}%` }} />
        </span>
      ) : null}
    </button>
  );
}
