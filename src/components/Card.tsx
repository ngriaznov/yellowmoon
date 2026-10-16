import { useId } from "react";
import type { ButtonHTMLAttributes, ReactNode } from "react";
import { cx } from "./cx";
import { Card as GlassCard } from "@/components/glass/card";
import { Tag } from "./Tag";
import type { TagVariant } from "./Tag";
import { Switch } from "./Switch";
import type { TabId } from "./tabs";

export interface CardProps {
  /** Section title, drawn in 13 px 800 caps ("Exposure", "High dynamic range"). */
  title: string;
  /** A `Tag` or `AutoToggle` at the right of the title. */
  tag?: ReactNode;
  /** The Reset or Re-run link at the far right. */
  action?: ReactNode;
  /**
   * `tool` (default) is a card of the tool panel: a 28 px header. `option` is a section of the
   * Export panel: the header takes its natural height and the card is a little shorter.
   */
  density?: "tool" | "option";
  /** Drops the 3 px rule under the last card of a panel. Default false. */
  last?: boolean;
  /** Tab colour for everything inside; inherited from an ancestor's `data-tab` when omitted. */
  tab?: TabId;
  className?: string;
  children?: ReactNode;
}

/**
 * A card of the tool panel: glasscn's `Card` (a frosted pane), with a title, an optional tag and a Reset or Re-run link on the right.
 * `Rows` inside it draws hairlines between rows. A `<section>`-like region named by its title.
 */
export function Card({ title, tag, action, density = "tool", last = false, tab, className, children }: CardProps) {
  const titleId = useId();
  return (
    <GlassCard
      role="region"
      aria-labelledby={titleId}
      size="sm"
      className={cx("gap-2 py-3.5", className)}
      data-density={density === "tool" ? undefined : density}
      data-last={last || undefined}
      data-tab={tab}
    >
      <div className="flex min-h-7 items-center gap-2.5 px-(--card-spacing)">
        <h3 id={titleId} className="type-glass-heading mr-auto text-[0.9375rem] leading-none">
          {title}
        </h3>
        {tag}
        {action}
      </div>
      <div className="flex flex-col px-(--card-spacing)">{children}</div>
    </GlassCard>
  );
}

export interface RowsProps {
  children?: ReactNode;
  /** Adds 8 px below the last rule, as under RAW › Decode's switches. */
  spaced?: boolean;
  className?: string;
}

/** A list of rows with a 2 px rule above and each row ruled below (the Automatic and RAW cards). */
export function Rows({ children, spaced = false, className }: RowsProps) {
  return (
    <div className={cx("flex flex-col divide-y divide-separator border-t border-separator", spaced && "mb-2", className)} data-spaced={spaced || undefined}>
      {children}
    </div>
  );
}

/** A short paragraph under the rows of a card (the Camera card's explanation). */
export function RowNote({ children }: { children: ReactNode }) {
  return <p className="ym-row-note mt-2 text-[0.8125rem] leading-snug text-muted-foreground">{children}</p>;
}

export interface InfoRowProps {
  label: string;
  value: ReactNode;
  /** A tag at the right end (STAND-IN, REDUCED). */
  tag?: TagVariant;
  /** Overrides the tag text. */
  tagLabel?: string;
}

/** A label, a value and an optional tag: the Camera card's "Detected" and "Profile" rows. */
export function InfoRow({ label, value, tag, tagLabel }: InfoRowProps) {
  return (
    <div className="flex min-h-10 items-center gap-3 text-sm">
      <span className="w-[92px] shrink-0 font-semibold">{label}</span>
      <span className="ym-info-row__value min-w-0 flex-1 truncate">{value}</span>
      {tag ? <Tag variant={tag}>{tagLabel}</Tag> : null}
    </div>
  );
}

export interface AutoRowProps extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, "type" | "children"> {
  /** "Exposure", "White balance", "Noise", "Camera". */
  label: string;
  /** "+0.30 EV", "5450 K · tint 0". */
  value: string;
  /** True while the value is automatic: an AUTO tag, otherwise EDITED. */
  isAuto: boolean;
  /** Name of the tab the arrow opens ("Light"), for the accessible name. */
  tabLabel: string;
  onOpen: () => void;
}

/**
 * A row of the Automatic card: label, value, AUTO or EDITED tag and an arrow. The whole row is
 * a button that opens the tab that owns the value.
 */
export function AutoRow({ label, value, isAuto, tabLabel, onOpen, className, ...rest }: AutoRowProps) {
  const tagText = isAuto ? "Auto" : "Edited";
  return (
    <button
      {...rest}
      type="button"
      className={cx(
        "glass-focus-inset group/row flex min-h-10 w-full items-center gap-3 rounded-control-sm px-0 text-left text-sm transition-colors hover:bg-fill",
        className,
      )}
      aria-label={`${label} ${value}, ${tagText}. Open ${tabLabel}`}
      onClick={onOpen}
    >
      <span className="w-[104px] shrink-0 font-semibold whitespace-nowrap">{label}</span>
      <span className="min-w-0 flex-1 truncate text-muted-foreground">{value}</span>
      <Tag variant={isAuto ? "auto" : "edited"} />
      <span className="w-4 text-center text-muted-foreground transition-transform group-hover/row:translate-x-0.5" aria-hidden="true">
        ›
      </span>
    </button>
  );
}

export interface SwitchRowProps {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
}

/** A 44 px row with a label and a `Switch` (RAW › Decode). */
export function SwitchRow({ label, checked, onChange, disabled }: SwitchRowProps) {
  return (
    <div className="flex min-h-11 items-center justify-between gap-3">
      <span className="text-sm font-semibold" aria-hidden="true">
        {label}
      </span>
      <Switch label={label} checked={checked} onChange={onChange} disabled={disabled} />
    </div>
  );
}

export interface FieldProps {
  /** The label column text, 13 px 800 ("Apply to", "Bit depth"). */
  label: string;
  children: ReactNode;
  className?: string;
}

/** A 92 px label column and a control, as "Apply to" under the Intensity slider. */
export function Field({ label, children, className }: FieldProps) {
  return (
    <div className={cx("grid min-h-10 grid-cols-[92px_minmax(0,1fr)] items-center gap-x-3 py-1", className)}>
      <span className="text-[0.8125rem] font-semibold leading-tight">{label}</span>
      {children}
    </div>
  );
}
