import { useId } from "react";
import type { ButtonHTMLAttributes, ReactNode } from "react";
import { cx } from "./cx";
import { ChevronRightIcon } from "lucide-react";
import { Item } from "@/components/glass/item";
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
 * A card of the tool panel: a caps title with an optional tag and Reset or Re-run link on the
 * right, a 2 px rule under the title (use `Rows` for the rule and row separators), and a 3 px
 * rule under the card. A `<section>` named by its title.
 */
export function Card({ title, tag, action, density = "tool", last = false, tab, className, children }: CardProps) {
  const titleId = useId();
  return (
    <section
      aria-labelledby={titleId}
      className={cx("ym-card", className)}
      data-density={density === "tool" ? undefined : density}
      data-last={last || undefined}
      data-tab={tab}
    >
      <div className="ym-card__header">
        <h3 id={titleId} className="ym-card__title">
          {title}
        </h3>
        {tag}
        {action}
      </div>
      {children}
    </section>
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
    <div className={cx("ym-rows", className)} data-spaced={spaced || undefined}>
      {children}
    </div>
  );
}

/** A short paragraph under the rows of a card (the Camera card's explanation). */
export function RowNote({ children }: { children: ReactNode }) {
  return <p className="ym-row-note">{children}</p>;
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
    <div className="ym-info-row">
      <span className="ym-info-row__label">{label}</span>
      <span className="ym-info-row__value">{value}</span>
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
 * glasscn's pressable `Item` (a button) that opens the tab that owns the value.
 */
export function AutoRow({ label, value, isAuto, tabLabel, onOpen, className, ...rest }: AutoRowProps) {
  const tagText = isAuto ? "Auto" : "Edited";
  return (
    <Item asChild size="sm" className={cx("ym-auto-row", className)}>
      <button {...rest} type="button" aria-label={`${label} ${value}, ${tagText}. Open ${tabLabel}`} onClick={onOpen}>
        <span className="ym-auto-row__label">{label}</span>
        <span className="ym-auto-row__value">{value}</span>
        <Tag variant={isAuto ? "auto" : "edited"} />
        <ChevronRightIcon className="ym-auto-row__arrow" aria-hidden="true" />
      </button>
    </Item>
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
    <div className="ym-switch-row">
      <span className="ym-switch-row__label" aria-hidden="true">
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
    <div className={cx("ym-field", className)}>
      <span className="ym-field__label">{label}</span>
      {children}
    </div>
  );
}
