import type { ButtonHTMLAttributes, ReactNode } from "react";
import { cx } from "./cx";

/**
 * Tag variants (`docs/11-ui.md` §Components).
 *
 * | Variant | Look | Used for |
 * |---|---|---|
 * | `auto` | paper | an automatic value (AUTO) |
 * | `edited` | ink | a value changed by hand (EDITED) |
 * | `on` | tab colour | a feature that is on |
 * | `stand-in` | paper | a camera profile borrowed from another body |
 * | `reduced` | peach | a reduced decode path (lossless M and S) |
 * | `default` | brand yellow | the default look |
 * | `cant-open` | ink | a file that could not be opened |
 * | `waiting` | paper | queued, not started |
 * | `developing` | brand yellow | being developed |
 * | `ready` | ink | developed |
 * | `skipped` | orange | skipped, with a reason beside it |
 *
 * Added for the export queue and the wizard: `done` (ink), `active` (yellow, "45%"),
 * `failed` (peach), `match` (green) and `recommended` (yellow).
 */
export type TagVariant =
  | "auto"
  | "edited"
  | "on"
  | "stand-in"
  | "reduced"
  | "default"
  | "cant-open"
  | "waiting"
  | "developing"
  | "ready"
  | "skipped"
  | "done"
  | "active"
  | "failed"
  | "match"
  | "recommended";

/** The caps text each variant shows unless `children` says otherwise. */
export const TAG_LABELS: Readonly<Record<TagVariant, string>> = {
  auto: "Auto",
  edited: "Edited",
  on: "On",
  "stand-in": "Stand-in",
  reduced: "Reduced",
  default: "Default",
  "cant-open": "Can't open",
  waiting: "Waiting",
  developing: "Developing",
  ready: "Ready",
  skipped: "Skipped",
  done: "Done",
  active: "Active",
  failed: "Failed",
  match: "Match",
  recommended: "Recommended",
};

export interface TagProps {
  variant: TagVariant;
  /** Text, drawn in capitals. Defaults to the variant's own label. */
  children?: ReactNode;
  /** `sm` is the 14 px tag laid over a filmstrip thumbnail; `lg` the 26 px tag of the wizard's package and recognition rows. */
  size?: "md" | "sm" | "lg";
  className?: string;
  title?: string;
}

/** A 22 px boxed status label: 2 px border, 10.5 px 800 caps, 0.05em tracking. */
export function Tag({ variant, children, size = "md", className, title }: TagProps) {
  return (
    <span className={cx("ym-tag", className)} data-variant={variant} data-size={size === "md" ? undefined : size} title={title}>
      {children ?? TAG_LABELS[variant]}
    </span>
  );
}

export interface AutoToggleProps extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, "type" | "onClick" | "children"> {
  /** True while the card's values are automatic: the button is ink. */
  isAuto: boolean;
  /** Called when the button is pressed while the card is not automatic: back to Auto. */
  onAuto: () => void;
}

/**
 * The AUTO button in a card header (Exposure, White balance, Noise reduction). It is a 26 px
 * toggle: ink while the card is automatic, paper with an outline once something was changed;
 * pressing it then restores Auto. Pressing it while automatic does nothing.
 */
export function AutoToggle({ isAuto, onAuto, className, ...rest }: AutoToggleProps) {
  return (
    <button
      {...rest}
      type="button"
      className={cx("ym-auto-toggle", className)}
      aria-pressed={isAuto}
      onClick={() => {
        if (!isAuto) onAuto();
      }}
    >
      Auto
    </button>
  );
}
