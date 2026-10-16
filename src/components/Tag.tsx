import type { ButtonHTMLAttributes, ReactNode } from "react";
import { Badge } from "@/components/glass/badge";
import { Toggle } from "@/components/glass/toggle";
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

/**
 * Which glasscn badge each tag variant uses: neutral fills for state (a rim for what is under way), the solid accent only for what was
 * changed by hand, red for what failed. No tinted accent badges: a translucent yellow chip on dark glass reads olive.
 */
const BADGE: Readonly<Record<TagVariant, "default" | "tinted" | "secondary" | "outline" | "destructive">> = {
  auto: "secondary",
  edited: "default",
  on: "secondary",
  "stand-in": "outline",
  reduced: "outline",
  default: "secondary",
  "cant-open": "destructive",
  waiting: "secondary",
  developing: "outline",
  ready: "secondary",
  skipped: "destructive",
  done: "secondary",
  active: "outline",
  failed: "destructive",
  match: "secondary",
  recommended: "outline",
};

/** A status label: glasscn's `Badge`, a small capsule. */
export function Tag({ variant, children, size = "md", className, title }: TagProps) {
  return (
    <Badge
      variant={BADGE[variant]}
      data-variant={variant}
      data-size={size === "md" ? undefined : size}
      title={title}
      className={cx(size === "sm" && "h-4 px-1.5 text-[0.625rem]", size === "lg" && "h-control-xs px-pad-xs", className)}
    >
      {children ?? TAG_LABELS[variant]}
    </Badge>
  );
}

export interface AutoToggleProps extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, "type" | "onClick" | "children"> {
  /** True while the card's values are automatic: the button is ink. */
  isAuto: boolean;
  /** Called when the button is pressed while the card is not automatic: back to Auto. */
  onAuto: () => void;
}

/**
 * The AUTO button in a card header (Exposure, White balance, Noise reduction): glasscn's `Toggle` as it comes (its own pressed style). On while the card
 * is automatic; once something was changed it is off, and pressing it restores Auto. Pressing it while automatic does nothing.
 */
export function AutoToggle({ isAuto, onAuto, className, ...rest }: AutoToggleProps) {
  return (
    <Toggle
      {...rest}
      size="sm"
      variant="outline"
      pressed={isAuto}
      className={cx("h-control-xs min-w-0 px-2.5 text-xs font-semibold", className)}
      onPressedChange={() => {
        if (!isAuto) onAuto();
      }}
    >
      Auto
    </Toggle>
  );
}
