import type { ButtonHTMLAttributes } from "react";
import { cx } from "./cx";

export interface ResetLinkProps extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, "type" | "children"> {
  /** Visible text: "Reset" (default) or "Re-run" on the Automatic card. */
  label?: string;
  /** `row` is 12 px (beside a slider), `card` is 13 px (in a card header). */
  size?: "row" | "card";
}

/**
 * Underlined 12 px 700 "Reset" (2 px underline, 3 px offset). Shown only when the row or card
 * differs from its default: the parent renders it conditionally. Give it an `aria-label`
 * that names what it resets ("Reset Exposure") when several share a page.
 */
export function ResetLink({ label = "Reset", size = "row", className, ...rest }: ResetLinkProps) {
  return (
    <button {...rest} type="button" className={cx("ym-reset-link", className)} data-size={size}>
      {label}
    </button>
  );
}
