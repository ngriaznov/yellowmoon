import type { ButtonHTMLAttributes } from "react";
import { Button as GlassButton } from "@/components/glass/button";
import { BUTTON_HANDLERS, memoStable } from "./stable";

export interface ResetLinkProps extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, "type" | "children"> {
  /** The text. Default "Reset"; the Look tab's Automatic card says "Re-run". */
  label?: string;
  /** `row` sits at the right end of a slider row, `card` in a card's header. */
  size?: "row" | "card";
}

/** A small accent text action: Reset at the end of a slider row, Reset or Re-run in a card header. */
function ResetLinkImpl({ label = "Reset", size = "row", className, ...rest }: ResetLinkProps) {
  return (
    <GlassButton {...rest} type="button" variant="link" size="xs" className={className} data-size={size}>
      {label}
    </GlassButton>
  );
}

/** Drawn again only when its data changes, not for a new `onClick` function (U-02). */
export const ResetLink = memoStable<ResetLinkProps>(ResetLinkImpl, BUTTON_HANDLERS);
