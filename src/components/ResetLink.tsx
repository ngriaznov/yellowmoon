import type { ButtonHTMLAttributes } from "react";
import { Button as GlassButton } from "@/components/glass/button";
import { cx } from "./cx";

export interface ResetLinkProps extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, "type" | "children"> {
  /** The text. Default "Reset"; the Look tab's Automatic card says "Re-run". */
  label?: string;
  /** `row` sits at the right end of a slider row, `card` in a card's header. */
  size?: "row" | "card";
}

/** A small accent text action: Reset at the end of a slider row, Reset or Re-run in a card header. */
export function ResetLink({ label = "Reset", size = "row", className, ...rest }: ResetLinkProps) {
  return (
    <GlassButton
      {...rest}
      type="button"
      variant="link"
      size="xs"
      className={cx("h-auto! px-0! font-semibold", size === "card" ? "text-[0.8125rem]!" : "text-xs!", className)}
      data-size={size}
    >
      {label}
    </GlassButton>
  );
}
