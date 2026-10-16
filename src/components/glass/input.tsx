import * as React from "react"
import { cn } from "cn"

/**
 * A filled field on glass: soft fill, a hairline rim, accent caret, a focus outline; invalid draws
 * a destructive rim. `size` is the control height (`sm | default | lg`) — the HTML `size`
 * attribute is rarely wanted on a styled field, so it's taken over. Search fields drop the
 * browser's cancel glyph and number fields its spinners: put a clear button or a Stepper in an
 * InputGroup instead.
 */
function Input({ className, type, size = "default", ...props }: Omit<React.ComponentProps<"input">, "size"> & { size?: "sm" | "default" | "lg" }) {
  return (
    <input
      type={type}
      data-slot="input"
      data-size={size}
      className={cn(
        "h-control w-full min-w-0 rounded-control bg-fill inset-ring inset-ring-glass-border forced-colors:border forced-colors:border-[CanvasText] px-pad-sm text-base text-foreground caret-primary transition-[box-shadow,background-color] duration-(--glass-duration) ease-out glass-focus placeholder:text-glass-label-tertiary selection:bg-primary/25 file:inline-flex file:h-full file:border-0 file:bg-transparent file:text-sm file:font-medium focus-visible:bg-fill-strong disabled:cursor-not-allowed disabled:bg-fill-subtle disabled:text-glass-label-tertiary aria-invalid:inset-ring-2 aria-invalid:inset-ring-destructive/70 md:text-sm",
        "data-[size=sm]:h-control-sm data-[size=sm]:px-pad-xs data-[size=sm]:text-sm data-[size=lg]:h-control-lg data-[size=lg]:px-pad data-[size=lg]:md:text-base",
        "[&::-webkit-search-cancel-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none [&[type=number]]:[appearance:textfield]",
        className
      )}
      {...props}
    />
  )
}

export { Input }
