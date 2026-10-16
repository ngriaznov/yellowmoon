import * as React from "react"
import { cn } from "cn"

/**
 * An activity indicator. `variant="ring"` (default) is an arc; `variant="ticks"` is iOS's radial
 * spokes. It's a `role="status"` with a label ("Loading" unless you pass one); pass `aria-hidden`
 * where something else announces the busy state (Button `loading` does). Under reduced motion the
 * ring holds a still frame and the ticks step slowly instead of spinning.
 */
function Spinner({
  className,
  variant = "ring",
  "aria-label": label = "Loading",
  ...props
}: React.ComponentProps<"svg"> & { variant?: "ring" | "ticks" }) {
  const hidden = props["aria-hidden"] === true || props["aria-hidden"] === "true"
  return (
    <svg
      data-slot="spinner"
      data-variant={variant}
      role={hidden ? undefined : "status"}
      aria-label={hidden ? undefined : label}
      viewBox="0 0 24 24"
      fill="none"
      className={cn(
        "size-4 shrink-0",
        variant === "ring" ? "animate-spin motion-reduce:animate-none" : "animate-[spin_0.9s_steps(8)_infinite] motion-reduce:animate-[spin_2.4s_steps(8)_infinite]",
        className
      )}
      {...props}
    >
      {variant === "ring" ? (
        <>
          <circle cx="12" cy="12" r="9" stroke="currentColor" strokeOpacity="0.25" strokeWidth="3" />
          <path d="M21 12a9 9 0 0 0-9-9" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
        </>
      ) : (
        Array.from({ length: 8 }, (_, i) => (
          <line key={i} x1="12" y1="3" x2="12" y2="7.5" stroke="currentColor" strokeWidth="2.25" strokeLinecap="round" strokeOpacity={0.25 + (0.75 * (i + 1)) / 8} transform={`rotate(${i * 45} 12 12)`} />
        ))
      )}
    </svg>
  )
}

export { Spinner }
