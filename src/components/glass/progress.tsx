import * as React from "react"
import { cn } from "cn"
import { Progress as ProgressPrimitive } from "radix-ui"

/**
 * A capsule bar whose fill runs from the second chart colour into the accent. `value` runs from 0 to `max` (100).
 * The fill is sized by `width`, so it grows from the inline start and runs right-to-left under `dir="rtl"`.
 */
function Progress({ className, value, max = 100, ...props }: React.ComponentProps<typeof ProgressPrimitive.Root>) {
  const pct = max > 0 ? Math.min(100, Math.max(0, ((value || 0) / max) * 100)) : 0
  return (
    <ProgressPrimitive.Root
      data-slot="progress"
      value={value}
      max={max}
      className={cn("relative h-2.5 w-full overflow-hidden rounded-button bg-fill", className)}
      {...props}
    >
      <ProgressPrimitive.Indicator
        data-slot="progress-indicator"
        data-glass-motion
        className="h-full rounded-button bg-linear-to-r from-chart-2 to-primary transition-[width] duration-[calc(var(--glass-duration)*3)] ease-out motion-reduce:transition-none rtl:bg-linear-to-l"
        style={{ width: `${pct}%` }}
      />
    </ProgressPrimitive.Root>
  )
}

export { Progress }
