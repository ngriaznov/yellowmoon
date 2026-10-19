"use client"

import * as React from "react"
import { cn } from "cn"

/**
 * A filled text area that grows with its content (`field-sizing: content`), up to `maxRows`
 * (default 10) and then scrolls. Browsers without field-sizing get the same growth from a small
 * script. Padding and the minimum height follow density.
 */
function Textarea({ className, maxRows = 10, style, onInput, ...props }: React.ComponentProps<"textarea"> & { maxRows?: number }) {
  const ref = React.useRef<HTMLTextAreaElement>(null)
  const fit = React.useCallback(() => {
    const el = ref.current
    if (!el || typeof CSS === "undefined" || CSS.supports?.("field-sizing", "content")) return
    el.style.height = "auto"
    el.style.height = `${el.scrollHeight + (el.offsetHeight - el.clientHeight)}px`
  }, [])
  React.useLayoutEffect(fit, [fit, props.value])
  return (
    <textarea
      ref={ref}
      data-slot="textarea"
      onInput={(e) => {
        fit()
        onInput?.(e)
      }}
      style={{ maxHeight: `calc(${maxRows}lh + 2 * 0.75rem * var(--glass-density))`, ...style }}
      className={cn(
        "flex field-sizing-content min-h-[calc(6rem*var(--glass-density))] w-full overflow-y-auto rounded-control bg-fill inset-ring inset-ring-glass-border forced-colors:border forced-colors:border-[CanvasText] px-pad-sm py-[calc(0.75rem*var(--glass-density))] text-base text-foreground caret-primary transition-[box-shadow,background-color] duration-(--glass-duration) ease-out glass-focus placeholder:text-glass-label-tertiary selection:bg-primary/25 focus-visible:bg-fill-strong disabled:cursor-not-allowed disabled:bg-fill-subtle disabled:text-glass-label-tertiary aria-invalid:inset-ring-2 aria-invalid:inset-ring-destructive/70 md:text-sm",
        className
      )}
      {...props}
    />
  )
}

export { Textarea }
