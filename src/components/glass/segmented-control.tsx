import * as React from "react"
import { cn } from "cn"
import { ToggleGroup as ToggleGroupPrimitive } from "radix-ui"

import { slidingIndicatorClass, useSlidingIndicator } from "@/hooks/use-sliding-indicator"

type SegmentedControlProps = Omit<
  React.ComponentProps<typeof ToggleGroupPrimitive.Root>,
  "type" | "value" | "defaultValue" | "onValueChange"
> & {
  value?: string
  defaultValue?: string
  onValueChange?: (value: string) => void
  size?: "sm" | "default" | "lg"
  /** `capsule` (Apple's default) or `rounded` (the control radius). */
  shape?: "capsule" | "rounded"
  /** Segments act as buttons: a press calls onValueChange and nothing stays selected. */
  momentary?: boolean
}

/**
 * The iOS segmented control: a soft track with one raised thumb that slides to the chosen
 * segment, concentric with the track. Always has a selection — tapping the chosen segment again
 * keeps it (unlike a toggle group). Drag the thumb to scrub across segments; while held it turns
 * into a liquid lens. `momentary` makes the segments plain actions.
 */
function SegmentedControl({
  className,
  value,
  defaultValue,
  onValueChange,
  size = "default",
  shape = "capsule",
  momentary = false,
  children,
  onPointerDown,
  onPointerMove,
  ref: forwardedRef,
  ...props
}: SegmentedControlProps) {
  const [inner, setInner] = React.useState(defaultValue ?? "")
  const current = value ?? inner
  const ref = useSlidingIndicator<HTMLDivElement>('[data-slot="segmented-control-item"][data-state="on"]', forwardedRef)
  const [dragging, setDragging] = React.useState(false)
  const press = React.useRef<{ x: number; id: number } | null>(null)
  const choose = (v: string) => {
    if (!v) return
    if (!momentary) setInner(v)
    onValueChange?.(v)
  }
  const endDrag = () => {
    press.current = null
    setDragging(false)
  }
  return (
    <ToggleGroupPrimitive.Root
      ref={ref}
      type="single"
      data-slot="segmented-control"
      data-size={size}
      data-shape={shape}
      data-dragging={dragging || undefined}
      value={momentary ? "" : current}
      onValueChange={choose}
      onPointerDown={(e) => {
        onPointerDown?.(e)
        // scrubbing starts on the selected segment, as on iOS
        const item = (e.target as HTMLElement).closest<HTMLElement>('[data-slot="segmented-control-item"]')
        if (!momentary && item?.dataset.state === "on") press.current = { x: e.clientX, id: e.pointerId }
      }}
      onPointerMove={(e) => {
        onPointerMove?.(e)
        const p = press.current
        if (!p || !e.buttons) return
        if (!dragging) {
          if (Math.abs(e.clientX - p.x) < 6) return
          // captured only once it's a drag, so a plain tap still clicks its segment
          e.currentTarget.setPointerCapture(p.id)
          setDragging(true)
        }
        const items = e.currentTarget.querySelectorAll<HTMLElement>('[data-slot="segmented-control-item"]:not(:disabled)')
        for (const item of items) {
          const r = item.getBoundingClientRect()
          if (e.clientX >= r.left && e.clientX < r.right && item.dataset.value && item.dataset.value !== current) choose(item.dataset.value)
        }
      }}
      onPointerUp={endDrag}
      onLostPointerCapture={endDrag}
      className={cn(
        // heights match Button's (h-control-sm / h-control / h-control-lg) so they line up in a toolbar;
        // the container radius + padding make the thumb concentric (rounded-concentric)
        "group/segmented relative inline-flex w-fit touch-pan-y items-stretch bg-fill [--glass-container-pad:0.125rem] p-(--glass-container-pad) data-[shape=capsule]:rounded-button data-[shape=capsule]:[--glass-container-radius:var(--glass-radius-button)] data-[shape=rounded]:rounded-control data-[shape=rounded]:[--glass-container-radius:var(--glass-radius-control)] data-[size=default]:h-control data-[size=lg]:h-control-lg data-[size=sm]:h-control-sm",
        className
      )}
      {...props}
    >
      <span
        data-slot="segmented-control-thumb"
        data-glass-motion
        aria-hidden
        // the thumb is a knob: solid at rest, a lens while scrubbing (data-dragging on the root)
        className={cn(slidingIndicatorClass, "glass-knob-lens rounded-concentric shadow-glass-knob [--glass-knob:var(--glass-thumb)]", momentary && "hidden")}
      />
      {children}
    </ToggleGroupPrimitive.Root>
  )
}

function SegmentedControlItem({ className, value, ...props }: React.ComponentProps<typeof ToggleGroupPrimitive.Item>) {
  return (
    <ToggleGroupPrimitive.Item
      data-slot="segmented-control-item"
      data-value={value}
      value={value}
      className={cn(
        // before the thumb is measured (server render, no JS) the chosen item draws the thumb itself
        "relative z-10 inline-flex min-w-0 flex-1 items-center justify-center gap-1.5 rounded-concentric px-3 text-sm font-medium whitespace-nowrap text-muted-foreground transition-colors duration-(--glass-duration) ease-out glass-focus-inset hover:text-foreground group-data-dragging/segmented:cursor-grabbing active:bg-fill group-data-dragging/segmented:active:bg-transparent disabled:pointer-events-none disabled:text-glass-label-tertiary data-[state=on]:font-semibold data-[state=on]:text-foreground not-in-data-indicator-ready:data-[state=on]:bg-glass-thumb not-in-data-indicator-ready:data-[state=on]:shadow-glass-knob group-data-[size=lg]/segmented:text-base group-data-[size=sm]/segmented:px-2 group-data-[size=sm]/segmented:text-xs [&_svg]:pointer-events-none [&_svg:not([class*='size-'])]:size-4",
        className
      )}
      {...props}
    />
  )
}

export { SegmentedControl, SegmentedControlItem }
