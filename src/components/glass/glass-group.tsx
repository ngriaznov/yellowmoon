"use client"

import * as React from "react"
import { cn } from "cn"
import { Slot } from "radix-ui"

import { slidingIndicatorClass, useSlidingIndicator } from "@/hooks/use-sliding-indicator"

const SELECTED = '[data-slot="glass-group-item"]:is([data-state="on"], [aria-pressed="true"], [aria-current="page"], [aria-selected="true"])'

/**
 * Apple's GlassEffectContainer for the web: neighbouring controls share ONE liquid backdrop
 * instead of each being its own glass (glass can't sample glass). Items inside are transparent;
 * an optional `indicator` slides a fill to the selected item, morphing to its size and corners.
 * Toolbar groups, button groups and segmented strips are built on it.
 */
function GlassGroup({
  className,
  orientation = "horizontal",
  indicator = false,
  children,
  ref: forwardedRef,
  ...props
}: React.ComponentProps<"div"> & { orientation?: "horizontal" | "vertical"; indicator?: boolean }) {
  const ref = useSlidingIndicator<HTMLDivElement>(SELECTED, forwardedRef)
  return (
    <div
      ref={ref}
      role="group"
      data-slot="glass-group"
      data-orientation={orientation}
      className={cn(
        "glass-liquid group/glass-group inline-flex w-fit items-center gap-0.5 rounded-button p-(--glass-container-pad) [--glass-container-pad:0.25rem] [--glass-container-radius:var(--glass-radius-button)] data-vertical:flex-col",
        className
      )}
      {...props}
    >
      {indicator && <span data-slot="glass-group-indicator" data-glass-motion aria-hidden className={cn(slidingIndicatorClass, "rounded-(--indicator-r) bg-fill-strong")} />}
      {children}
    </div>
  )
}

/**
 * A control inside a GlassGroup: transparent on the shared glass, a fill on hover, concentric
 * corners. `selected` marks it for the indicator (aria-pressed, unless `asChild` brings its own
 * semantics such as aria-current on a link).
 */
function GlassGroupItem({
  className,
  asChild = false,
  selected,
  ...props
}: React.ComponentProps<"button"> & { asChild?: boolean; selected?: boolean }) {
  const Comp = asChild ? Slot.Root : "button"
  return (
    <Comp
      data-slot="glass-group-item"
      data-state={selected === undefined ? undefined : selected ? "on" : "off"}
      aria-pressed={asChild || selected === undefined ? undefined : selected}
      {...(asChild ? {} : { type: "button" as const })}
      className={cn(
        "glass-press glass-focus-inset relative z-10 inline-flex h-control min-w-control shrink-0 items-center justify-center gap-1.5 rounded-concentric bg-transparent px-pad-sm text-sm font-medium whitespace-nowrap text-foreground transition-[background-color,color,scale] duration-(--glass-duration) ease-glass hover:bg-fill disabled:pointer-events-none disabled:text-glass-label-tertiary data-[state=on]:text-primary-ink not-in-data-indicator-ready:data-[state=on]:bg-fill-strong in-data-indicator-ready:hover:data-[state=on]:bg-transparent [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-[1.15rem]",
        className
      )}
      {...props}
    />
  )
}

export { GlassGroup, GlassGroupItem }
