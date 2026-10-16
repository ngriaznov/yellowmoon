import * as React from "react"
import { cn } from "cn"
import { Slot } from "radix-ui"

import { glassVariants, type GlassVariantProps } from "@/lib/glass-variants"

/**
 * shadcn's Card API on a glass pane. Same parts, same props — swap the import.
 * Padding is `--card-spacing` (the `card` / `card-sm` spacing tokens); the card
 * publishes its radius and padding as `--glass-container-radius` / `--glass-container-pad`
 * so tiles inside can take `rounded-concentric`.
 *
 * `material` picks the pane: `regular` (glass), `thick` (glass-strong — more
 * frost, for busy grounds) or `solid` (opaque `--card`, no blur — for dense
 * screens and long tables, where frost costs legibility and frames).
 * `asChild` makes the whole card a link or button (a `<a>` child) with a press state.
 */
const MATERIAL = { regular: "default", thick: "strong" } as const

function Card({
  className,
  size = "default",
  variant,
  material = "regular",
  intensity,
  tint,
  elevation,
  asChild = false,
  ...props
}: React.ComponentProps<"div"> &
  GlassVariantProps & {
    size?: "default" | "sm"
    material?: "regular" | "thick" | "solid"
    /** Render as the child element (a link) — the whole card is the target, with a press state. */
    asChild?: boolean
  }) {
  const Comp = asChild ? Slot.Root : "div"
  return (
    <Comp
      data-slot="card"
      data-size={size}
      data-material={material}
      className={cn(
        material === "solid"
          ? cn("relative border border-separator bg-card", elevation !== "flat" && "shadow-sm")
          : glassVariants({ variant, intensity: intensity ?? MATERIAL[material], tint, elevation }),
        asChild && "glass-focus glass-press cursor-pointer transition-[scale] duration-(--glass-duration) ease-glass",
        "group/card flex flex-col gap-(--card-spacing) overflow-hidden rounded-surface py-(--card-spacing) text-[calc(0.875rem*var(--glass-text-scale))] text-card-foreground [--card-spacing:--theme(--spacing-card)] [--glass-container-pad:var(--card-spacing)] [--glass-container-radius:var(--glass-radius-surface)] has-data-[slot=card-footer]:pb-0 data-[size=sm]:rounded-surface-sm data-[size=sm]:[--card-spacing:--theme(--spacing-card-sm)] data-[size=sm]:[--glass-container-radius:calc(var(--glass-radius-surface)*0.75)] data-[size=sm]:has-data-[slot=card-footer]:pb-0",
        className
      )}
      {...props}
    />
  )
}

/** A full-bleed image (or any media) at the top of the card; pass an `<img>` (or `next/image`) as the child. */
function CardImage({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="card-image"
      className={cn("-mt-(--card-spacing) overflow-hidden [&>img]:block [&>img]:aspect-video [&>img]:w-full [&>img]:object-cover", className)}
      {...props}
    />
  )
}

function CardHeader({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="card-header"
      className={cn(
        "group/card-header @container/card-header grid auto-rows-min items-start gap-1 px-(--card-spacing) has-data-[slot=card-action]:grid-cols-[1fr_auto] has-data-[slot=card-description]:grid-rows-[auto_auto]",
        className
      )}
      {...props}
    />
  )
}

function CardTitle({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="card-title"
      className={cn("type-glass-heading type-step-0 group-data-[size=sm]/card:text-[calc(0.875rem*var(--glass-text-scale))]", className)}
      {...props}
    />
  )
}

function CardDescription({ className, ...props }: React.ComponentProps<"div">) {
  return <div data-slot="card-description" className={cn("text-[calc(0.875rem*var(--glass-text-scale))] text-muted-foreground", className)} {...props} />
}

function CardAction({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="card-action"
      className={cn("col-start-2 row-span-2 row-start-1 self-start justify-self-end", className)}
      {...props}
    />
  )
}

function CardContent({ className, ...props }: React.ComponentProps<"div">) {
  return <div data-slot="card-content" className={cn("px-(--card-spacing)", className)} {...props} />
}

function CardFooter({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="card-footer"
      className={cn("flex items-center gap-2 border-t border-separator bg-fill-faint p-(--card-spacing)", className)}
      {...props}
    />
  )
}

export { Card, CardHeader, CardFooter, CardTitle, CardAction, CardDescription, CardContent, CardImage }
