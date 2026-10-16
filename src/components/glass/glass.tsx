import * as React from "react"
import { cn } from "cn"
import { Slot } from "radix-ui"

import { glassVariants, type GlassVariantProps } from "@/lib/glass-variants"

/**
 * A frosted pane: the base every glasscn surface is made from.
 * Blurs and saturates whatever is behind it, rimmed with a light edge.
 *
 * `variant="material"` (default) is the content layer — cards and panes, frosted by
 * `intensity`. `variant="liquid"` is the control layer — things floating over content
 * (toolbars, floating buttons). `variant="liquid-clear"` is for over photos and video only.
 * Never put liquid inside liquid: inside a liquid pane use fills (`bg-fill`).
 */
function Glass({
  className,
  variant,
  intensity,
  tint,
  elevation,
  asChild = false,
  ...props
}: React.ComponentProps<"div"> & GlassVariantProps & { asChild?: boolean }) {
  const Comp = asChild ? Slot.Root : "div"
  return (
    <Comp
      data-slot="glass"
      data-variant={variant ?? "material"}
      data-intensity={intensity ?? "default"}
      className={cn(glassVariants({ variant, intensity, tint, elevation }), "rounded-surface", className)}
      {...props}
    />
  )
}

export { Glass, glassVariants }
