"use client"

import * as React from "react"
import { cn } from "cn"
import { Popover as PopoverPrimitive } from "radix-ui"

function Popover({ ...props }: React.ComponentProps<typeof PopoverPrimitive.Root>) {
  return <PopoverPrimitive.Root data-slot="popover" {...props} />
}

function PopoverTrigger({ ...props }: React.ComponentProps<typeof PopoverPrimitive.Trigger>) {
  return <PopoverPrimitive.Trigger data-slot="popover-trigger" {...props} />
}

/**
 * A liquid-glass panel (the control layer), never taller than the space available — it scrolls.
 * It grows out of its trigger and materialises; reduced motion cross-fades. Inside it use fills, not glass.
 */
function PopoverContent({ className, align = "center", sideOffset = 8, ...props }: React.ComponentProps<typeof PopoverPrimitive.Content>) {
  return (
    <PopoverPrimitive.Portal>
      <PopoverPrimitive.Content
        data-slot="popover-content"
        data-glass-motion=""
        align={align}
        sideOffset={sideOffset}
        className={cn(
          "glass-liquid z-50 flex max-h-(--radix-popover-content-available-height) w-72 flex-col gap-3 overflow-y-auto overscroll-contain rounded-surface-sm p-4 text-sm text-popover-foreground outline-hidden glass-materialize glass-morph-origin [--glass-morph-w:var(--radix-popover-trigger-width)] [--glass-morph-h:var(--radix-popover-trigger-height)] [--glass-morph-origin:var(--radix-popover-content-transform-origin)]",
          className
        )}
        {...props}
      />
    </PopoverPrimitive.Portal>
  )
}

function PopoverAnchor({ ...props }: React.ComponentProps<typeof PopoverPrimitive.Anchor>) {
  return <PopoverPrimitive.Anchor data-slot="popover-anchor" {...props} />
}

export { Popover, PopoverAnchor, PopoverContent, PopoverTrigger }
