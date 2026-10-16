"use client"

import * as React from "react"
import { cn } from "cn"
import { ScrollArea as ScrollAreaPrimitive } from "radix-ui"

/**
 * shadcn's ScrollArea on glass: capsule thumbs that read on any pane, thin at rest and wider
 * under the pointer. The viewport is focusable (so keyboard users can scroll it) and draws the
 * focus ring inside. `edges` fades content into the top and bottom edges — the in-pane
 * cousin of ScrollEdge, for lists inside a card.
 */
function ScrollArea({ className, children, edges = false, ...props }: React.ComponentProps<typeof ScrollAreaPrimitive.Root> & { edges?: boolean }) {
  return (
    <ScrollAreaPrimitive.Root data-slot="scroll-area" className={cn("relative overflow-hidden", className)} {...props}>
      <ScrollAreaPrimitive.Viewport
        data-slot="scroll-area-viewport"
        tabIndex={0}
        className={cn(
          "size-full rounded-[inherit] transition-[outline-color] duration-(--glass-duration) glass-focus-inset",
          edges && "[mask-image:linear-gradient(to_bottom,transparent,black_1rem,black_calc(100%-1rem),transparent)]"
        )}
      >
        {children}
      </ScrollAreaPrimitive.Viewport>
      <ScrollBar />
      <ScrollAreaPrimitive.Corner />
    </ScrollAreaPrimitive.Root>
  )
}

function ScrollBar({ className, orientation = "vertical", ...props }: React.ComponentProps<typeof ScrollAreaPrimitive.ScrollAreaScrollbar>) {
  return (
    <ScrollAreaPrimitive.ScrollAreaScrollbar
      data-slot="scroll-area-scrollbar"
      orientation={orientation}
      className={cn(
        "group/scrollbar flex touch-none p-0.5 transition-[width,height] duration-(--glass-duration) ease-out select-none",
        orientation === "vertical" && "h-full w-2 hover:w-3",
        orientation === "horizontal" && "h-2 flex-col hover:h-3",
        className
      )}
      {...props}
    >
      <ScrollAreaPrimitive.ScrollAreaThumb data-slot="scroll-area-thumb" className="relative flex-1 rounded-full bg-muted-foreground/50 group-hover/scrollbar:bg-muted-foreground/70" />
    </ScrollAreaPrimitive.ScrollAreaScrollbar>
  )
}

export { ScrollArea, ScrollBar }
