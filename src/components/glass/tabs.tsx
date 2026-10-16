"use client"

import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "cn"
import { Tabs as TabsPrimitive } from "radix-ui"

import { slidingIndicatorClass, useSlidingIndicator } from "@/hooks/use-sliding-indicator"

function Tabs({ className, orientation = "horizontal", ...props }: React.ComponentProps<typeof TabsPrimitive.Root>) {
  return (
    <TabsPrimitive.Root
      data-slot="tabs"
      data-orientation={orientation}
      orientation={orientation}
      className={cn("group/tabs flex gap-3 data-horizontal:flex-col", className)}
      {...props}
    />
  )
}

const tabsListVariants = cva(
  "group/tabs-list relative inline-flex w-fit items-center justify-center text-muted-foreground group-data-vertical/tabs:h-fit group-data-vertical/tabs:flex-col group-data-vertical/tabs:items-stretch",
  {
    variants: {
      variant: {
        /** A glass capsule with a sliding pill. */
        default: "glass rounded-button [--glass-elevation:0_0_#0000] [--glass-container-pad:0.25rem] [--glass-container-radius:var(--glass-radius-button)] p-(--glass-container-pad)",
        /** No track — the pill slides over the page. */
        plain: "rounded-button [--glass-container-pad:0.25rem] [--glass-container-radius:var(--glass-radius-button)] p-(--glass-container-pad)",
        /** An underline that slides. */
        line: "gap-1 rounded-none",
      },
    },
    defaultVariants: { variant: "default" },
  }
)

function TabsList({ className, variant = "default", children, ref: forwardedRef, ...props }: React.ComponentProps<typeof TabsPrimitive.List> & VariantProps<typeof tabsListVariants>) {
  const ref = useSlidingIndicator<HTMLDivElement>('[data-slot="tabs-trigger"][data-state="active"]', forwardedRef)
  return (
    <TabsPrimitive.List ref={ref} data-slot="tabs-list" data-variant={variant} className={cn(tabsListVariants({ variant }), className)} {...props}>
      <span
        data-slot="tabs-indicator"
        data-glass-motion
        aria-hidden
        className={cn(
          slidingIndicatorClass,
          variant === "line"
            ? "top-auto bottom-0 h-0.5! [transform:translateX(var(--indicator-x))]! rounded-button bg-primary"
            : "rounded-concentric bg-fill-strong inset-shadow-2xs inset-shadow-glass-highlight"
        )}
      />
      {children}
    </TabsPrimitive.List>
  )
}

function TabsTrigger({ className, ...props }: React.ComponentProps<typeof TabsPrimitive.Trigger>) {
  return (
    <TabsPrimitive.Trigger
      data-slot="tabs-trigger"
      className={cn(
        // before the indicator is measured (server render, no JS) the active trigger draws it itself
        "relative z-10 inline-flex h-control-sm flex-1 items-center justify-center gap-1.5 rounded-concentric px-3.5 text-sm font-medium whitespace-nowrap text-muted-foreground transition-colors duration-(--glass-duration) ease-out glass-focus-inset group-data-vertical/tabs:w-full group-data-vertical/tabs:justify-start hover:text-foreground disabled:pointer-events-none disabled:text-glass-label-tertiary data-[state=active]:text-foreground not-in-data-indicator-ready:data-[state=active]:bg-fill-strong not-in-data-indicator-ready:data-[state=active]:inset-shadow-2xs not-in-data-indicator-ready:data-[state=active]:inset-shadow-glass-highlight group-data-[variant=line]/tabs-list:rounded-none group-data-[variant=line]/tabs-list:px-pad-xs group-data-[variant=line]/tabs-list:data-[state=active]:text-primary-ink group-data-[variant=line]/tabs-list:not-in-data-indicator-ready:data-[state=active]:bg-transparent group-data-[variant=line]/tabs-list:not-in-data-indicator-ready:data-[state=active]:inset-shadow-none group-data-[variant=line]/tabs-list:not-in-data-indicator-ready:data-[state=active]:shadow-[inset_0_-2px_0_var(--primary)] [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4",
        className
      )}
      {...props}
    />
  )
}

function TabsContent({ className, ...props }: React.ComponentProps<typeof TabsPrimitive.Content>) {
  return (
    <TabsPrimitive.Content
      data-slot="tabs-content"
      className={cn("flex-1 text-sm glass-focus data-[state=active]:animate-in data-[state=active]:fade-in-0 data-[state=active]:duration-(--glass-duration) data-[state=active]:slide-in-from-bottom-1 motion-reduce:animate-none", className)}
      {...props}
    />
  )
}

export { Tabs, TabsList, TabsTrigger, TabsContent, tabsListVariants }
