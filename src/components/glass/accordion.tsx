"use client"

import * as React from "react"
import { cn } from "cn"
import { Accordion as AccordionPrimitive } from "radix-ui"
import { ChevronRightIcon } from "lucide-react"

/**
 * shadcn's Accordion as Apple's disclosure rows: a trailing chevron that turns down when the row
 * opens, and the content sliding open beneath (Radix measures its height). `variant="grouped"`
 * sets the rows as an inset grouped list — a fill with hairlines between, like GroupedList.
 */
function Accordion({ className, variant = "default", ...props }: React.ComponentProps<typeof AccordionPrimitive.Root> & { variant?: "default" | "grouped" }) {
  return (
    <AccordionPrimitive.Root
      data-slot="accordion"
      data-variant={variant}
      className={cn("group/accordion w-full", variant === "grouped" && "overflow-hidden rounded-surface-sm bg-fill", className)}
      {...props}
    />
  )
}

function AccordionItem({ className, ...props }: React.ComponentProps<typeof AccordionPrimitive.Item>) {
  return (
    <AccordionPrimitive.Item
      data-slot="accordion-item"
      className={cn("border-b border-separator last:border-b-0 group-data-[variant=grouped]/accordion:ms-pad-sm", className)}
      {...props}
    />
  )
}

function AccordionTrigger({ className, children, ...props }: React.ComponentProps<typeof AccordionPrimitive.Trigger>) {
  return (
    <AccordionPrimitive.Header className="flex">
      <AccordionPrimitive.Trigger
        data-slot="accordion-trigger"
        className={cn(
          "group/accordion-trigger flex flex-1 items-center justify-between gap-4 rounded-control-sm py-[calc(0.875rem*var(--glass-density))] text-start text-sm font-medium transition-colors duration-(--glass-duration) glass-focus-inset hover:text-primary-ink disabled:pointer-events-none disabled:opacity-50 group-data-[variant=grouped]/accordion:pe-pad-sm",
          className
        )}
        {...props}
      >
        {children}
        <ChevronRightIcon
          aria-hidden
          className="pointer-events-none size-4 shrink-0 text-muted-foreground transition-transform duration-(--glass-duration) ease-glass rtl:-scale-x-100 group-data-[state=open]/accordion-trigger:rotate-90 rtl:group-data-[state=open]/accordion-trigger:-rotate-90 motion-reduce:transition-none"
        />
      </AccordionPrimitive.Trigger>
    </AccordionPrimitive.Header>
  )
}

function AccordionContent({ className, children, ...props }: React.ComponentProps<typeof AccordionPrimitive.Content>) {
  return (
    <AccordionPrimitive.Content
      data-slot="accordion-content"
      className="overflow-hidden text-sm text-muted-foreground data-[state=closed]:animate-[glass-collapsible-up_var(--glass-duration)_ease-out] data-[state=open]:animate-[glass-collapsible-down_calc(var(--glass-duration)*1.5)_ease-out] motion-reduce:animate-none! [--radix-collapsible-content-height:var(--radix-accordion-content-height)]"
      {...props}
    >
      <div className={cn("pt-0 pb-[calc(0.875rem*var(--glass-density))] group-data-[variant=grouped]/accordion:pe-pad-sm", className)}>{children}</div>
    </AccordionPrimitive.Content>
  )
}

export { Accordion, AccordionItem, AccordionTrigger, AccordionContent }
