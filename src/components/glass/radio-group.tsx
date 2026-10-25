"use client"

import * as React from "react"
import { cn } from "cn"
import { RadioGroup as RadioGroupPrimitive } from "radix-ui"

/**
 * shadcn's RadioGroup on glass: a soft circle that fills with the accent around a dot, with a
 * 44pt hit area on touch. `variant="card"` turns each item into a selectable tile — wrap the
 * item and its label in a `<label>` and the whole tile is the hit target. Arrow keys move the
 * choice (Radix); use 2–5 options, a Select beyond that.
 */
function RadioGroup({ className, variant = "default", ...props }: React.ComponentProps<typeof RadioGroupPrimitive.Root> & { variant?: "default" | "card" }) {
  return <RadioGroupPrimitive.Root data-slot="radio-group" data-variant={variant} className={cn("group/radio-group grid gap-3", className)} {...props} />
}

function RadioGroupItem({ className, ...props }: React.ComponentProps<typeof RadioGroupPrimitive.Item>) {
  return (
    <RadioGroupPrimitive.Item
      data-slot="radio-group-item"
      className={cn(
        "peer group/radio relative inline-flex size-[calc(1.125rem*var(--glass-density))] shrink-0 items-center justify-center rounded-full border border-muted-foreground bg-fill transition-[background-color,border-color,transform] duration-(--glass-duration) ease-glass glass-focus hit-area active:scale-(--glass-press-scale) disabled:cursor-not-allowed disabled:opacity-50 aria-invalid:border-destructive data-[state=checked]:border-transparent data-[state=checked]:bg-primary",
        className
      )}
      {...props}
    >
      <RadioGroupPrimitive.Indicator data-slot="radio-group-indicator" className="size-[40%] rounded-full bg-primary-foreground" />
    </RadioGroupPrimitive.Item>
  )
}

/**
 * The tile for `variant="card"`: a fill that takes an accent rim when its radio is checked.
 *
 *   <RadioGroupCard>
 *     <RadioGroupItem value="daily" id="daily" />
 *     <span>Daily</span>
 *   </RadioGroupCard>
 */
function RadioGroupCard({ className, ...props }: React.ComponentProps<"label">) {
  return (
    <label
      data-slot="radio-group-card"
      className={cn(
        "flex cursor-pointer items-start gap-3 rounded-control bg-fill p-[calc(0.875rem*var(--glass-density))] inset-ring inset-ring-glass-border transition-[box-shadow,background-color] duration-(--glass-duration) ease-out hover:bg-fill-strong has-data-[state=checked]:inset-ring-2 has-data-[state=checked]:inset-ring-primary has-disabled:cursor-not-allowed has-disabled:opacity-50",
        className
      )}
      {...props}
    />
  )
}

export { RadioGroup, RadioGroupItem, RadioGroupCard }
