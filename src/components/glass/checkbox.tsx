"use client"

import * as React from "react"
import { cn } from "cn"
import { Checkbox as CheckboxPrimitive } from "radix-ui"
import { CheckIcon, MinusIcon } from "lucide-react"

/**
 * shadcn's Checkbox on glass: a soft filled box that fills with the accent; `checked="indeterminate"`
 * shows a dash (a parent of a mixed group). `shape="circle"` is iOS's selection checkmark (lists,
 * photo pickers); `size` is `sm | default | lg`. The rim meets 3:1 against the pane and the hit
 * area grows to 44pt on touch.
 */
function Checkbox({
  className,
  shape = "square",
  size = "default",
  ...props
}: React.ComponentProps<typeof CheckboxPrimitive.Root> & { shape?: "square" | "circle"; size?: "sm" | "default" | "lg" }) {
  return (
    <CheckboxPrimitive.Root
      data-slot="checkbox"
      data-shape={shape}
      data-size={size}
      className={cn(
        "peer group/checkbox inline-flex shrink-0 items-center justify-center rounded-control-xs border border-muted-foreground bg-fill text-primary-foreground transition-[background-color,border-color,box-shadow,transform] duration-(--glass-duration) ease-glass glass-focus hit-area active:scale-(--glass-press-scale) disabled:cursor-not-allowed disabled:opacity-50 aria-invalid:border-destructive aria-invalid:ring-(length:--glass-ring-width) aria-invalid:ring-destructive/30 data-[state=checked]:border-transparent data-[state=checked]:bg-primary data-[state=indeterminate]:border-transparent data-[state=indeterminate]:bg-primary",
        "data-[size=default]:size-[calc(1.125rem*var(--glass-density))] data-[size=sm]:size-[calc(0.875rem*var(--glass-density))] data-[size=lg]:size-[calc(1.375rem*var(--glass-density))] data-[shape=circle]:rounded-full",
        className
      )}
      {...props}
    >
      <CheckboxPrimitive.Indicator data-slot="checkbox-indicator" className="flex items-center justify-center text-current [&_svg]:size-[85%] [&_svg]:stroke-3 group-data-[shape=circle]/checkbox:[&_svg]:size-[70%]">
        <CheckIcon className="hidden group-data-[state=checked]/checkbox:block" />
        <MinusIcon className="hidden group-data-[state=indeterminate]/checkbox:block" />
      </CheckboxPrimitive.Indicator>
    </CheckboxPrimitive.Root>
  )
}

export { Checkbox }
