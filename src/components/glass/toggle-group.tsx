"use client"

import * as React from "react"
import { type VariantProps } from "class-variance-authority"
import { cn } from "cn"
import { ToggleGroup as ToggleGroupPrimitive } from "radix-ui"

import { toggleVariants } from "@/components/glass/toggle"

type ToggleGroupContextValue = VariantProps<typeof toggleVariants> & { spacing?: number }
const ToggleGroupContext = React.createContext<ToggleGroupContextValue>({ size: "default", variant: "default", spacing: 0 })

/**
 * shadcn's ToggleGroup: `type="single"` or `type="multiple"` toggles in a row. With `spacing={0}`
 * (default) they join into one control; any other spacing (in 0.25rem steps) separates them.
 */
function ToggleGroup({
  className,
  variant,
  size,
  spacing = 0,
  style,
  children,
  ...props
}: React.ComponentProps<typeof ToggleGroupPrimitive.Root> & VariantProps<typeof toggleVariants> & { spacing?: number }) {
  return (
    <ToggleGroupPrimitive.Root
      data-slot="toggle-group"
      data-variant={variant}
      data-size={size}
      data-spacing={spacing}
      style={{ "--gap": spacing, ...style } as React.CSSProperties}
      className={cn("group/toggle-group flex w-fit items-center gap-[calc(0.25rem*var(--gap))] rounded-control", className)}
      {...props}
    >
      <ToggleGroupContext.Provider value={{ variant, size, spacing }}>{children}</ToggleGroupContext.Provider>
    </ToggleGroupPrimitive.Root>
  )
}

function ToggleGroupItem({ className, children, variant, size, ...props }: React.ComponentProps<typeof ToggleGroupPrimitive.Item> & VariantProps<typeof toggleVariants>) {
  const context = React.useContext(ToggleGroupContext)
  return (
    <ToggleGroupPrimitive.Item
      data-slot="toggle-group-item"
      data-variant={context.variant || variant}
      data-size={context.size || size}
      data-spacing={context.spacing}
      className={cn(
        toggleVariants({ variant: context.variant || variant, size: context.size || size }),
        "w-auto min-w-0 shrink-0 focus:z-10 focus-visible:z-10",
        "data-[spacing=0]:rounded-none data-[spacing=0]:first:rounded-s-control data-[spacing=0]:last:rounded-e-control data-[spacing=0]:data-[variant=outline]:not-first:-ms-px",
        className
      )}
      {...props}
    >
      {children}
    </ToggleGroupPrimitive.Item>
  )
}

export { ToggleGroup, ToggleGroupItem }
