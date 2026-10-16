import * as React from "react"
import { cn } from "cn"
import { Select as SelectPrimitive } from "radix-ui"
import { CheckIcon, ChevronDownIcon, ChevronsUpDownIcon, ChevronUpIcon } from "lucide-react"

/**
 * shadcn's Select on glass. The trigger is a field (Input's height, radius and fill); the list is
 * a liquid-glass panel that materialises out of the trigger, the chosen row marked with a leading
 * check. `variant="pill"` on the trigger is Apple's pop-up button: a capsule showing the current
 * choice with up-down chevrons — for a short set of mutually exclusive options, no submenus.
 */
function Select({ ...props }: React.ComponentProps<typeof SelectPrimitive.Root>) {
  return <SelectPrimitive.Root data-slot="select" {...props} />
}

function SelectGroup({ ...props }: React.ComponentProps<typeof SelectPrimitive.Group>) {
  return <SelectPrimitive.Group data-slot="select-group" {...props} />
}

function SelectValue({ ...props }: React.ComponentProps<typeof SelectPrimitive.Value>) {
  return <SelectPrimitive.Value data-slot="select-value" {...props} />
}

function SelectTrigger({
  className,
  size = "default",
  variant = "default",
  children,
  ...props
}: React.ComponentProps<typeof SelectPrimitive.Trigger> & { size?: "sm" | "default" | "lg"; variant?: "default" | "pill" }) {
  return (
    <SelectPrimitive.Trigger
      data-slot="select-trigger"
      data-size={size}
      data-variant={variant}
      className={cn(
        "flex w-fit min-w-0 items-center justify-between gap-2 text-sm whitespace-nowrap text-foreground transition-[box-shadow,background-color] duration-(--glass-duration) ease-out glass-focus disabled:cursor-not-allowed disabled:opacity-50 aria-invalid:inset-ring-2 aria-invalid:inset-ring-destructive/70 data-placeholder:text-glass-label-tertiary [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4 *:data-[slot=select-value]:line-clamp-1 *:data-[slot=select-value]:flex *:data-[slot=select-value]:items-center *:data-[slot=select-value]:gap-2",
        "data-[size=default]:h-control data-[size=lg]:h-control-lg data-[size=sm]:h-control-sm",
        variant === "pill"
          ? "rounded-button bg-fill px-pad-sm font-medium hover:bg-fill-strong data-[size=sm]:px-pad-xs"
          : "rounded-control bg-fill px-pad-sm inset-ring inset-ring-glass-border forced-colors:border forced-colors:border-[CanvasText] focus-visible:bg-fill-strong data-[state=open]:bg-fill-strong",
        className
      )}
      {...props}
    >
      {children}
      <SelectPrimitive.Icon asChild>
        {variant === "pill" ? <ChevronsUpDownIcon className="size-3.5 text-muted-foreground" /> : <ChevronDownIcon className="text-muted-foreground" />}
      </SelectPrimitive.Icon>
    </SelectPrimitive.Trigger>
  )
}

/** The dropdown menu's liquid panel, morphing out of the select trigger. Rows inside are fills. */
const selectSurface =
  "glass-liquid relative z-50 min-w-(--radix-select-trigger-width) overflow-x-hidden overflow-y-auto overscroll-contain rounded-surface-sm text-popover-foreground glass-materialize glass-morph-origin [--glass-morph-w:var(--radix-select-trigger-width)] [--glass-morph-h:var(--radix-select-trigger-height)] [--glass-morph-origin:var(--radix-select-content-transform-origin)] max-h-(--radix-select-content-available-height)"

function SelectContent({ className, children, position = "popper", align = "start", ...props }: React.ComponentProps<typeof SelectPrimitive.Content>) {
  return (
    <SelectPrimitive.Portal>
      <SelectPrimitive.Content
        data-slot="select-content"
        data-glass-motion=""
        className={cn(selectSurface, position === "popper" && "data-[side=bottom]:translate-y-1.5 data-[side=top]:-translate-y-1.5", className)}
        position={position}
        align={align}
        {...props}
      >
        <SelectScrollUpButton />
        <SelectPrimitive.Viewport className={cn("p-1.5", position === "popper" && "h-(--radix-select-trigger-height) w-full min-w-(--radix-select-trigger-width) scroll-my-1.5")}>{children}</SelectPrimitive.Viewport>
        <SelectScrollDownButton />
      </SelectPrimitive.Content>
    </SelectPrimitive.Portal>
  )
}

function SelectLabel({ className, ...props }: React.ComponentProps<typeof SelectPrimitive.Label>) {
  return <SelectPrimitive.Label data-slot="select-label" className={cn("px-2.5 pt-2 pb-1 ps-8.5 text-xs font-semibold text-muted-foreground", className)} {...props} />
}

function SelectItem({ className, children, ...props }: React.ComponentProps<typeof SelectPrimitive.Item>) {
  return (
    <SelectPrimitive.Item
      data-slot="select-item"
      className={cn(
        "glass-focus-inset relative flex w-full cursor-default items-center gap-2 rounded-control-sm py-2 ps-8.5 pe-2.5 text-sm outline-hidden select-none pointer-coarse:py-3 focus:bg-fill-strong data-disabled:pointer-events-none data-disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4 [&_svg:not([class*='text-'])]:text-muted-foreground *:[span]:last:flex *:[span]:last:items-center *:[span]:last:gap-2",
        className
      )}
      {...props}
    >
      <span data-slot="select-item-indicator" className="absolute start-2.5 flex size-4 items-center justify-center">
        <SelectPrimitive.ItemIndicator>
          <CheckIcon className="text-primary!" />
        </SelectPrimitive.ItemIndicator>
      </span>
      <SelectPrimitive.ItemText>{children}</SelectPrimitive.ItemText>
    </SelectPrimitive.Item>
  )
}

function SelectSeparator({ className, ...props }: React.ComponentProps<typeof SelectPrimitive.Separator>) {
  return <SelectPrimitive.Separator data-slot="select-separator" className={cn("pointer-events-none -mx-1.5 my-1.5 h-px bg-separator", className)} {...props} />
}

function SelectScrollUpButton({ className, ...props }: React.ComponentProps<typeof SelectPrimitive.ScrollUpButton>) {
  return (
    <SelectPrimitive.ScrollUpButton data-slot="select-scroll-up-button" className={cn("flex cursor-default items-center justify-center py-1 text-muted-foreground", className)} {...props}>
      <ChevronUpIcon className="size-4" />
    </SelectPrimitive.ScrollUpButton>
  )
}

function SelectScrollDownButton({ className, ...props }: React.ComponentProps<typeof SelectPrimitive.ScrollDownButton>) {
  return (
    <SelectPrimitive.ScrollDownButton data-slot="select-scroll-down-button" className={cn("flex cursor-default items-center justify-center py-1 text-muted-foreground", className)} {...props}>
      <ChevronDownIcon className="size-4" />
    </SelectPrimitive.ScrollDownButton>
  )
}

export { Select, SelectContent, SelectGroup, SelectItem, SelectLabel, SelectScrollDownButton, SelectScrollUpButton, SelectSeparator, SelectTrigger, SelectValue }
