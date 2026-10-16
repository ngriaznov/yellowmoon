import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "cn"
import { Slot } from "radix-ui"

const badgeVariants = cva(
  "inline-flex h-control-2xs w-fit shrink-0 items-center justify-center gap-1 overflow-hidden rounded-badge px-pad-xs text-xs font-semibold whitespace-nowrap transition-colors duration-(--glass-duration) ease-out glass-focus [&>svg]:pointer-events-none [&>svg]:size-3",
  {
    variants: {
      variant: {
        default: "bg-primary text-primary-foreground",
        tinted: "bg-primary/16 text-primary-ink dark:bg-primary/22",
        glass: "glass [--glass-elevation:0_0_#0000] text-foreground",
        secondary: "bg-fill text-foreground",
        outline: "border border-glass-border text-foreground",
        destructive: "bg-destructive/14 text-destructive-ink",
      },
    },
    defaultVariants: { variant: "default" },
  }
)

function Badge({
  className,
  variant = "default",
  asChild = false,
  ...props
}: React.ComponentProps<"span"> & VariantProps<typeof badgeVariants> & { asChild?: boolean }) {
  const Comp = asChild ? Slot.Root : "span"
  return <Comp data-slot="badge" data-variant={variant} className={cn(badgeVariants({ variant }), className)} {...props} />
}

/**
 * A count or dot pinned to the corner of its parent — an icon Button, an Avatar, a DockItem.
 * Apple's notification badge: a red oval with a white number, for things that need attention.
 * The parent needs `relative` (Avatar's AvatarBadge and Dock items handle this themselves).
 * Counts over `max` read "99+"; `count={0}` hides it unless `showZero`; no count is a dot.
 * Give it a `label` — "3 unread" — when the number alone isn't clear; it's read with the parent.
 */
function BadgeCount({
  className,
  count,
  max = 99,
  showZero = false,
  label,
  ...props
}: React.ComponentProps<"span"> & { count?: number; max?: number; showZero?: boolean; label?: string }) {
  if (count !== undefined && count <= 0 && !showZero) return null
  const dot = count === undefined
  return (
    <span
      data-slot="badge-count"
      data-dot={dot || undefined}
      className={cn(
        "pointer-events-none absolute -top-1 -end-1 z-10 inline-flex h-[calc(1.125rem*var(--glass-density))] min-w-[calc(1.125rem*var(--glass-density))] items-center justify-center rounded-badge bg-destructive px-1 text-[0.6875rem] leading-none font-semibold text-destructive-foreground numeric-glass ring-2 ring-(--glass-tint) data-dot:top-0 data-dot:end-0 data-dot:size-2.5 data-dot:min-w-0 data-dot:p-0",
        className
      )}
      {...props}
    >
      {dot ? null : count > max ? `${max}+` : count}
      {label ? <span className="sr-only"> {label}</span> : null}
    </span>
  )
}

export { Badge, BadgeCount, badgeVariants }
