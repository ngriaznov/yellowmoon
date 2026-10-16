"use client"

import * as React from "react"
import { cn } from "cn"
import { Switch as SwitchPrimitive } from "radix-ui"

/**
 * The iOS 27 switch: a wide capsule that fills with the accent and a capsule knob that springs
 * across. Pressed, the knob turns into a liquid lens and stretches toward where it's going; drag
 * it to either side to choose (a drag back to where it started cancels the toggle).
 *
 * The knob moves by `margin-inline-start`, so it slides toward the inline end in either
 * direction without `rtl:` selectors (which misfire inside Radix roots that write `dir="ltr"`).
 */
function Switch({
  className,
  size = "default",
  onPointerDown,
  onPointerMove,
  onClick,
  ...props
}: React.ComponentProps<typeof SwitchPrimitive.Root> & { size?: "sm" | "default" }) {
  // where the pointer went down, and which side it was dragged to (null: not dragged)
  const drag = React.useRef<{ x: number; wantOn: boolean | null } | null>(null)
  const stretch = (el: HTMLElement, on: boolean) => {
    // grow from the side it leaves, toward the side it's heading
    const rtl = getComputedStyle(el).direction === "rtl"
    el.style.setProperty("--glass-knob-stretch-x", "1")
    const thumb = el.querySelector<HTMLElement>("[data-slot=switch-thumb]")
    if (thumb) thumb.style.transformOrigin = on !== rtl ? "left center" : "right center"
  }
  return (
    <SwitchPrimitive.Root
      data-slot="switch"
      data-size={size}
      onPointerDown={(e) => {
        onPointerDown?.(e)
        drag.current = { x: e.clientX, wantOn: null }
        stretch(e.currentTarget, e.currentTarget.dataset.state !== "checked")
      }}
      onPointerMove={(e) => {
        onPointerMove?.(e)
        const d = drag.current
        if (!d || !e.buttons) return
        const dx = e.clientX - d.x
        if (Math.abs(dx) < 6) return
        const rtl = getComputedStyle(e.currentTarget).direction === "rtl"
        d.wantOn = dx > 0 !== rtl
        stretch(e.currentTarget, d.wantOn)
      }}
      onClick={(e) => {
        onClick?.(e)
        const d = drag.current
        drag.current = null
        // dragged to the side it's already on: keep it (the click would toggle it)
        if (e.detail > 0 && d?.wantOn != null && d.wantOn === (e.currentTarget.dataset.state === "checked")) e.preventDefault()
      }}
      className={cn(
        "peer group group/switch relative inline-flex shrink-0 touch-pan-y items-center rounded-button p-0.5 transition-colors duration-(--glass-duration) ease-out glass-focus hit-area data-[size=default]:h-[calc(1.75rem*var(--glass-density))] data-[size=default]:w-[calc(3.9375rem*var(--glass-density))] data-[size=sm]:h-[calc(1.25rem*var(--glass-density))] data-[size=sm]:w-[calc(2.75rem*var(--glass-density))] data-checked:bg-primary data-unchecked:bg-fill-strong data-disabled:cursor-not-allowed data-disabled:opacity-50",
        className
      )}
      {...props}
    >
      <SwitchPrimitive.Thumb
        data-slot="switch-thumb"
        data-glass-motion
        className="glass-knob-lens pointer-events-none block rounded-button shadow-glass-knob [transition:margin_calc(var(--glass-duration)*1.5)_var(--glass-ease),scale_calc(var(--glass-duration)*1.5)_var(--glass-ease),background-color_var(--glass-duration)_ease-out,box-shadow_var(--glass-duration)_ease-out] group-data-[size=default]/switch:h-[calc(1.5rem*var(--glass-density))] group-data-[size=default]/switch:w-[calc(2.375rem*var(--glass-density))] group-data-[size=sm]/switch:h-[calc(1rem*var(--glass-density))] group-data-[size=sm]/switch:w-[calc(1.625rem*var(--glass-density))] data-unchecked:ms-0 group-data-[size=default]/switch:data-checked:ms-[calc(1.3125rem*var(--glass-density))] group-data-[size=sm]/switch:data-checked:ms-[calc(0.875rem*var(--glass-density))] motion-reduce:transition-none"
      />
    </SwitchPrimitive.Root>
  )
}

export { Switch }
