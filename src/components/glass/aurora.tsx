"use client"

import * as React from "react"
import { cn } from "cn"

/**
 * The living ground glass sits on: three soft palette blobs drifting over the
 * base colour. Pure CSS (radial gradients + transforms) — no JS, stills for
 * `prefers-reduced-motion`. Themed by `--aurora-1..3`, `--aurora-base`,
 * `--aurora-opacity`, `--aurora-scale`, `--aurora-speed` and `--aurora-blur`;
 * the props below override those for one instance. Fixed behind the page by
 * default (`fixed inset-0 -z-10`); pass `className="absolute"` to fill a container.
 *
 * Cheap when idle: the blobs are GPU layers only while they move, the drift
 * pauses while the aurora is off screen or the page is scrolling, and `speed={0}` (or `animate={false}`)
 * holds it still.
 */
// One passive scroll listener for every aurora on the page: the drift pauses while scrolling,
// so the compositor spends scroll frames on the content, not on repainting blobs under glass.
const scrolling = new Set<HTMLElement>()
let scrollTimer: ReturnType<typeof setTimeout> | undefined
function onScroll() {
  if (scrollTimer === undefined) scrolling.forEach((el) => el.setAttribute("data-scrolling", ""))
  clearTimeout(scrollTimer)
  scrollTimer = setTimeout(() => {
    scrollTimer = undefined
    scrolling.forEach((el) => el.removeAttribute("data-scrolling"))
  }, 150)
}
function trackScroll(el: HTMLElement) {
  if (scrolling.size === 0) window.addEventListener("scroll", onScroll, { passive: true, capture: true })
  scrolling.add(el)
  return () => {
    scrolling.delete(el)
    el.removeAttribute("data-scrolling")
    if (scrolling.size === 0) window.removeEventListener("scroll", onScroll, { capture: true })
  }
}

function Aurora({
  ref,
  className,
  style,
  speed,
  intensity,
  scale,
  blur,
  animate = true,
  ...props
}: React.ComponentProps<"div"> & {
  /** Drift speed multiplier (--aurora-speed). */
  speed?: number
  /** Blob opacity 0–1 (--aurora-opacity). */
  intensity?: number
  /** Blob size multiplier (--aurora-scale). */
  scale?: number
  /** Extra blur on the blobs, px (--aurora-blur). */
  blur?: number
  animate?: boolean
}) {
  const local = React.useRef<HTMLDivElement | null>(null)
  const setRef = React.useCallback(
    (node: HTMLDivElement | null) => {
      local.current = node
      if (typeof ref === "function") ref(node)
      else if (ref) ref.current = node
    },
    [ref]
  )
  const moving = animate && !(speed !== undefined && speed <= 0)
  // pause the drift while off screen (an aurora per preview adds up) or while scrolling
  React.useEffect(() => {
    const el = local.current
    if (!el || !moving) return
    const untrack = trackScroll(el)
    if (typeof IntersectionObserver === "undefined") return untrack
    const observer = new IntersectionObserver((entries) => el.toggleAttribute("data-paused", !entries[entries.length - 1]?.isIntersecting))
    observer.observe(el)
    return () => {
      untrack()
      observer.disconnect()
      el.removeAttribute("data-paused")
    }
  }, [moving])

  const overrides: Record<string, string> = {}
  if (speed !== undefined && speed > 0) overrides["--aurora-speed"] = String(speed)
  if (intensity !== undefined) overrides["--aurora-opacity"] = `${intensity * 100}%`
  if (scale !== undefined) overrides["--aurora-scale"] = String(scale)
  if (blur !== undefined) overrides["--aurora-blur"] = `${blur}px`

  const blob = (i: 1 | 2 | 3, position: string, size: number, minPx: number, seconds: number) => (
    <div
      data-slot="aurora-blob"
      aria-hidden
      className={cn(
        "absolute aspect-square rounded-full",
        moving && "will-change-transform group-data-paused/aurora:will-change-auto group-data-paused/aurora:[animation-play-state:paused] group-data-scrolling/aurora:[animation-play-state:paused] motion-reduce:will-change-auto",
        position
      )}
      style={{
        width: `calc(${size}% * var(--aurora-scale, 1))`,
        minWidth: `calc(${minPx}px * var(--aurora-scale, 1))`,
        background: `radial-gradient(closest-side, var(--aurora-${i}), color-mix(in oklch, var(--aurora-${i}) 55%, transparent) 45%, transparent)`,
        opacity: "var(--aurora-opacity, 100%)",
        filter: "blur(var(--aurora-blur, 0px))",
        // longhands, so the paused class can set animation-play-state; max() guards a zero speed
        ...(moving
          ? {
              animationName: `glass-aurora-${i}`,
              animationDuration: `calc(${seconds}s / max(var(--aurora-speed, 1), 0.0001))`,
              animationTimingFunction: "ease-in-out",
              animationIterationCount: "infinite",
            }
          : null),
      }}
    />
  )
  return (
    <div
      ref={setRef}
      data-slot="aurora"
      aria-hidden
      className={cn("group/aurora pointer-events-none fixed inset-0 -z-10 overflow-hidden bg-aurora-base", className)}
      style={{ ...overrides, ...style }}
      {...props}
    >
      {blob(1, "-left-[25%] -top-[15%]", 125, 560, 34)}
      {blob(2, "left-[40%] top-[10%]", 105, 480, 42)}
      {blob(3, "-left-[10%] top-[55%]", 120, 540, 38)}
    </div>
  )
}

export { Aurora }
