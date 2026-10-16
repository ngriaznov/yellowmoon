"use client"

import * as React from "react"

/**
 * Tracks the selected child of a group and exposes its box as CSS variables on
 * the group, so one indicator element can slide between items:
 *
 *   --indicator-x / --indicator-y / --indicator-w / --indicator-h / --indicator-r / --indicator-opacity
 *
 * `--indicator-r` is the selected item's corner radius, so a thumb can morph between items of
 * different shapes (`rounded-(--indicator-r)`, as GlassGroup does).
 *
 * Works for any primitive that marks the selection with an attribute (Radix's
 * `data-state="on"` / `"active"`), controlled or not. Follows the group's size,
 * every item's size (a badge count, a font swap), items added or removed, and a
 * change of `dir` (Radix writes it on the root).
 *
 * `data-indicator-ready` is set after the first measurement. Before that — on the
 * server, without JS, during hydration — the indicator is hidden and the selected
 * item draws the same background itself: give items a matching
 * `not-in-data-indicator-ready:data-[state=on]:bg-…` (see `SegmentedControlItem`).
 */
export function useSlidingIndicator<T extends HTMLElement>(selector: string, forwardedRef?: React.Ref<T>) {
  // our node lives in state and is handed to the caller's ref, so a user `ref` can't replace ours
  const [node, setNode] = React.useState<T | null>(null)
  React.useImperativeHandle(forwardedRef, () => node as T, [node])

  React.useLayoutEffect(() => {
    const root = node
    if (!root) return
    let frame = 0
    // observe the group and each item; a Set so re-observing never re-fires the observer
    const resizes = new ResizeObserver(() => update())
    const observed = new Set<Element>()
    const observe = (el: Element) => {
      if (observed.has(el)) return
      observed.add(el)
      resizes.observe(el)
    }
    const observeItems = () => {
      for (const el of observed) {
        if (el === root || root.contains(el)) continue
        resizes.unobserve(el)
        observed.delete(el)
      }
      observe(root)
      for (const child of Array.from(root.children)) observe(child)
    }
    const update = () => {
      const active = root.querySelector<HTMLElement>(selector)
      if (!active) {
        root.style.setProperty("--indicator-opacity", "0")
        return
      }
      observe(active)
      root.style.setProperty("--indicator-x", `${active.offsetLeft}px`)
      root.style.setProperty("--indicator-y", `${active.offsetTop}px`)
      root.style.setProperty("--indicator-w", `${active.offsetWidth}px`)
      root.style.setProperty("--indicator-h", `${active.offsetHeight}px`)
      root.style.setProperty("--indicator-r", getComputedStyle(active).borderTopLeftRadius || "0px")
      root.style.setProperty("--indicator-opacity", "1")
      if (!root.dataset.indicatorReady && !frame) {
        frame = requestAnimationFrame(() => {
          root.dataset.indicatorReady = "true"
        })
      }
    }

    observeItems()
    update()

    const mutations = new MutationObserver((records) => {
      if (records.some((r) => r.type === "childList")) observeItems()
      update()
    })
    mutations.observe(root, { subtree: true, childList: true, attributes: true, attributeFilter: ["data-state", "aria-selected", "aria-checked", "aria-pressed", "aria-current", "dir"] })
    return () => {
      cancelAnimationFrame(frame)
      mutations.disconnect()
      resizes.disconnect()
    }
  }, [selector, node])

  return setNode
}

/**
 * Classes for the sliding element itself.
 *
 * - `left-0`/`top-0` stay physical on purpose: the hook measures `offsetLeft`/`offsetTop`,
 *   which are physical in RTL too, so the indicator lands on the item in either direction.
 * - Hidden until the first measurement; the selected item draws its own background until then.
 * - Only the transform takes the theme's (possibly overshooting) spring; size and opacity use a
 *   plain ease-out, so the pill never grows past its track.
 * - `data-glass-motion` belongs on the element too, so reduced motion stills it.
 */
export const slidingIndicatorClass =
  "pointer-events-none absolute top-0 left-0 h-(--indicator-h) w-(--indicator-w) [transform:translate(var(--indicator-x),var(--indicator-y))] opacity-(--indicator-opacity) not-in-data-indicator-ready:opacity-0 in-data-indicator-ready:[transition:transform_calc(var(--glass-duration)*1.5)_var(--glass-ease),width_calc(var(--glass-duration)*1.5)_ease-out,height_calc(var(--glass-duration)*1.5)_ease-out,border-radius_calc(var(--glass-duration)*1.5)_ease-out,opacity_var(--glass-duration)_ease-out] motion-reduce:transition-none"
