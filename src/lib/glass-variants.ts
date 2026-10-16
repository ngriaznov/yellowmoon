import { cva, type VariantProps } from "class-variance-authority"

/**
 * The glass surface every glasscn component is built from.
 *
 * Two layers, as in Apple's Liquid Glass:
 *   material      the content layer — `glass`, `glass-strong`, `glass-subtle` by `intensity`.
 *                 Cards, panes, tiles: anything in the flow of the page.
 *   liquid        the control layer — `glass-liquid`. Toolbars, the dock, floating buttons,
 *                 menus, popovers, sheets, dialogs. Inside it use fills, never more glass.
 *   liquid-clear  `glass-liquid-clear`: over photos and video only (it dims what's behind).
 * `intensity` applies to the material only.
 *
 * Any primitive can be overridden on any element, e.g. `[--glass-blur:8px]` or
 * `[--glass-radius-surface:0]`; three hooks replace whole layers:
 *   --glass-bg         the fill            e.g. `[--glass-bg:var(--primary)]`
 *   --glass-elevation  the outer shadow    e.g. `[--glass-elevation:0_0_#0000]`
 *   --glass-tinted     the frost colour    (set by `tint`; overrides --glass-tint)
 */
export const glassVariants = cva("relative", {
  variants: {
    variant: {
      material: "",
      liquid: "glass-liquid",
      "liquid-clear": "glass-liquid-clear",
    },
    intensity: {
      default: "",
      strong: "",
      subtle: "",
    },
    tint: {
      none: "",
      // a tint mixes a colour into the frost, keeping its opacity. It writes --glass-tinted, never
      // --glass-tint: a property that references itself is invalid and would make the pane clear.
      primary: "[--glass-tinted:color-mix(in_oklch,var(--primary)_30%,var(--glass-tint))]",
      destructive: "[--glass-tinted:color-mix(in_oklch,var(--destructive)_30%,var(--glass-tint))]",
    },
    elevation: {
      raised: "",
      flat: "[--glass-elevation:0_0_#0000]",
    },
  },
  compoundVariants: [
    { variant: "material", intensity: "default", class: "glass" },
    { variant: "material", intensity: "strong", class: "glass-strong" },
    { variant: "material", intensity: "subtle", class: "glass-subtle" },
  ],
  defaultVariants: {
    variant: "material",
    intensity: "default",
    tint: "none",
    elevation: "raised",
  },
})

export type GlassVariantProps = VariantProps<typeof glassVariants>
