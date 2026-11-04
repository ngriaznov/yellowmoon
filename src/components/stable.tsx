import { memo, useRef } from "react";
import type { ComponentType } from "react";

/**
 * Makes a component draw only when its data changes (U-02). A screen draws again at every tick of a slider drag, and every `onClick={() => ...}` it
 * passes is a new function, so a plain `memo` never holds. `memoStable(Inner, ["onClick"])` hands `Inner` a function that never changes and that
 * calls the newest one the caller passed, and memoises it; the other props (strings, numbers, the same objects) decide as usual. A child
 * element passed as a prop or as children is a new object at every render of the caller, so a component that takes one still draws with
 * its caller: use this where the props are data.
 */
export function memoStable<P extends object>(Inner: ComponentType<P>, handlers: readonly string[], equal?: (prev: Readonly<P>, next: Readonly<P>) => boolean): ComponentType<P> {
  const Memo = memo(Inner, equal) as ComponentType<P>;
  function Stable(props: P) {
    const latest = useRef(props);
    latest.current = props;
    const proxies = useRef<Record<string, (...args: unknown[]) => unknown>>({});
    const bag = props as Record<string, unknown>;
    const next: Record<string, unknown> = { ...bag };
    for (const key of handlers) {
      if (typeof bag[key] !== "function") continue;
      proxies.current[key] ??= (...args: unknown[]) => (latest.current as Record<string, ((...a: unknown[]) => unknown) | undefined>)[key]?.(...args);
      next[key] = proxies.current[key];
    }
    return <Memo {...(next as P)} />;
  }
  Stable.displayName = `Stable(${Inner.displayName ?? Inner.name})`;
  return Stable;
}

/** The handlers of the buttons and toggles the screens pass as new functions at every render. */
export const BUTTON_HANDLERS = ["onClick", "onKeyDown", "onPointerDown", "onMouseDown", "onFocus", "onBlur", "onPressedChange", "onAuto", "onOff"] as const;

/**
 * A comparison of props for a component with a list of options (`Segmented`, `ChoiceList`): the options are rebuilt at each render of the
 * caller, so they count as the same when each holds the same fields; every other prop is compared as it is.
 */
export function sameWithOptions<P extends { options: ReadonlyArray<object> }>(prev: Readonly<P>, next: Readonly<P>): boolean {
  const a = prev as unknown as Record<string, unknown>;
  const b = next as unknown as Record<string, unknown>;
  const keys = Object.keys(a);
  if (keys.length !== Object.keys(b).length) return false;
  for (const key of keys) {
    if (key === "options") continue;
    if (!Object.is(a[key], b[key])) return false;
  }
  const x = prev.options as ReadonlyArray<Record<string, unknown>>;
  const y = next.options as ReadonlyArray<Record<string, unknown>>;
  return x.length === y.length && x.every((option, i) => Object.entries(option).every(([field, value]) => Object.is(value, y[i]?.[field])) && Object.keys(option).length === Object.keys(y[i] ?? {}).length);
}
