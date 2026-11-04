/**
 * Structural sharing for the settings mirror (U-02). The core's reply to a patch is a new object even when it says what the page already
 * showed (the page folds the patch in at once, `applyDirect`), and a new object redraws everything that reads it. `share` returns `prev`
 * itself when `next` holds the same values, and otherwise a copy of `next` in which every part that did not change is the part of `prev`,
 * so a control that reads one field sees the same field and does not draw.
 *
 * Only for plain JSON values (objects, arrays, numbers, strings, booleans, null): that is all the core sends.
 */
export function share<T>(prev: T, next: T): T {
  if (prev === next) return prev;
  if (typeof prev !== "object" || typeof next !== "object" || prev === null || next === null) return next;
  if (Array.isArray(prev) || Array.isArray(next)) {
    if (!Array.isArray(prev) || !Array.isArray(next)) return next;
    let same = prev.length === next.length;
    const out = next.map((item: unknown, i: number) => {
      const merged = i < prev.length ? share(prev[i] as unknown, item) : item;
      if (merged !== prev[i]) same = false;
      return merged;
    });
    return (same ? prev : out) as T;
  }
  const before = prev as Record<string, unknown>;
  const after = next as Record<string, unknown>;
  const keys = Object.keys(after);
  let same = keys.length === Object.keys(before).length;
  const out: Record<string, unknown> = {};
  for (const key of keys) {
    const merged = key in before ? share(before[key], after[key]) : after[key];
    if (merged !== before[key] || !(key in before)) same = false;
    out[key] = merged;
  }
  return (same ? prev : out) as T;
}
