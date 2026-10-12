/** True inside a Tauri window; false in a plain browser, Vitest or Playwright. */
export function isTauri(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}
