/**
 * A handle for the end-to-end tests and for people poking at the app in a browser console: outside Tauri,
 * `window.__yellowmoon` holds the mock API the app runs on and the updater. Tests reach the very objects the
 * app uses through it; importing `/src/api.mock.ts` from a test would give a second copy whenever Vite has
 * tacked an HMR timestamp onto the app's own import.
 */
import type { MockApi } from "./api.mock";
import type { Updater } from "./updater";

/** What `window.__yellowmoon` holds. */
export interface YellowmoonHooks {
  mockApi: MockApi;
  updater: Updater;
}

declare global {
  interface Window {
    __yellowmoon?: YellowmoonHooks;
  }
}

/** Publishes the hooks on `window`. */
export function installTestHooks(hooks: YellowmoonHooks): void {
  window.__yellowmoon = hooks;
}
