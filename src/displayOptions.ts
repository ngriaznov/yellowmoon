import type { Api } from "./api";
import type { DisplayOptions } from "./api/types";
import { DISPLAY_ATTRIBUTES } from "./displayAttributes";

// The system's display options on the root element (`docs/28` A-P12, `docs/11` *Display options*). The shell reads Reduce Transparency, Reduce
// Motion and Increase Contrast from the system and the CSS answers them next to the media queries, because WebKit may not match every query.
// Each setting that is on sets its attribute on `<html>` and each that is off takes it away. Nothing is saved: the system is asked at start and
// tells the page when it changes.

/** Sets or removes the three attributes on `root` to match `options`. */
export function applyDisplayOptions(options: DisplayOptions, root: HTMLElement = document.documentElement): void {
  for (const key of Object.keys(DISPLAY_ATTRIBUTES) as Array<keyof DisplayOptions>) {
    const name = DISPLAY_ATTRIBUTES[key];
    if (options[key] === true) root.setAttribute(name, "");
    else root.removeAttribute(name);
  }
}

/**
 * Asks for the options now and follows the event. The answer of the command and an event can cross: the one that arrives last wins, but
 * an event that arrived before the answer is the newer news, so the answer is dropped then. Returns the function that stops listening.
 */
export function watchDisplayOptions(api: Pick<Api, "displayOptions" | "on">, root?: HTMLElement): () => void {
  // No document (a test that runs without a page): nothing to set.
  if (root === undefined && typeof document === "undefined") return () => {};
  const target = root ?? document.documentElement;
  let stopped = false;
  let heard = false;
  const stop = api.on("display_options", (options) => {
    if (stopped) return;
    heard = true;
    applyDisplayOptions(options, target);
  });
  api.displayOptions().then(
    (options) => {
      if (!stopped && !heard) applyDisplayOptions(options, target);
    },
    () => {
      // The command is not there (an older shell): the media queries alone answer.
    },
  );
  return () => {
    stopped = true;
    stop();
  };
}
