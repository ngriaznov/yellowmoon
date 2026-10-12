/**
 * Auto-update, per `docs/12-distribution.md` §In the app. Phases: idle, available,
 * downloading, ready, error (the `UpdateState` of `api/types.ts`).
 *
 *  - A quiet check 10 s after launch and then every 6 hours while "Check for updates
 *    automatically" is on. Failures stay silent: an unreachable endpoint never bothers the user.
 *  - "Check for Updates…" checks now and reports the result.
 *  - Updates come from `@tauri-apps/plugin-updater` (`check`, `downloadAndInstall`) and the app
 *    restarts through `@tauri-apps/plugin-process` (`relaunch`).
 *  - An update never interrupts an export: "Update ready" can't restart while one is running
 *    (`restartAvailability`).
 *
 * `createUpdater` holds the state machine and takes its collaborators as arguments so tests can
 * fake them; `updater` is the instance the app uses, wired to the real plugins. `useUpdater` is
 * the React hook, built on `useSyncExternalStore`.
 */
import { relaunch } from "@tauri-apps/plugin-process";
import { check, type DownloadEvent, type Update } from "@tauri-apps/plugin-updater";
import { useCallback, useSyncExternalStore } from "react";
import type { UpdateState } from "./api/types";
import { isTauri } from "./env";

/** First quiet check after launch. */
export const FIRST_CHECK_DELAY_MS = 10_000;
/** Interval between quiet checks while the app runs. */
export const CHECK_INTERVAL_MS = 6 * 60 * 60 * 1000;

/** The part of the plugin's `Update` that the state machine uses. */
export type UpdateHandle = Pick<Update, "version" | "downloadAndInstall">;

/** Everything the state machine touches outside itself. */
export interface UpdaterDeps {
  /** Asks the update endpoint; resolves `null` when this version is current. */
  check: () => Promise<UpdateHandle | null>;
  /** Restarts the app into the new version. */
  relaunch: () => Promise<void>;
  /** Whether "Check for updates automatically" is on; read before every quiet check. */
  autoCheck?: () => boolean | Promise<boolean>;
}

/** What a manual "Check for Updates…" found. */
export type CheckResult =
  | { kind: "available"; version: string }
  | { kind: "up_to_date" }
  | { kind: "error"; message: string };

/** The updater's store and actions. */
export interface Updater {
  getState(): UpdateState;
  /** For `useSyncExternalStore`: the listener runs after every state change. */
  subscribe(listener: () => void): () => void;
  /** The scheduled check: only from idle or error, and silent when it fails. */
  quietCheck(): Promise<void>;
  /** "Check for Updates…": always asks, and reports what happened. */
  checkNow(): Promise<CheckResult>;
  /** Downloads and installs the pending update (idle → downloading → ready, or error). */
  download(): Promise<void>;
  /** Restarts into the update. Does nothing and resolves false unless `restartAvailability` allows it. */
  restart(exportRunning: boolean): Promise<boolean>;
  /** Starts the schedule (10 s, then every 6 h) and returns a function that stops it. */
  start(): () => void;
  /** Takes over a state sent by the core (the `update_state` event, for example from the app menu). */
  applyExternal(state: UpdateState): void;
}

/** Whether "Update ready" can restart the app now, and why not. */
export interface RestartAvailability {
  enabled: boolean;
  /** Tooltip for a disabled restart; null when there is nothing to explain. */
  reason: string | null;
}

/** Tooltip on the disabled "Update ready" cell (`docs/11` §Photos). */
export const RESTART_AFTER_EXPORT = "Restarts after the export finishes";

/**
 * The export-aware rule: a ready update restarts the app only when no export is running
 * (`docs/12`: "An update never interrupts an export").
 */
export function restartAvailability(state: UpdateState, exportRunning: boolean): RestartAvailability {
  if (state.phase !== "ready") return { enabled: false, reason: null };
  if (exportRunning) return { enabled: false, reason: RESTART_AFTER_EXPORT };
  return { enabled: true, reason: null };
}

/** What the header's update cell shows (`docs/11` §Photos). */
export interface UpdateCell {
  phase: Exclude<UpdateState["phase"], "idle">;
  label: string;
  /** Clicking does this; null when the cell is only a status. */
  action: "download" | "restart" | null;
  disabled: boolean;
  /** Hover text: the failure reason, or why restarting is disabled. */
  title: string | null;
}

/** The cell for a state, or null while idle (the cell is hidden). Pure, so it is easy to test. */
export function updateCell(state: UpdateState, exportRunning: boolean): UpdateCell | null {
  switch (state.phase) {
    case "idle":
      return null;
    case "available":
      return { phase: "available", label: "Update available", action: "download", disabled: false, title: null };
    case "downloading":
      return {
        phase: "downloading",
        label: state.percent === undefined ? "Downloading…" : `Downloading ${state.percent}%`,
        action: null,
        disabled: false,
        title: null,
      };
    case "ready": {
      const restart = restartAvailability(state, exportRunning);
      return { phase: "ready", label: "Update ready", action: "restart", disabled: !restart.enabled, title: restart.reason };
    }
    case "error":
      return { phase: "error", label: "Update failed", action: null, disabled: false, title: state.message };
  }
}

const messageOf = (error: unknown): string =>
  error instanceof Error ? error.message : typeof error === "string" ? error : "The update could not be checked.";

/** Creates an updater. Nothing runs until `start()` or one of the actions is called. */
export function createUpdater(deps: UpdaterDeps): Updater {
  let state: UpdateState = { phase: "idle" };
  let pending: UpdateHandle | null = null;
  const listeners = new Set<() => void>();
  let stopSchedule: (() => void) | null = null;

  function setState(next: UpdateState): void {
    state = next;
    for (const listener of [...listeners]) listener();
  }

  function found(update: UpdateHandle): void {
    pending = update;
    setState({ phase: "available", version: update.version });
  }

  async function quietCheck(): Promise<void> {
    // Never disturb a download or a ready update; an error may be replaced by a fresh result.
    if (state.phase !== "idle" && state.phase !== "error") return;
    try {
      const update = await deps.check();
      if (state.phase !== "idle" && state.phase !== "error") return;
      if (update) found(update);
      else if (state.phase === "error") setState({ phase: "idle" });
    } catch {
      // Silent by design: an unreachable endpoint never bothers the user.
    }
  }

  async function checkNow(): Promise<CheckResult> {
    // An update that is already downloading or ready is the answer.
    if (state.phase === "downloading" || state.phase === "ready") return { kind: "available", version: state.version };
    try {
      const update = await deps.check();
      if (update) {
        found(update);
        return { kind: "available", version: update.version };
      }
      pending = null;
      setState({ phase: "idle" });
      return { kind: "up_to_date" };
    } catch (error) {
      const message = messageOf(error);
      setState({ phase: "error", message });
      return { kind: "error", message };
    }
  }

  async function download(): Promise<void> {
    if (state.phase !== "available" || !pending) return;
    const { version } = state;
    const update = pending;
    setState({ phase: "downloading", version });
    let total: number | null = null;
    let received = 0;
    const onEvent = (event: DownloadEvent): void => {
      if (event.event === "Started") {
        total = event.data.contentLength ?? null;
        setState(total ? { phase: "downloading", version, percent: 0 } : { phase: "downloading", version });
      } else if (event.event === "Progress") {
        received += event.data.chunkLength;
        if (total) setState({ phase: "downloading", version, percent: Math.min(100, Math.round((received / total) * 100)) });
      } else {
        setState({ phase: "downloading", version, percent: 100 });
      }
    };
    try {
      await update.downloadAndInstall(onEvent);
      setState({ phase: "ready", version });
    } catch (error) {
      setState({ phase: "error", message: messageOf(error) });
    }
  }

  async function restart(exportRunning: boolean): Promise<boolean> {
    if (!restartAvailability(state, exportRunning).enabled) return false;
    try {
      await deps.relaunch();
      return true;
    } catch (error) {
      setState({ phase: "error", message: messageOf(error) });
      return false;
    }
  }

  async function scheduledCheck(): Promise<void> {
    const enabled = deps.autoCheck ? await deps.autoCheck() : true;
    if (enabled) await quietCheck();
  }

  function start(): () => void {
    stopSchedule?.();
    let interval: ReturnType<typeof setInterval> | undefined;
    const first = setTimeout(() => {
      void scheduledCheck();
      interval = setInterval(() => void scheduledCheck(), CHECK_INTERVAL_MS);
    }, FIRST_CHECK_DELAY_MS);
    const stop = (): void => {
      clearTimeout(first);
      if (interval !== undefined) clearInterval(interval);
      if (stopSchedule === stop) stopSchedule = null;
    };
    stopSchedule = stop;
    return stop;
  }

  return {
    getState: () => state,
    subscribe(listener) {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
    quietCheck,
    checkNow,
    download,
    restart,
    start,
    applyExternal(next) {
      setState(next);
    },
  };
}

/** Outside a Tauri window there is no updater: a check finds nothing. */
const realDeps: UpdaterDeps = {
  check: async () => (isTauri() ? await check() : null),
  relaunch: () => relaunch(),
};

/** The app's updater, wired to the Tauri plugins. Call `updater.start()` once at launch. */
export const updater: Updater = createUpdater(realDeps);

/** What `useUpdater` gives a component. */
export interface UseUpdater {
  state: UpdateState;
  /** The header cell for the current state; null while idle. */
  cell: UpdateCell | null;
  /** Runs the cell's action: download when available, restart when ready and no export runs. */
  activate: () => void;
  checkNow: () => Promise<CheckResult>;
}

/**
 * React hook over an updater (the app's by default). Pass `exportRunning` so a ready update
 * can't restart in the middle of an export.
 */
export function useUpdater(exportRunning = false, instance: Updater = updater): UseUpdater {
  const state = useSyncExternalStore(instance.subscribe, instance.getState, instance.getState);
  const cell = updateCell(state, exportRunning);
  const activate = useCallback(() => {
    const current = instance.getState();
    if (current.phase === "available") void instance.download();
    else if (current.phase === "ready") void instance.restart(exportRunning);
  }, [instance, exportRunning]);
  return { state, cell, activate, checkNow: instance.checkNow };
}
