/**
 * Whole-window file drops. In the app the webview does not see DOM drop events (Tauri takes
 * them over and hands out real file paths through `onDragDropEvent`); in a browser, in Vitest
 * and in Playwright the DOM events carry `File` objects with no path, so the names stand in
 * for paths and the mock API opens its twelve fixture photos whatever they are called.
 *
 * `useWindowDrop` picks the right source, tracks "a drag is over the window" for the drop
 * zone's highlight, and calls `onDrop` with the paths.
 */
import { useEffect, useState } from "react";
import { isTauri } from "../env";
import { useLatest } from "./hooks";

/** What a drop source tells its listener. */
export interface DropHandlers {
  /** A drag carrying files entered. */
  onEnter: () => void;
  /** It left, or ended without a drop. */
  onLeave: () => void;
  /** Files were dropped. */
  onDrop: (paths: string[]) => void;
}

/** Somewhere drops come from. `subscribe` returns the unsubscribe function. */
export interface DropSource {
  subscribe: (handlers: DropHandlers) => () => void;
}

/** Paths of the files in a `DataTransfer`: the real path when the platform gives one, else the name. */
export function pathsFromDataTransfer(data: DataTransfer | null): string[] {
  if (!data) return [];
  return Array.from(data.files).map((file) => (file as File & { path?: string }).path || file.name);
}

const carriesFiles = (event: DragEvent): boolean => Array.from(event.dataTransfer?.types ?? []).includes("Files");

/** Drops from DOM drag events on `target` (the window by default). Only drags that carry files count. */
export function domDropSource(target: Window | HTMLElement = window): DropSource {
  return {
    subscribe({ onEnter, onLeave, onDrop }) {
      // dragenter and dragleave fire for every child, so count them.
      let depth = 0;
      const enter = (event: Event): void => {
        const e = event as DragEvent;
        if (!carriesFiles(e)) return;
        e.preventDefault();
        depth += 1;
        if (depth === 1) onEnter();
      };
      const over = (event: Event): void => {
        const e = event as DragEvent;
        if (!carriesFiles(e)) return;
        e.preventDefault();
        if (e.dataTransfer) e.dataTransfer.dropEffect = "copy";
      };
      const leave = (event: Event): void => {
        if (!carriesFiles(event as DragEvent) || depth === 0) return;
        depth -= 1;
        if (depth === 0) onLeave();
      };
      const drop = (event: Event): void => {
        const e = event as DragEvent;
        if (!carriesFiles(e)) return;
        e.preventDefault();
        depth = 0;
        onLeave();
        onDrop(pathsFromDataTransfer(e.dataTransfer));
      };
      target.addEventListener("dragenter", enter);
      target.addEventListener("dragover", over);
      target.addEventListener("dragleave", leave);
      target.addEventListener("drop", drop);
      return () => {
        target.removeEventListener("dragenter", enter);
        target.removeEventListener("dragover", over);
        target.removeEventListener("dragleave", leave);
        target.removeEventListener("drop", drop);
      };
    },
  };
}

/** Drops from Tauri's webview drag-and-drop event, which carries real paths. */
export function tauriDropSource(): DropSource {
  return {
    subscribe({ onEnter, onLeave, onDrop }) {
      let stop: (() => void) | undefined;
      let cancelled = false;
      void import("@tauri-apps/api/webview").then(async ({ getCurrentWebview }) => {
        const unlisten = await getCurrentWebview().onDragDropEvent((event) => {
          const payload = event.payload;
          if (payload.type === "enter") onEnter();
          else if (payload.type === "leave") onLeave();
          else if (payload.type === "drop") {
            onLeave();
            onDrop(payload.paths);
          }
        });
        if (cancelled) unlisten();
        else stop = unlisten;
      });
      return () => {
        cancelled = true;
        stop?.();
      };
    },
  };
}

/** Tauri's source inside the app, the window's DOM events everywhere else. */
export const defaultDropSource = (): DropSource => (isTauri() ? tauriDropSource() : domDropSource());

export interface UseWindowDropOptions {
  /** Files were dropped anywhere in the window. */
  onDrop: (paths: string[]) => void;
  /** Turn the listener off (while a modal is open). Default true. */
  enabled?: boolean;
  /** For tests and demos. Default `defaultDropSource()`. */
  source?: DropSource;
}

/**
 * Handles files dropped anywhere in the window. Returns `dragging`, true while a drag with files
 * is over the window; pass it to `DropZone` as `active`. Do not also pass `onDropPaths` to a
 * `DropZone` for the same paths: it would open them twice.
 */
export function useWindowDrop({ onDrop, enabled = true, source }: UseWindowDropOptions): { dragging: boolean } {
  const [dragging, setDragging] = useState(false);
  const latest = useLatest(onDrop);
  useEffect(() => {
    if (!enabled) {
      setDragging(false);
      return;
    }
    return (source ?? defaultDropSource()).subscribe({
      onEnter: () => setDragging(true),
      onLeave: () => setDragging(false),
      onDrop: (paths) => latest.current(paths),
    });
  }, [enabled, source, latest]);
  return { dragging };
}

/** True for a Sony raw file name: ends in `.ARW`, in any case. */
export const isArwPath = (path: string): boolean => /\.arw$/i.test(path);

/**
 * True for a file name that ends in a file extension: letters or digits after the last dot, at least
 * one a letter, and no spaces. A folder called "Lisbon 2026.10" or "Trip. Day 2" has none, so it is
 * handed to the core, which opens it and keeps the ARW files in it.
 */
const hasExtension = (path: string): boolean => {
  const name = path.split(/[\\/]/).filter(Boolean).pop() ?? "";
  const extension = /\.([^./\\\s]+)$/.exec(name)?.[1];
  return extension !== undefined && /[a-z]/i.test(extension);
};

/**
 * Splits a drop for the Start screen: `accepted` holds `.ARW` files and anything without an
 * extension (a folder, which the core opens and filters itself); `ignored` holds the rest,
 * which are neither opened nor listed. When `accepted` is empty the drop zone says
 * "No ARW files in what you dropped."
 */
export function filterArwPaths(paths: readonly string[]): { accepted: string[]; ignored: string[] } {
  const accepted: string[] = [];
  const ignored: string[] = [];
  for (const path of paths) (isArwPath(path) || !hasExtension(path) ? accepted : ignored).push(path);
  return { accepted, ignored };
}
