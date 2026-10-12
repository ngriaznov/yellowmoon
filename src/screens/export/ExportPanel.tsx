import { useEffect, useMemo, useRef } from "react";
import { Button, ExportQueueList, ProgressBar, TitleBlock } from "../../components";
import { estimateTitle, formatEstimate, photoCount } from "../../export/format";
import { exportModel } from "../../export/model";
import { queueItems, runBlurb, runNote, runTitle, summarise } from "../../export/run";
import { useAppState } from "../../state/context";
import type { AppStore } from "../../state/store";
import { DNG_BLOCKED, ExportOptionsForm } from "./ExportOptionsForm";
import { useExportPreview } from "./useExportPreview";

export interface ExportPanelProps {
  store: AppStore;
}

/**
 * The Export panel (`design/mockups/export.html`, `docs/11-ui.md` §Export, `docs/09` §Queue). It replaces the tool panel.
 * Three views: the options with the "Export N photos" button; the queue while files are written, with Cancel; and "Exported"
 * with Show in Finder and Done. The run lives in the store, so closing the panel does not stop it and reopening shows where it
 * has got to.
 */
export function ExportPanel({ store }: ExportPanelProps) {
  const session = useAppState((s) => s.session);
  const settings = useAppState((s) => s.settings);
  const looks = useAppState((s) => s.looks);
  const preferences = useAppState((s) => s.preferences);
  const scope = useAppState((s) => s.exportScope);
  const options = useAppState((s) => s.exportOptions);
  const run = useAppState((s) => s.exportRun);
  const stopped = useAppState((s) => s.exportStopped);

  const model = useMemo(() => exportModel({ session, settings, looks, preferences }, scope, options), [session, settings, looks, preferences, scope, options]);
  const job = useMemo(() => ({ scope, options }), [scope, options]);
  const key = model.photos.map((p) => p.id).join(",");
  const preview = useExportPreview(store, job, key, run === null && model.count > 0);
  const { estimate } = preview;

  const dngBlocked = options.format === "dng" && estimate?.dng_unavailable === true;
  const blocker =
    model.count === 0
      ? scope === "picked"
        ? "No photos are picked. Pick some photos, or choose All."
        : "There is nothing to export."
      : model.problem
        ? model.problem
        : dngBlocked
          ? DNG_BLOCKED
          : null;

  const start = async (): Promise<void> => {
    if (run !== null || blocker !== null) return;
    await store.startExport(model.names, model.destination);
  };
  const startRef = useRef(start);
  startRef.current = start;

  // ⌘↩ starts the export, also from the file name box.
  useEffect(() => {
    if (run !== null) return;
    const onKeyDown = (event: KeyboardEvent): void => {
      // A dialog over the window (Preferences) has the keyboard.
      if (event.key === "Enter" && event.metaKey && !event.ctrlKey && !event.altKey && store.getState().overlay === null) {
        event.preventDefault();
        void startRef.current();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [run, store]);

  // When the view changes (options, queue, finished) focus moves to its main button, so the keyboard is not left on nothing.
  const root = useRef<HTMLElement>(null);
  const phase = run === null ? "options" : run.phase;
  const first = useRef(true);
  useEffect(() => {
    if (first.current) {
      first.current = false;
      return;
    }
    root.current?.querySelector<HTMLElement>("[data-autofocus]")?.focus();
  }, [phase]);

  const folder = session?.folder_name ?? "this folder";
  const title = run ? runTitle(run) : "Export";
  const blurb = run
    ? runBlurb(run)
    : stopped
      ? stopped
      : model.count === 0
        ? scope === "picked"
          ? "No photos are picked."
          : "There is nothing to export."
        : `${photoCount(model.count)} from ${folder}. Nothing is saved besides the exported files.`;

  return (
    <aside ref={root} aria-label="Export" className="ym-tools" data-view={phase}>
      <TitleBlock tone="ink" title={title} blurb={blurb} live />
      <div className="ym-tools__scroll">
        {run === null ? (
          <ExportOptionsForm store={store} options={options} mode="panel" model={model} scope={scope} estimate={estimate} cli={preview.cli} />
        ) : (
          <div className="ym-export__queue">
            <ProgressBar value={summarise(run).percent} label="Exporting" />
            <div className="ym-export__list">
              <ExportQueueList items={queueItems(run)} />
            </div>
            <p className="ym-row-note ym-export__queue-note">{runNote(run)}</p>
          </div>
        )}
      </div>
      <div className="ym-tools__footer">
        {run === null ? (
          <Button
            block
            size="lg"
            disabled={blocker !== null}
            title={blocker ?? (estimate ? estimateTitle(estimate) : undefined)}
            aside={`${estimate ? formatEstimate(estimate.bytes) : "…"} · ⌘↩`}
            onClick={() => void start()}
          >
            {model.count > 0 ? `Export ${photoCount(model.count)}` : "Export"}
          </Button>
        ) : run.phase === "running" ? (
          <Button variant="secondary" size="lg" block data-autofocus="" onClick={() => void store.cancelExport()}>
            Cancel
          </Button>
        ) : (
          <div className="ym-export__footer-pair">
            <Button variant="secondary" size="lg" onClick={() => void store.revealExport()}>
              Show in Finder
            </Button>
            <Button size="lg" data-autofocus="" onClick={() => store.finishExport()}>
              Done
            </Button>
          </div>
        )}
      </div>
    </aside>
  );
}
