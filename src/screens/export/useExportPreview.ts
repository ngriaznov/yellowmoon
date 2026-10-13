import { useEffect, useRef, useState } from "react";
import type { ExportEstimate, ExportJob } from "../../api/types";
import type { AppStore } from "../../state/store";

/** What the Export panel knows about a job before it runs. */
export interface ExportPreview {
  /** Size and time ("≈ 120 MB"), once the core has answered. */
  estimate: ExportEstimate | null;
  /** The equivalent `ym render …` command. */
  cli: string | null;
}

/** Milliseconds a change waits before the core is asked again, so a dragged slider asks once. The first question is not delayed. */
const DEBOUNCE_MS = 150;

/**
 * Asks the core for `export_estimate` and `cli_preview` whenever the job changes (`docs/09` §Options). `key` stands for
 * everything the job depends on beyond its options (which photos are in scope), so the answer is asked for again when that
 * changes. An answer to an older question is thrown away. While `enabled` is false nothing is asked.
 */
export function useExportPreview(store: AppStore, job: ExportJob, key: string, enabled: boolean): ExportPreview {
  const [preview, setPreview] = useState<ExportPreview>({ estimate: null, cli: null });
  const latest = useRef(0);
  const answered = useRef(false);
  const text = JSON.stringify(job);

  useEffect(() => {
    if (!enabled) return;
    const ticket = ++latest.current;
    const timer = setTimeout(() => {
      const asked: ExportJob = JSON.parse(text) as ExportJob;
      void Promise.allSettled([store.api.exportEstimate(asked), store.api.cliPreview(asked)]).then(([estimate, cli]) => {
        if (ticket !== latest.current) return;
        answered.current = true;
        setPreview({
          estimate: estimate.status === "fulfilled" ? estimate.value : null,
          cli: cli.status === "fulfilled" ? cli.value : null,
        });
      });
    }, answered.current ? DEBOUNCE_MS : 0);
    return () => clearTimeout(timer);
  }, [store, text, key, enabled]);

  return preview;
}
