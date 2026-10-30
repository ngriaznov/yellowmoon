import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import type { LookInput } from "../../api/types";
import { useWindowDrop } from "../../components";
import { useAppState } from "../../state/context";
import type { AppStore } from "../../state/store";
import {
  ARRI_LOOKS_URL,
  canAdvance,
  importPlan,
  initialWizard,
  reduceWizard,
  STEP_FINISH,
  STEP_GET,
  STEP_IMPORT,
  STEP_RECOGNISE,
  trashCandidates,
} from "../../wizard/model";
import type { ImportOutcome, PlanItem, WizardAction, WizardState } from "../../wizard/model";

/** What `useWizard` gives the screen. */
export interface Wizard {
  state: WizardState;
  /** A drag with files is over the window (step 2 only). */
  dragging: boolean;
  /** Downloads are being watched. */
  watching: boolean;
  /** Why the watcher could not start, in words. */
  watchProblem: string | null;
  dispatch: (action: WizardAction) => void;
  advance: () => void;
  back: () => void;
  select: (step: number) => void;
  /** Receive files (dropped or chosen). */
  receive: (paths: string[]) => void;
  chooseFiles: () => Promise<void>;
  openArriPage: () => void;
  setWatch: (on: boolean) => void;
  answer: (path: string, input: LookInput) => void;
  remove: (path: string) => void;
  retryImport: () => void;
  /** Cancel the wizard (leaves without importing anything more). */
  cancel: () => void;
}

/**
 * The wizard's behaviour: the reducer of `wizard/model.ts` plus the calls to the core. Recognition happens as files arrive
 * and again on entering Recognise (a file may have gone); the import starts from the Import button, not from an effect, so
 * StrictMode's double effects can't start it twice.
 */
export function useWizard(store: AppStore): Wizard {
  const [state, dispatch] = useReducer(reduceWizard, undefined, initialWizard);
  const stateRef = useRef(state);
  stateRef.current = state;
  const watchPref = useAppState((s) => s.preferences?.watch_downloads_during_import ?? false);
  const [watching, setWatching] = useState(false);
  const [watchProblem, setWatchProblem] = useState<string | null>(null);

  const recognise = useCallback(
    async (path: string): Promise<void> => {
      try {
        dispatch({ type: "recognised", found: await store.api.looksRecognise(path) });
      } catch (error) {
        store.reportError(error);
      }
    },
    [store],
  );

  const receive = useCallback(
    (paths: string[], fromDownloads = false): void => {
      for (const path of paths) {
        if (stateRef.current.sources.some((s) => s.path === path)) continue;
        dispatch({ type: "received", path, fromDownloads });
        void recognise(path);
      }
    },
    [recognise],
  );

  const { dragging } = useWindowDrop({ onDrop: (paths) => receive(paths), enabled: state.step === STEP_GET });

  // Watch Downloads while step 2 is on screen and the switch is on; zips that land arrive as events.
  useEffect(() => {
    if (state.step !== STEP_GET || !watchPref) {
      setWatching(false);
      return;
    }
    let live = true;
    setWatchProblem(null);
    const stop = store.api.on("download_found", (found) => {
      if (live) receive([found.path], true);
    });
    store.api.looksWatchDownloads(true).then(
      (status) => {
        if (!live) return;
        setWatching(status.watching);
        if (!status.watching) {
          setWatchProblem(status.problem ?? "Yellowmoon can't watch the Downloads folder. Choose the file yourself.");
          void store.setPreferences({ watch_downloads_during_import: false });
        }
      },
      (error) => {
        if (live) store.reportError(error);
      },
    );
    return () => {
      live = false;
      stop();
      setWatching(false);
      void store.api.looksWatchDownloads(false).catch(() => {});
    };
  }, [store, state.step, watchPref, receive]);

  const recheck = useCallback(async (): Promise<void> => {
    dispatch({ type: "checking", on: true });
    await Promise.all(stateRef.current.sources.map((s) => recognise(s.path)));
    dispatch({ type: "checking", on: false });
  }, [recognise]);

  const runImport = useCallback(
    async (plan: PlanItem[]): Promise<void> => {
      const total = plan.reduce((sum, item) => sum + item.files, 0);
      dispatch({ type: "import-start", total });
      let before = 0;
      let current: PlanItem | null = null;
      const stop = store.api.on("looks_import_progress", (e) => {
        if (current && e.path === current.path) dispatch({ type: "import-progress", done: before + Math.min(e.done, current.files) });
      });
      const outcome: ImportOutcome = { looks: 0, zips: [] };
      try {
        for (const item of plan) {
          current = item;
          await store.api.looksImport(item.path, item.input);
          before += item.files;
          dispatch({ type: "import-progress", done: before });
          outcome.looks += item.files;
        }
        outcome.zips = trashCandidates(stateRef.current);
        await store.refreshLooks();
        dispatch({ type: "import-done", outcome });
      } catch (error) {
        const message = (error as { message?: string } | null)?.message;
        dispatch({ type: "import-failed", error: message ?? "The import did not finish. Go back and try again." });
        await store.refreshLooks();
      } finally {
        stop();
      }
    },
    [store],
  );

  const finish = useCallback(async (): Promise<void> => {
    const cur = stateRef.current;
    try {
      const looks = store.getState().looks ?? [];
      const currentDefault = looks.find((l) => l.is_default)?.id;
      if (cur.finish.defaultLook && cur.finish.defaultLook !== currentDefault) await store.makeDefault(cur.finish.defaultLook);
      if (cur.finish.zip === "trash") {
        for (const path of trashCandidates(cur)) await store.api.trashPath(path);
      }
    } catch (error) {
      store.reportError(error);
    }
    store.back();
  }, [store]);

  const advance = useCallback((): void => {
    const cur = stateRef.current;
    if (!canAdvance(cur)) return;
    if (cur.step === STEP_FINISH) {
      void finish();
      return;
    }
    dispatch({ type: "advance" });
    if (cur.step === STEP_GET) void recheck();
    if (cur.step === STEP_RECOGNISE) void runImport(importPlan(cur));
  }, [finish, recheck, runImport]);

  const goBack = useCallback(
    (action: WizardAction): void => {
      const cur = stateRef.current;
      if (cur.import.status === "running") return;
      const next = reduceWizard(cur, action);
      if (next.step === cur.step) return;
      dispatch(action);
      if (next.step === STEP_RECOGNISE) void recheck();
    },
    [recheck],
  );

  return {
    state,
    dragging,
    watching,
    watchProblem,
    dispatch,
    advance,
    back: useCallback(() => goBack({ type: "back" }), [goBack]),
    select: useCallback((step: number) => goBack({ type: "select", step }), [goBack]),
    receive: useCallback((paths: string[]) => receive(paths), [receive]),
    chooseFiles: useCallback(async () => {
      try {
        const paths = await store.api.chooseLookFiles();
        if (paths) receive(paths);
      } catch (error) {
        store.reportError(error);
      }
    }, [store, receive]),
    openArriPage: useCallback(() => {
      void store.api.openUrl(ARRI_LOOKS_URL).catch((error) => store.reportError(error));
    }, [store]),
    setWatch: useCallback(
      (on: boolean) => {
        setWatchProblem(null);
        void store.setPreferences({ watch_downloads_during_import: on });
      },
      [store],
    ),
    answer: useCallback((path: string, input: LookInput) => dispatch({ type: "answer", path, input }), []),
    remove: useCallback((path: string) => dispatch({ type: "remove", path }), []),
    retryImport: useCallback(() => {
      const cur = stateRef.current;
      if (cur.step === STEP_IMPORT && cur.import.status === "failed") void runImport(importPlan(cur));
    }, [runImport]),
    cancel: useCallback(() => {
      if (stateRef.current.import.status === "running") return;
      store.back();
    }, [store]),
  };
}
