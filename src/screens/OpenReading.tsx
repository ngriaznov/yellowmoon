import { Button } from "../components";
import { useAppState, useStore } from "../state/context";

/**
 * "Reading {done} of {total} files…" with a Cancel button, in place of what offers to open more files while the core reads the
 * files of an open (`docs/26` H-10, P3-3). Nothing while no open has taken long enough to say. Cancel stops the reading; the files
 * read so far stay, and the screen says how many ("Stopped. 40 files were opened.").
 */
export function OpenReading({ variant = "secondary" }: { variant?: "secondary" | "header" }) {
  const store = useStore();
  const progress = useAppState((s) => s.openProgress);
  if (progress === null) return null;
  return (
    <span className="ym-open-reading" data-variant={variant === "header" ? "header" : undefined}>
      <span role="status">
        Reading {progress.done} of {progress.total} files…
      </span>
      <Button variant={variant} onClick={() => void store.cancelOpen()}>
        Cancel
      </Button>
    </span>
  );
}
