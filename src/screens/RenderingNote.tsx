import { useEffect, useState } from "react";
import { Badge } from "@/components/glass/badge";
import { Spinner } from "@/components/glass/spinner";
import { useAppState } from "../state/context";

/** How long the CPU must have been rendering before the note shows, so that a quick frame never flashes it. */
export const RENDERING_NOTE_DELAY_MS = 250;

/** What the note says. */
export const RENDERING_NOTE_TEXT = "Rendering on the CPU";

/**
 * "Rendering on the CPU", with a spinner, at the top centre of the photo cell (`docs/11` §Photos): shown once the core has said for
 * {@link RENDERING_NOTE_DELAY_MS} that the viewport is working on a frame or its full-resolution tiles and that the photo's raw stages
 * run on the CPU (`viewport_busy` with `busy` and `cpu`), and gone the moment `busy` is false. On a graphics card a frame takes
 * milliseconds and the note never shows. A polite status; it takes no pointer events, so the cell's pan, zoom and picks work through it.
 */
export function RenderingNote() {
  const rendering = useAppState((s) => s.viewportBusy.busy && s.viewportBusy.cpu);
  const [held, setHeld] = useState(false);
  useEffect(() => {
    if (!rendering) return;
    const timer = setTimeout(() => setHeld(true), RENDERING_NOTE_DELAY_MS);
    return () => {
      clearTimeout(timer);
      setHeld(false);
    };
  }, [rendering]);
  if (!rendering || !held) return null;
  return (
    <Badge variant="outline" className="ym-photo-cell__busy" role="status" aria-live="polite">
      <Spinner aria-hidden="true" />
      {RENDERING_NOTE_TEXT}
    </Badge>
  );
}
