import { AutoToggle, Button, Card, Field, KeyHint, ResetLink, RowNote, Segmented, Slider, TitleBlock } from "../../components";
import type { CropAspect, CropSuggestion } from "../../api/types";
import { angleLabel } from "./cropMath";
import type { AutoCropNote, CropSession } from "./useCropSession";

const ASPECTS: ReadonlyArray<{ value: CropAspect; label: string }> = [
  { value: "original", label: "Original" },
  { value: "square", label: "1:1" },
  { value: "four_by_five", label: "4:5" },
  { value: "sixteen_by_nine", label: "16:9" },
  { value: "free", label: "Free" },
];

/** The lines of the Frame card's note about Auto crop, word for word (`docs/20-auto-crop.md` §1). */
export const AUTO_CROP_LINES = {
  asking: "Finding a frame…",
  cropped: "Cropped to frame the subject.",
  kept: "No better frame found, so the whole photo is kept.",
  level: "Already level.",
  unsure: "No horizon or straight lines found, so the photo is not turned.",
  failed: "Couldn't suggest a frame for this photo.",
} as const;

/** "Levelled by 1.4°.": the magnitude of the angle, as the Straighten slider writes it without its sign. */
export const levelledLine = (degrees: number): string => `Levelled by ${angleLabel(Math.abs(degrees)).replace(/^\+/, "")}.`;

/** What a proposal did, as the note's two lines: the frame, then the angle. */
export function autoCropLines(suggestion: CropSuggestion): [string, string] {
  const frame = suggestion.keep ? AUTO_CROP_LINES.kept : AUTO_CROP_LINES.cropped;
  const angle =
    suggestion.straighten === "rotate" ? levelledLine(suggestion.angle_deg) : suggestion.straighten === "level" ? AUTO_CROP_LINES.level : AUTO_CROP_LINES.unsure;
  return [frame, angle];
}

/** The note's lines for its state: none while there is nothing to say. */
export function autoCropNoteLines(note: AutoCropNote): string[] {
  if (note === null) return [];
  if (note.kind === "asking") return [AUTO_CROP_LINES.asking];
  if (note.kind === "failed") return [AUTO_CROP_LINES.failed];
  return autoCropLines(note.suggestion);
}

/**
 * The crop panel (`docs/11-ui.md` §Crop): replaces the tool panel with a black title block, an
 * Aspect row, a Straighten slider with its detent at 0°, "Rotate 90°" and "Reset", and the Done
 * and Cancel buttons in the footer. Return and Esc do the same as the buttons. The Frame card's
 * AUTO is Auto crop (`docs/20-auto-crop.md` §1), with its note at the foot of the card.
 */
export function CropPanel({ session }: { session: CropSession }) {
  const { draft } = session;
  const lines = autoCropNoteLines(session.autoNote);
  return (
    <aside aria-label="Crop" className="ym-tools" data-panel="crop">
      <div className="ym-tools__scroll">
        <TitleBlock tone="ink" title="Crop" blurb="Drag the frame over the photo, or use the arrow keys. Straighten to level the horizon." />
        <Card
          title="Frame"
          last
          tag={
            <AutoToggle
              aria-label="Auto crop"
              isAuto={draft.auto}
              onAuto={session.toggleAuto}
              onOff={session.toggleAuto}
              aria-disabled={session.autoLocked || undefined}
              className="aria-disabled:pointer-events-none aria-disabled:opacity-45"
            />
          }
          action={<ResetLink size="card" onClick={session.reset} />}
        >
          <div className="ym-stack-12">
            <Segmented label="Aspect" size="sm" value={draft.aspect} onChange={session.setAspect} options={ASPECTS} />
          </div>
          <Slider
            label="Straighten"
            value={draft.angle_deg}
            min={-45}
            max={45}
            step={0.1}
            defaultValue={0}
            format={angleLabel}
            onChange={(degrees) => session.setAngle(degrees)}
          />
          <Field label="Orientation">
            <Button variant="secondary" size="md" onClick={session.rotate}>
              Rotate 90°
            </Button>
          </Field>
          {/* Said as it changes: "Finding a frame…", then what Auto crop did. */}
          <div role="status">
            {lines.map((line) => (
              <RowNote key={line}>{line}</RowNote>
            ))}
          </div>
        </Card>
      </div>
      <div className="ym-tools__footer">
        <Button size="lg" block onClick={session.done}>
          Done <KeyHint>↩</KeyHint>
        </Button>
        <Button variant="secondary" size="lg" block onClick={session.cancel}>
          Cancel <KeyHint>Esc</KeyHint>
        </Button>
      </div>
    </aside>
  );
}
