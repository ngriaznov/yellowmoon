import { Button, Card, Field, KeyHint, ResetLink, Segmented, Slider, TitleBlock } from "../../components";
import type { CropAspect } from "../../api/types";
import { angleLabel } from "./cropMath";
import type { CropSession } from "./useCropSession";

const ASPECTS: ReadonlyArray<{ value: CropAspect; label: string }> = [
  { value: "original", label: "Original" },
  { value: "square", label: "1:1" },
  { value: "four_by_five", label: "4:5" },
  { value: "sixteen_by_nine", label: "16:9" },
  { value: "free", label: "Free" },
];

/**
 * The crop panel (`docs/11-ui.md` §Crop): replaces the tool panel with a black title block, an
 * Aspect row, a Straighten slider with its detent at 0°, "Rotate 90°" and "Reset", and the Done
 * and Cancel buttons in the footer. Return and Esc do the same as the buttons.
 */
export function CropPanel({ session }: { session: CropSession }) {
  const { draft } = session;
  return (
    <aside aria-label="Crop" className="ym-tools" data-panel="crop">
      <div className="ym-tools__scroll">
        <TitleBlock tone="ink" title="Crop" blurb="Drag the frame over the photo, or use the arrow keys. Straighten to level the horizon." />
        <Card title="Frame" last action={<ResetLink size="card" onClick={session.reset} />}>
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
