import { memo, useEffect } from "react";
import { AutoToggle, Card, formats, Histogram, ResetLink, Slider } from "../../components";
import type { NodeId, PhotoSummary, SettingsPatch } from "../../api/types";
import { useAppState } from "../../state/context";
import { shutterLabel } from "../../state/photo";
import { foldPatches, inputPatch, MAIN_ID, mainGradePatch, patchOf, resolvedCorrector } from "../../state/settingsGraph";
import type { AppStore } from "../../state/store";
import { ContrastSlider, ExposureSlider, PercentSlider } from "./controls";
import { AUTO } from "./shared";
import type { TabProps } from "./shared";

/** Drawn until the first histogram arrives, so the panel doesn't jump. */
const EMPTY_HISTOGRAM = { red: [0, 0], green: [0, 0], blue: [0, 0] };

/**
 * The histogram at the top of the Light tab, with the display values at the middle of the frame at its left and the photo's ISO, shutter
 * and aperture at its right. `at` is the node whose signal it is (`docs/01`): left out, Main's; the node view names the selected node.
 */
export function LightHistogram({ store, photo, at }: { store: AppStore; photo: PhotoSummary; at?: NodeId }) {
  const histogram = useAppState((s) => s.histogram);
  const middle = useAppState((s) => s.middleReadout);
  const readout = middle ? `R ${middle.r}  G ${middle.g}  B ${middle.b}` : null;

  // The plot follows the frames the viewport draws, at most five times a second; the left caption reads the middle of the frame
  // (`store.watchHistogram`). Nothing is asked for per frame: every request is a job in the queue that also draws the viewport.
  useEffect(() => store.watchHistogram({ readout: true, at }), [store, photo.id, at]);

  return (
    <Histogram
      data={histogram ?? EMPTY_HISTOGRAM}
      left={readout ?? undefined}
      right={`ISO ${photo.iso} · ${shutterLabel(photo.shutter_s)} · f/${Math.round(photo.aperture_f * 10) / 10}`}
    />
  );
}

/**
 * The High dynamic range card: Highlights, Shadows, White and Black. On Main the three that can be Auto keep their Auto semantics and the
 * card's Reset puts them back to Auto; on another corrector (`node`, the node view) they are plain numbers and Reset puts them at 0.
 */
export function HdrCard({ store, rs, node = MAIN_ID }: Pick<TabProps, "store" | "rs"> & { node?: number }) {
  const main = node === MAIN_ID;
  const grade = resolvedCorrector(rs.values, node)?.grade;
  if (!grade) return null;
  const zero = { kind: "manual", value: 0 } as const;
  const send = (partial: { white: number }) => (final: boolean) => void store.update(patchOf(node, { grade: partial }), final, rs.photo_id);
  return (
    <Card
      title="High dynamic range"
      action={
        <ResetLink
          size="card"
          onClick={() =>
            void store.update(
              patchOf(node, { grade: main ? { highlights: AUTO, shadows: AUTO, white: 0, black: AUTO } : { highlights: zero, shadows: zero, white: 0, black: zero } }),
              true,
              rs.photo_id,
            )
          }
        />
      }
    >
      <PercentSlider store={store} rs={rs} node={node} field="highlights" label="Highlights" />
      <PercentSlider store={store} rs={rs} node={node} field="shadows" label="Shadows" />
      <Slider label="White" value={grade.white} min={-100} max={100} defaultValue={0} format={formats.signed} onChange={(v, final) => send({ white: v })(final)} />
      <PercentSlider store={store} rs={rs} node={node} field="black" label="Black" />
    </Card>
  );
}

/**
 * Light: Exposure (with the AUTO tag) and High dynamic range (`docs/11-ui.md` §Photos). Exposure is
 * Automatic until the person moves it; Reset and the AUTO button go back to Automatic, not to the
 * number Automatic resolves to. Highlights, Shadows and Black are Auto or a value, like Contrast
 * (`docs/19` section 10): their detent is the Auto value (0, or Apple Auto's), and the card's Reset
 * puts them back to Auto.
 */
export const LightTab = memo(function LightTab({ store, photo, rs }: TabProps) {
  const { auto } = rs;
  const grade = rs.values.main.grade;

  // A drag names the photo it began on (`store.update`), so moving to another photo before its frame or its release cannot send it there.
  const send = (patch: SettingsPatch) => (final: boolean) => void store.update(patch, final, rs.photo_id);

  return (
    <>
      <LightHistogram store={store} photo={photo} />
      <Card
        title="Exposure"
        tag={<AutoToggle isAuto={auto.input.exposure} onAuto={() => void store.update(inputPatch({ exposure: AUTO }))} />}
        action={
          <ResetLink
            size="card"
            onClick={() =>
              void store.update(foldPatches(inputPatch({ exposure: AUTO }), mainGradePatch({ contrast: AUTO, brightness: 0 })), true, rs.photo_id)
            }
          />
        }
      >
        <ExposureSlider store={store} rs={rs} />
        <ContrastSlider store={store} rs={rs} />
        <Slider
          label="Brightness"
          value={grade.brightness}
          min={-100}
          max={100}
          defaultValue={0}
          format={formats.signed}
          onChange={(v, final) => send(mainGradePatch({ brightness: v }))(final)}
        />
      </Card>
      <HdrCard store={store} rs={rs} />
    </>
  );
});
