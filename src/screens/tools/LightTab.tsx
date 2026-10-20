import { memo, useEffect } from "react";
import { AutoToggle, Card, formats, Histogram, ResetLink, Slider } from "../../components";
import type { SettingsPatch } from "../../api/types";
import { useAppState } from "../../state/context";
import { shutterLabel } from "../../state/photo";
import { ContrastSlider, PercentSlider } from "./controls";
import { AUTO, WHITE_OFFSET_DEFAULT, WHITE_OFFSET_RANGE_EV, manualOrAuto } from "./shared";
import type { TabProps } from "./shared";

/** Drawn until the first histogram arrives, so the panel doesn't jump. */
const EMPTY_HISTOGRAM = { red: [0, 0], green: [0, 0], blue: [0, 0] };

/**
 * Light: Exposure (with the AUTO tag), White (the automatic white lift and its offset) and High
 * dynamic range (`docs/11-ui.md` §Photos). Exposure is Automatic until the person moves it; Reset and
 * the AUTO button go back to Automatic, not to the number Automatic resolves to. The White card's
 * Offset is a plain number of stops added to the automatic lift: 0 is Automatic. Highlights, Shadows
 * and Black are Auto or a value, like Contrast (`docs/19` section 10): their detent is the Auto value
 * (0, or Apple Auto's), and the card's Reset puts them back to Auto.
 */
export const LightTab = memo(function LightTab({ store, photo, rs }: TabProps) {
  const { values, estimates, auto } = rs;
  const histogram = useAppState((s) => s.histogram);
  const middle = useAppState((s) => s.middleReadout);
  const readout = middle ? `R ${middle.r}  G ${middle.g}  B ${middle.b}` : null;

  // The plot follows the frames the viewport draws, at most five times a second; the left caption reads the middle of the frame
  // (`store.watchHistogram`). Nothing is asked for per frame: every request is a job in the queue that also draws the viewport.
  useEffect(() => store.watchHistogram({ readout: true }), [store, photo.id]);

  // A drag names the photo it began on (`store.update`), so moving to another photo before its frame or its release cannot send it there.
  const send = (patch: SettingsPatch) => (final: boolean) => void store.update(patch, final, rs.photo_id);
  const exposure = (ev: number, final: boolean): void =>
    void store.update({ exposure: manualOrAuto({ ev }, { ev: estimates.exposure_ev }) }, final, rs.photo_id);

  return (
    <>
      <Histogram
        data={histogram ?? EMPTY_HISTOGRAM}
        left={readout ?? undefined}
        right={`ISO ${photo.iso} · ${shutterLabel(photo.shutter_s)} · f/${Math.round(photo.aperture_f * 10) / 10}`}
      />
      <Card
        title="Exposure"
        tag={<AutoToggle isAuto={auto.exposure} onAuto={() => void store.update({ exposure: AUTO })} />}
        action={
          <ResetLink size="card" onClick={() => void store.update({ exposure: AUTO, contrast: AUTO, brightness: 0 }, true, rs.photo_id)} />
        }
      >
        <Slider
          label="Exposure"
          value={values.exposure_ev}
          min={-4}
          max={4}
          step={0.01}
          defaultValue={estimates.exposure_ev}
          format={formats.ev}
          onChange={exposure}
          onReset={() => void store.update({ exposure: AUTO })}
        />
        <ContrastSlider store={store} rs={rs} />
        <Slider
          label="Brightness"
          value={values.brightness}
          min={-100}
          max={100}
          defaultValue={0}
          format={formats.signed}
          onChange={(v, final) => send({ brightness: v })(final)}
        />
      </Card>
      <Card
        title="White"
        tag={
          <AutoToggle
            aria-label="Auto white"
            isAuto={values.white_offset_ev === 0}
            onAuto={() => void store.update({ white_offset_ev: WHITE_OFFSET_DEFAULT })}
          />
        }
        action={<ResetLink size="card" onClick={() => void store.update({ white_offset_ev: WHITE_OFFSET_DEFAULT })} />}
      >
        <Slider
          label="Offset"
          value={values.white_offset_ev}
          min={WHITE_OFFSET_RANGE_EV[0]}
          max={WHITE_OFFSET_RANGE_EV[1]}
          step={0.05}
          defaultValue={WHITE_OFFSET_DEFAULT}
          format={formats.ev}
          onChange={(v, final) => send({ white_offset_ev: v })(final)}
        />
      </Card>
      <Card
        title="High dynamic range"
        action={<ResetLink size="card" onClick={() => void store.update({ highlights: AUTO, shadows: AUTO, white: 0, black: AUTO }, true, rs.photo_id)} />}
      >
        <PercentSlider store={store} rs={rs} field="highlights" label="Highlights" />
        <PercentSlider store={store} rs={rs} field="shadows" label="Shadows" />
        <Slider label="White" value={values.white} min={-100} max={100} defaultValue={0} format={formats.signed} onChange={(v, final) => send({ white: v })(final)} />
        <PercentSlider store={store} rs={rs} field="black" label="Black" />
      </Card>
    </>
  );
});
