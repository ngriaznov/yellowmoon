import { useEffect, useState } from "react";
import { AutoToggle, Card, formats, Histogram, ResetLink, Slider } from "../../components";
import type { SettingsPatch } from "../../api/types";
import { useAppState } from "../../state/context";
import { shutterLabel } from "../../state/photo";
import { AUTO, manualOrAuto } from "./shared";
import type { TabProps } from "./shared";

/** Drawn until the first histogram arrives, so the panel doesn't jump. */
const EMPTY_HISTOGRAM = { red: [0, 0], green: [0, 0], blue: [0, 0] };

/**
 * Light: Exposure (with the AUTO tag) and High dynamic range (`docs/11-ui.md` §Photos). Exposure
 * is Automatic until the person moves it; Reset and the AUTO button go back to Automatic, not to
 * the number Automatic resolves to.
 */
export function LightTab({ store, photo, rs }: TabProps) {
  const { values, estimates, auto } = rs;
  const histogram = useAppState((s) => s.histogram);
  const stats = useAppState((s) => s.stats);
  const [readout, setReadout] = useState<string | null>(null);

  // The plot follows every render. The left caption reads the middle of the frame.
  useEffect(() => {
    void store.requestHistogram();
    let live = true;
    void store.samplePixelAt(0.5, 0.5).then((pixel) => {
      if (live && pixel) setReadout(`R ${pixel.r}  G ${pixel.g}  B ${pixel.b}`);
    });
    return () => {
      live = false;
    };
  }, [store, stats, photo.id]);

  const send = (patch: SettingsPatch) => (final: boolean) => void store.update(patch, final);
  const exposure = (ev: number, final: boolean): void => void store.update({ exposure: manualOrAuto({ ev }, { ev: estimates.exposure_ev }) }, final);

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
          <ResetLink size="card" onClick={() => void store.update({ exposure: AUTO, contrast: 0, brightness: 0 })} />
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
        <Slider
          label="Contrast"
          value={values.contrast}
          min={-100}
          max={100}
          defaultValue={0}
          format={formats.signed}
          onChange={(v, final) => send({ contrast: v })(final)}
        />
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
        title="High dynamic range"
        action={<ResetLink size="card" onClick={() => void store.update({ highlights: 0, shadows: 0, white: 0, black: 0 })} />}
      >
        <Slider label="Highlights" value={values.highlights} min={-100} max={100} defaultValue={0} format={formats.signed} onChange={(v, final) => send({ highlights: v })(final)} />
        <Slider label="Shadows" value={values.shadows} min={-100} max={100} defaultValue={0} format={formats.signed} onChange={(v, final) => send({ shadows: v })(final)} />
        <Slider label="White" value={values.white} min={-100} max={100} defaultValue={0} format={formats.signed} onChange={(v, final) => send({ white: v })(final)} />
        <Slider label="Black" value={values.black} min={-100} max={100} defaultValue={0} format={formats.signed} onChange={(v, final) => send({ black: v })(final)} />
      </Card>
    </>
  );
}
