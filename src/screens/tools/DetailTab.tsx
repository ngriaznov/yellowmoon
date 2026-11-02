import { memo } from "react";
import { AutoToggle, Card, formats, ResetLink, Slider } from "../../components";
import type { SettingsPatch } from "../../api/types";
import { foldPatches, inputPatch, outputPatch } from "../../state/settingsGraph";
import { AUTO, manualOrAuto, OUTPUT_SHARPENING_DEFAULT } from "./shared";
import type { TabProps } from "./shared";

/** Which rows of the Sharpening card show: the capture sharpening (Amount, Radius), the Output row, or both (`docs/23` §7.5). */
export type SharpeningRows = "both" | "capture" | "output";

/**
 * The Sharpening card: Amount and Radius (the capture sharpening, the Input's) and Output (the output sharpening, the Output's). The simple
 * view's Detail tab shows all three; the node view splits them by node: the Input's Detail tab has the first two and the Output's Detail tab
 * has the Output row only (`rows`). Reset puts back what the card shows: Auto sharpening, and 100 % output sharpening.
 */
export function SharpeningCard({ store, rs, rows = "both" }: Pick<TabProps, "store" | "rs"> & { rows?: SharpeningRows }) {
  const sharp = rs.values.input.sharpen;
  const est = rs.estimates.input;
  const capture = rows !== "output";
  const output = rows !== "capture";
  const setSharpen = (amount: number, radius_px: number, final: boolean): void =>
    void store.update(inputPatch({ sharpen: manualOrAuto({ amount, radius_px }, est.sharpen) }), final, rs.photo_id);
  const reset = (): void => {
    const patches: SettingsPatch[] = [];
    if (capture) patches.push(inputPatch({ sharpen: AUTO }));
    if (output) patches.push(outputPatch({ output_sharpening: OUTPUT_SHARPENING_DEFAULT }));
    void store.update(patches.reduce((all, next) => foldPatches(all, next)));
  };
  return (
    <Card title="Sharpening" action={<ResetLink size="card" onClick={reset} />}>
      {capture ? (
        <>
          <Slider
            label="Amount"
            value={sharp.amount}
            min={0}
            max={300}
            defaultValue={est.sharpen.amount}
            format={formats.integer}
            onChange={(amount, final) => setSharpen(amount, sharp.radius_px, final)}
            onReset={() => setSharpen(est.sharpen.amount, sharp.radius_px, true)}
          />
          <Slider
            label="Radius"
            value={sharp.radius_px}
            min={0.3}
            max={2.5}
            step={0.1}
            defaultValue={est.sharpen.radius_px}
            format={formats.px1}
            onChange={(radius, final) => setSharpen(sharp.amount, radius, final)}
            onReset={() => setSharpen(sharp.amount, est.sharpen.radius_px, true)}
          />
        </>
      ) : null}
      {output ? (
        <Slider
          label="Output"
          value={rs.values.output.output_sharpening}
          min={0}
          max={200}
          step={5}
          defaultValue={OUTPUT_SHARPENING_DEFAULT}
          format={formats.percent}
          onChange={(output_sharpening, final) => void store.update(outputPatch({ output_sharpening }), final, rs.photo_id)}
        />
      ) : null}
    </Card>
  );
}

/** The Noise reduction card: Luminance, Colour and Detail, with the AUTO tag (the Input's). */
export function NoiseCard({ store, rs }: Pick<TabProps, "store" | "rs">) {
  const { estimates, auto } = rs;
  const nr = rs.values.input.noise_reduction;
  const est = estimates.input;
  const setNoise = (luminance: number, color: number, detail: number, final: boolean): void =>
    void store.update(inputPatch({ noise_reduction: manualOrAuto({ luminance, color, detail }, est.noise_reduction) }), final, rs.photo_id);
  return (
    <Card
      title="Noise reduction"
      tag={<AutoToggle isAuto={auto.input.noise_reduction} onAuto={() => void store.update(inputPatch({ noise_reduction: AUTO }))} />}
      action={<ResetLink size="card" onClick={() => void store.update(inputPatch({ noise_reduction: AUTO }))} />}
    >
      <Slider
        label="Luminance"
        value={nr.luminance}
        min={0}
        max={100}
        defaultValue={est.noise_reduction.luminance}
        format={formats.integer}
        onChange={(v, final) => setNoise(v, nr.color, nr.detail, final)}
        onReset={() => setNoise(est.noise_reduction.luminance, nr.color, nr.detail, true)}
      />
      <Slider
        label="Colour"
        value={nr.color}
        min={0}
        max={100}
        defaultValue={est.noise_reduction.color}
        format={formats.integer}
        onChange={(v, final) => setNoise(nr.luminance, v, nr.detail, final)}
        onReset={() => setNoise(nr.luminance, est.noise_reduction.color, nr.detail, true)}
      />
      <Slider
        label="Detail"
        value={nr.detail}
        min={0}
        max={100}
        defaultValue={est.noise_reduction.detail}
        format={formats.integer}
        onChange={(v, final) => setNoise(nr.luminance, nr.color, v, final)}
        onReset={() => setNoise(nr.luminance, nr.color, est.noise_reduction.detail, true)}
      />
    </Card>
  );
}

/**
 * Detail: Sharpening (Amount, Radius, Output) and Noise reduction (Luminance, Colour, Detail, with the
 * AUTO tag). Amount and Radius are the capture sharpening and resolve from Automatic; Output is the
 * sharpening of the picture at the size it is shown and written (the owner's decision of 2026-10-13,
 * `docs/11-ui.md`): a plain percentage, 100 by default, so Reset goes back to 100 and not to Auto.
 */
export const DetailTab = memo(function DetailTab({ store, rs }: TabProps) {
  return (
    <>
      <SharpeningCard store={store} rs={rs} />
      <NoiseCard store={store} rs={rs} />
    </>
  );
});
