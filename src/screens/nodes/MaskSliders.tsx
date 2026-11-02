import type { MaskPatch } from "../../api/types";
import { formats, Slider } from "../../components";
import { growField, growSlider, opacityField, opacitySlider, refineField, refineSlider, unitField, unitSlider } from "../tools/maskScales";

/** Sends one field of a row as a `masks.edit` row patch; `final` is false for the frames of a drag and true for its release, a key or a typed value. */
export type SendRow = (edit: MaskPatch, final: boolean) => void;

/** Opacity, 0 to 100 % for 0 to 1 (every kind of row has it). */
export function OpacitySlider({ value, send }: { value: number; send: SendRow }) {
  return (
    <Slider label="Opacity" value={opacitySlider(value)} min={0} max={100} defaultValue={100} format={formats.percent} onChange={(v, final) => send({ opacity: opacityField(v) }, final)} />
  );
}

/** Feather, 0 to 100 for 0 to 1; `defaultValue` is the kind's own (a new Linear is 20, a new Radial 30, the raster kinds 0). */
export function FeatherSlider({ value, defaultValue, send }: { value: number; defaultValue: number; send: SendRow }) {
  return (
    <Slider
      label="Feather"
      value={unitSlider(value)}
      min={0}
      max={100}
      defaultValue={unitSlider(defaultValue)}
      format={formats.integer}
      onChange={(v, final) => send({ feather: unitField(v) }, final)}
    />
  );
}

/** The four numbers every raster row has besides its source (`docs/23` §3.3): what the sliders of the raster cards show and write. */
export interface RasterFields {
  refine: number;
  feather: number;
  grow: number;
  opacity: number;
}

/**
 * The sliders the raster cards share (Brush now; Object, Subject, People and Background reuse them), in the order of `docs/23` §7.6:
 * **Refine edges** (percent; a new row's value is the kind's own, 50 % for the AI masks and 0 for Brush), **Feather** (0 to 100), **Grow**
 * (−100 to 100, a bipolar slider: negative takes the mask in, positive lets it out) and **Opacity** (percent), each with its scaling from
 * `docs/algorithms/06` §4.5 (`maskScales.ts`). Every slider writes one field of the row as a `masks.edit` row patch: a drag is a patch per frame
 * that folds into one, and one undo step. `defaults` are the fields of a new row of the kind, for the sliders' marks and Reset.
 */
export function RasterSliders({ row, defaults, send }: { row: RasterFields; defaults: RasterFields; send: SendRow }) {
  return (
    <>
      <Slider
        label="Refine edges"
        value={refineSlider(row.refine)}
        min={0}
        max={100}
        defaultValue={refineSlider(defaults.refine)}
        format={formats.percent}
        onChange={(v, final) => send({ refine: refineField(v) }, final)}
      />
      <FeatherSlider value={row.feather} defaultValue={defaults.feather} send={send} />
      <Slider
        label="Grow"
        value={growSlider(row.grow)}
        min={-100}
        max={100}
        defaultValue={growSlider(defaults.grow)}
        format={formats.signed}
        onChange={(v, final) => send({ grow: growField(v) }, final)}
      />
      <OpacitySlider value={row.opacity} send={send} />
    </>
  );
}
