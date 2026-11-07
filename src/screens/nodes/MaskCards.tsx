import { PipetteIcon } from "lucide-react";
import type { MaskItem, MaskPatch, ResolvedSettings } from "../../api/types";
import { Button, Card, formats, ResetLink, Slider } from "../../components";
import { newMask, resetRowPatch } from "../../state/maskRows";
import { patchOf } from "../../state/settingsGraph";
import type { AppStore } from "../../state/store";
import { angleSlider, radiusField, radiusSlider, unitField, unitSlider, widthField, widthSlider } from "../tools/maskScales";
import { BrushCard } from "./BrushCard";
import { DepthCard } from "./DepthCard";
import { ObjectCard } from "./ObjectCard";
import { FeatherSlider, OpacitySlider, RasterSliders } from "./MaskSliders";

/** The kinds that have a card in the Mask tab: Object, Subject, People, Background, the shapes, Brush and the ranges. */
export const CARD_KINDS: ReadonlySet<MaskItem["kind"]> = new Set(["object", "subject", "people", "background", "depth", "linear", "radial", "brush", "color_range", "brightness_range"]);

/** The title of a kind's card: the kind in capitals (the card's own style does the capitals; this is the word). */
const TITLE: Readonly<Partial<Record<MaskItem["kind"], string>>> = {
  object: "Object",
  subject: "Subject",
  people: "People",
  background: "Background",
  depth: "Depth",
  linear: "Linear",
  radial: "Radial",
  brush: "Brush",
  color_range: "Colour range",
  brightness_range: "Brightness range",
};

export interface MaskCardProps {
  store: AppStore;
  rs: ResolvedSettings;
  /** The corrector the row is in. */
  node: number;
  /** The row's index in the node's `masks.items`, and the row. */
  index: number;
  row: MaskItem;
  /** Pick colour of a Colour range is on (`Photos` keeps it) and turns it on or off. */
  pickingRange: boolean;
  onPickRange: (on: boolean) => void;
  /** Pick depth of a Depth row is on (`Photos` keeps it) and turns it on or off. */
  pickingDepth: boolean;
  onPickDepth: (on: boolean) => void;
  /** The sentence of the failure when the photo's depth map failed, for a Depth row's card; else null. */
  failure: string | null;
}

/**
 * The selected mask's card (`docs/23` §7.6) for an Object, a Background, a Linear, a Radial, a Brush, a Colour range or a Brightness range: its title is the kind in capitals, its
 * **Reset** puts the row back as a new row of the kind in the same mode, and its sliders are the standard slider with the scales of
 * `docs/algorithms/06` §4.5 (`maskScales.ts`). Every slider writes one field of the row, as a `masks.edit` row patch (`docs/23` §3.7): a drag is
 * a patch per frame that folds into one, and one undo step.
 */
export function MaskCard({ store, rs, node, index, row, pickingRange, onPickRange, pickingDepth, onPickDepth, failure }: MaskCardProps) {
  const send = (edit: MaskPatch, final: boolean): void => void store.update(patchOf(node, { masks: { edit: { [String(index)]: edit } } }), final, rs.photo_id);
  const fresh = newMask(row.kind, row.mode);
  const reset = (): void => void store.update(patchOf(node, { masks: { edit: { [String(index)]: resetRowPatch(row) } } }), true, rs.photo_id);
  const opacity = <OpacitySlider value={row.opacity} send={send} />;
  const feather = (value: number, defaultValue: number) => <FeatherSlider value={value} defaultValue={defaultValue} send={send} />;
  const softness = (value: number) => (
    <Slider
      label="Softness"
      value={unitSlider(value)}
      min={0}
      max={100}
      defaultValue={50}
      format={formats.integer}
      onChange={(v, final) => send({ softness: unitField(v) }, final)}
    />
  );
  const action = <ResetLink size="card" onClick={reset} />;

  switch (row.kind) {
    case "subject":
    case "people": {
      // Vision's mask has nothing of its own to set: the raster sliders every raster card has (`docs/23` §7.6).
      const was = fresh as typeof row;
      return (
        <Card title={TITLE[row.kind] ?? row.kind} action={action}>
          <RasterSliders row={row} defaults={was} send={send} />
        </Card>
      );
    }
    case "background": {
      // Level and Softness cut the photo's blur map (`docs/algorithms/06` §5.4); the raster sliders after them are the ones every raster card has.
      const was = fresh as typeof row;
      return (
        <Card title={TITLE.background ?? "Background"} action={action}>
          <Slider
            label="Level"
            value={unitSlider(row.level)}
            min={0}
            max={100}
            defaultValue={unitSlider(was.level)}
            format={formats.integer}
            onChange={(v, final) => send({ level: unitField(v) }, final)}
          />
          {softness(row.softness)}
          <RasterSliders row={row} defaults={was} send={send} />
        </Card>
      );
    }
    case "depth":
      return <DepthCard store={store} rs={rs} node={node} index={index} row={row} picking={pickingDepth} onPick={onPickDepth} failure={failure} />;
    case "linear": {
      const was = fresh as typeof row;
      return (
        <Card title={TITLE.linear ?? "Linear"} action={action}>
          <Slider
            label="Angle"
            value={angleSlider(row.angle_deg)}
            min={-180}
            max={180}
            defaultValue={was.angle_deg}
            format={formats.signed}
            onChange={(v, final) => send({ angle_deg: v }, final)}
          />
          {feather(row.feather, was.feather)}
          {opacity}
        </Card>
      );
    }
    case "radial": {
      const was = fresh as typeof row;
      return (
        <Card title={TITLE.radial ?? "Radial"} action={action}>
          <Slider
            label="Width"
            value={radiusSlider(row.radii[0])}
            min={1}
            max={200}
            defaultValue={radiusSlider(was.radii[0])}
            format={formats.integer}
            onChange={(v, final) => send({ radii: [radiusField(v), row.radii[1]] }, final)}
          />
          <Slider
            label="Height"
            value={radiusSlider(row.radii[1])}
            min={1}
            max={200}
            defaultValue={radiusSlider(was.radii[1])}
            format={formats.integer}
            onChange={(v, final) => send({ radii: [row.radii[0], radiusField(v)] }, final)}
          />
          <Slider
            label="Angle"
            value={angleSlider(row.angle_deg)}
            min={-180}
            max={180}
            defaultValue={was.angle_deg}
            format={formats.signed}
            onChange={(v, final) => send({ angle_deg: v }, final)}
          />
          {feather(row.feather, was.feather)}
          {opacity}
        </Card>
      );
    }
    case "brush":
      return <BrushCard store={store} rs={rs} node={node} index={index} row={row} />;
    case "object":
      return <ObjectCard store={store} rs={rs} node={node} index={index} row={row} />;
    case "color_range": {
      const was = fresh as typeof row;
      return (
        <Card title={TITLE.color_range ?? "Colour range"} action={action}>
          <div className="ym-mask-pick">
            <Button variant="header" size="sm" pressed={pickingRange} icon={<PipetteIcon aria-hidden="true" />} onClick={() => onPickRange(!pickingRange)}>
              Pick colour
            </Button>
          </div>
          <Slider
            label="Width"
            value={widthSlider(row.width)}
            min={0}
            max={100}
            defaultValue={widthSlider(was.width)}
            format={formats.integer}
            onChange={(v, final) => send({ width: widthField(v) }, final)}
          />
          {softness(row.softness)}
          {opacity}
        </Card>
      );
    }
    case "brightness_range": {
      // Low stays at or under High, and High at or over Low: the core reads Low above High as both at their mean, which no slider should ask for.
      const low = unitSlider(row.low);
      const high = unitSlider(row.high);
      return (
        <Card title={TITLE.brightness_range ?? "Brightness range"} action={action}>
          <Slider label="Low" value={low} min={0} max={high} defaultValue={35} format={formats.integer} onChange={(v, final) => send({ low: unitField(v) }, final)} />
          <Slider label="High" value={high} min={low} max={100} defaultValue={100} format={formats.integer} onChange={(v, final) => send({ high: unitField(v) }, final)} />
          {softness(row.softness)}
          {opacity}
        </Card>
      );
    }
    default:
      return null;
  }
}
