/**
 * What each tab of each kind of node holds in the node view (`docs/23` §7.5). Where a tab matches one of the simple view's, its cards are
 * the simple view's (the same components), written to the node the panel is on; the node view has no Automatic card, no Tone card and no
 * Re-run (decision 21). The Input's controls write the Input and the Output's the Output; a corrector's write that corrector, Main included
 * (Main keeps its Auto semantics, every other corrector has plain numbers with the detent and Reset at 0).
 */
import { AutoToggle, Card, formats, ResetLink, Slider } from "../../components";
import type { NodeId } from "../../api/types";
import type { TabId } from "../../components";
import { inputPatch, MAIN_ID, patchOf, resolvedCorrector } from "../../state/settingsGraph";
import { ColourBalanceCard, WhiteBalanceCard } from "../tools/ColorTab";
import type { ColorTabProps } from "../tools/ColorTab";
import { ColorWarpCard } from "../tools/ColorWarpCard";
import { ContrastSlider, ExposureSlider } from "../tools/controls";
import { CurveCard } from "../tools/CurveTab";
import { NoiseCard, SharpeningCard } from "../tools/DetailTab";
import { LensTab } from "../tools/LensTab";
import { HdrCard, LightHistogram } from "../tools/LightTab";
import { LookCard } from "../tools/LookTab";
import { RawTab } from "../tools/RawTab";
import { AUTO } from "../tools/shared";
import type { TabProps } from "../tools/shared";
import { MaskTab } from "./MaskTab";

// ---- The Input ------------------------------------------------------------------------------------------------------------------------

/** Input › Light: the histogram (the Input's output) and the Exposure card with the row Exposure only, the AUTO tag and Reset. */
export function InputLightTab({ store, photo, rs }: TabProps) {
  return (
    <>
      <LightHistogram store={store} photo={photo} at="input" />
      <Card
        title="Exposure"
        tag={<AutoToggle isAuto={rs.auto.input.exposure} onAuto={() => void store.update(inputPatch({ exposure: AUTO }))} />}
        action={<ResetLink size="card" onClick={() => void store.update(inputPatch({ exposure: AUTO }), true, rs.photo_id)} />}
      >
        <ExposureSlider store={store} rs={rs} />
      </Card>
    </>
  );
}

/** Input › Colour: the White balance card (light presets, Pick white, Kelvin, Tint), which is all of the tab that belongs to the Input. */
export function InputColourTab({ store, rs, pickingWhite, onPickWhite }: Pick<ColorTabProps, "store" | "rs" | "pickingWhite" | "onPickWhite">) {
  return <WhiteBalanceCard store={store} rs={rs} pickingWhite={pickingWhite} onPickWhite={onPickWhite} />;
}

/** Input › Detail: the capture sharpening (Amount, Radius; the Output row is the Output's) and Noise reduction. */
export function InputDetailTab({ store, rs }: TabProps) {
  return (
    <>
      <SharpeningCard store={store} rs={rs} rows="capture" />
      <NoiseCard store={store} rs={rs} />
    </>
  );
}

// ---- A corrector ----------------------------------------------------------------------------------------------------------------------

/** What the corrector's tabs need: a tab's props and the node. */
export interface CorrectorTabProps extends TabProps {
  /** The corrector the panel is on (Main is 1). */
  node: number;
}

/**
 * Corrector › Light: the histogram at the node's curve input, the Exposure card (Exposure trim, Contrast, Brightness; no AUTO tag, because
 * Auto belongs to the Input) and High dynamic range. On Main Contrast, Highlights, Shadows and Black keep their Auto semantics.
 */
export function CorrectorLightTab({ store, photo, rs, node }: CorrectorTabProps) {
  const main = node === MAIN_ID;
  const values = resolvedCorrector(rs.values, node);
  if (!values) return null;
  return (
    <>
      <LightHistogram store={store} photo={photo} at={main ? undefined : node} />
      <Card
        title="Exposure"
        action={
          <ResetLink
            size="card"
            onClick={() =>
              void store.update(
                patchOf(node, { trim: { exposure_ev: 0 }, grade: { contrast: main ? AUTO : { kind: "manual", value: 0 }, brightness: 0 } }),
                true,
                rs.photo_id,
              )
            }
          />
        }
      >
        <Slider
          label="Exposure trim"
          value={values.trim.exposure_ev}
          min={-4}
          max={4}
          step={0.01}
          defaultValue={0}
          format={formats.ev}
          onChange={(exposure_ev, final) => void store.update(patchOf(node, { trim: { exposure_ev } }), final, rs.photo_id)}
        />
        <ContrastSlider store={store} rs={rs} node={node} />
        <Slider
          label="Brightness"
          value={values.grade.brightness}
          min={-100}
          max={100}
          defaultValue={0}
          format={formats.signed}
          onChange={(brightness, final) => void store.update(patchOf(node, { grade: { brightness } }), final, rs.photo_id)}
        />
      </Card>
      <HdrCard store={store} rs={rs} node={node} />
    </>
  );
}

/** The corrector's Colour tab's own props: Pick colour acts on this node. */
export interface CorrectorColourProps extends CorrectorTabProps {
  pickingColor?: boolean;
  onPickColor?: (on: boolean) => void;
}

/**
 * Corrector › Colour: Temperature and tint (the trim, −100 to 100 each, no Pick white in this version), Colour balance, and Colour warp,
 * whose scatter and Pick colour are at the input of this node's warp.
 */
export function CorrectorColourTab({ store, photo, rs, node, pickingColor, onPickColor }: CorrectorColourProps) {
  const values = resolvedCorrector(rs.values, node);
  if (!values) return null;
  const send = (partial: { temperature: number } | { tint: number }, final: boolean): void =>
    void store.update(patchOf(node, { trim: partial }), final, rs.photo_id);
  return (
    <>
      <Card title="Temperature and tint" action={<ResetLink size="card" onClick={() => void store.update(patchOf(node, { trim: { temperature: 0, tint: 0 } }), true, rs.photo_id)} />}>
        <Slider
          label="Temperature"
          value={values.trim.temperature}
          min={-100}
          max={100}
          defaultValue={0}
          format={formats.signed}
          onChange={(temperature, final) => send({ temperature }, final)}
        />
        <Slider label="Tint" value={values.trim.tint} min={-100} max={100} defaultValue={0} format={formats.signed} onChange={(tint, final) => send({ tint }, final)} />
      </Card>
      <ColourBalanceCard store={store} rs={rs} node={node} />
      <ColorWarpCard store={store} photo={photo} rs={rs} node={node} pickingColor={pickingColor} onPickColor={onPickColor} />
    </>
  );
}

/** Corrector › Curve: the node's Curve card, with the histogram at the input of its curve behind the plot. */
export function CorrectorCurveTab({ store, photo, rs, node }: CorrectorTabProps) {
  return <CurveCard store={store} photo={photo} rs={rs} node={node} />;
}

// ---- The Output -----------------------------------------------------------------------------------------------------------------------

/** Output › Look: the Look card, today's (Intensity, Apply to, search, the libraries), one look per photo. */
export function OutputLookTab({ store, photo, rs }: TabProps) {
  return <LookCard store={store} photo={photo} rs={rs} />;
}

/** Output › Detail: the Sharpening card with the Output row only. */
export function OutputDetailTab({ store, rs }: TabProps) {
  return <SharpeningCard store={store} rs={rs} rows="output" />;
}

// ---- The tab a node shows -------------------------------------------------------------------------------------------------------------

/** What `NodeTab` needs: the node and the tab it shows, the photo, and the state of the two picks (`Photos` keeps it). */
export interface NodeTabProps extends TabProps {
  node: NodeId;
  tab: TabId;
  pickingWhite: boolean;
  onPickWhite: (on: boolean) => void;
  pickingColor: boolean;
  onPickColor: (on: boolean) => void;
}

/** The cards of one tab of one node: the Input's (`node` `"input"`), the Output's (`"output"`) or a corrector's. */
export function NodeTab({ store, photo, rs, node, tab, pickingWhite, onPickWhite, pickingColor, onPickColor }: NodeTabProps) {
  if (node === "input") {
    switch (tab) {
      case "lens":
        return <LensTab store={store} photo={photo} rs={rs} />;
      case "light":
        return <InputLightTab store={store} photo={photo} rs={rs} />;
      case "color":
        return <InputColourTab store={store} rs={rs} pickingWhite={pickingWhite} onPickWhite={onPickWhite} />;
      case "detail":
        return <InputDetailTab store={store} photo={photo} rs={rs} />;
      default:
        return <RawTab store={store} photo={photo} rs={rs} />;
    }
  }
  if (node === "output") {
    return tab === "detail" ? <OutputDetailTab store={store} photo={photo} rs={rs} /> : <OutputLookTab store={store} photo={photo} rs={rs} />;
  }
  switch (tab) {
    case "color":
      return <CorrectorColourTab store={store} photo={photo} rs={rs} node={node} pickingColor={pickingColor} onPickColor={onPickColor} />;
    case "curve":
      return <CorrectorCurveTab store={store} photo={photo} rs={rs} node={node} />;
    case "mask":
      return <MaskTab rs={rs} node={node} />;
    default:
      return <CorrectorLightTab store={store} photo={photo} rs={rs} node={node} />;
  }
}

