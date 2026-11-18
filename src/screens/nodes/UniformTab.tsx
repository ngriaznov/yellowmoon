import { InfoIcon, PipetteIcon } from "lucide-react";
import { TooltipTrigger } from "@/components/glass/tooltip";
import { OptionalTooltip } from "@/components/OptionalTooltip";
import type { ResolvedSettings, TargetValue, UniformAmounts } from "../../api/types";
import { Button, Card, formats, ResetLink, Slider } from "../../components";
import {
  colourName,
  hueGradient,
  LIGHTNESS_MAX_EV,
  LIGHTNESS_MIN_EV,
  SATURATION_MAX,
  SATURATION_MIN,
  swatchLabel,
  targetHex,
} from "../../components/uniformMath";
import { useAppState } from "../../state/context";
import { nodeOf, patchOf } from "../../state/settingsGraph";
import type { ColorUniformItem } from "../../state/settingsGraph";
import type { AppStore } from "../../state/store";

/** What the Target card says with no target (`docs/32` §2.1; `design/colour-uniform/states.jpg`). */
export const UNIFORM_TEXT = {
  noTarget: "No colour yet",
  noTargetLine: "Click a colour in the photo, or set Hue, Saturation and Lightness.",
  picked: "Pick again on the photo, or set the numbers.",
  tooGrey: "That colour is too grey for a hue. It was moved to Saturation 2.",
  pickTip: "Click a colour in the photo. Esc stops.",
  needTarget: "Pick or set a target colour first.",
  atFull: "At 100 % every colour in the mask takes the target's value.",
  targetReset: "Clears the target. The amounts and the masks stay.",
  compressReset: "Sets Hue, Saturation and Lightness back to 0.",
} as const;

/** What the three target fields rest at while there is no target: the first change to one sets a target from these (`docs/32` §2.1). */
export const TARGET_REST: Readonly<TargetValue> = { hue_deg: 0, saturation: SATURATION_MIN, lightness_ev: 0 };

const MINUS = "−";
/** 9.7 → "9.7", 2 → "2". */
const satText = (v: number): string => String(Number(v.toFixed(1)));
/** −0.03 → "−0.03", 0 → "0.00". */
const evText = (v: number): string => formats.ev(v);

const AMOUNT_KEYS = [
  ["hue", "Hue"],
  ["saturation", "Saturation"],
  ["lightness", "Lightness"],
] as const;

const NO_AMOUNTS: UniformAmounts = { hue: 0, saturation: 0, lightness: 0 };

export interface UniformTabProps {
  store: AppStore;
  rs: ResolvedSettings;
  node: number;
  item: ColorUniformItem;
  /** Pick colour of the target is on (`Photos` keeps it) and turns it on or off. */
  picking: boolean;
  onPick: (on: boolean) => void;
}

/**
 * A Colour Uniform node's Compress tab (`docs/32` §2.1, §2.2, §2.6): the Target card (the swatch and its name, Pick colour, and Hue, Saturation
 * and Lightness) and the Compress card (the three amounts, off with their line while there is no target). The swatch opens no picker: glasscn has
 * none, so the three number fields are the other way to a colour, and the keyboard's. Every control writes the node as one patch; a drag is a
 * patch per frame that folds into one undo step.
 */
export function UniformTab({ store, rs, node, item, picking, onPick }: UniformTabProps) {
  const grey = useAppState((s) => s.uniformGrey);
  const target = item.target;
  const send = (partial: Parameters<typeof patchOf>[1], final = true): void => void store.update(patchOf(node, partial), final, rs.photo_id);
  const setTarget = (field: keyof TargetValue, value: number, final: boolean): void => send({ target: { ...(target ?? TARGET_REST), [field]: value } }, final);
  const setAmount = (field: keyof UniformAmounts, value: number, final: boolean): void => send({ compress: { [field]: value } }, final);
  const note = target !== null && grey !== null && grey.photo === rs.photo_id && grey.node === node && grey.target.hue_deg === target.hue_deg && grey.target.saturation === target.saturation && grey.target.lightness_ev === target.lightness_ev;
  const shown = target ?? TARGET_REST;
  const dash = (text: (v: number) => string) => (target === null ? () => "–" : text);
  const amountsSet = AMOUNT_KEYS.some(([key]) => item.compress[key] !== 0);

  return (
    <>
      <Card title="Target" action={target ? <ResetLink size="card" aria-label="Reset target" title={UNIFORM_TEXT.targetReset} onClick={() => send({ target: null })} /> : undefined}>
        <div className="ym-uniform__head">
          {target ? (
            <div className="ym-uniform__swatch" role="img" aria-label={swatchLabel(target)} style={{ background: targetHex(target) }} />
          ) : (
            <div className="ym-uniform__swatch" data-empty="" role="img" aria-label="No target colour yet" />
          )}
          <div className="ym-uniform__text">
            <strong className="ym-uniform__name">{target ? colourName(target) : UNIFORM_TEXT.noTarget}</strong>
            <span className="ym-uniform__line">{target ? UNIFORM_TEXT.picked : UNIFORM_TEXT.noTargetLine}</span>
          </div>
          <OptionalTooltip text={UNIFORM_TEXT.pickTip} side="top">
            <TooltipTrigger asChild>
              <Button variant="header" size="sm" pressed={picking} icon={<PipetteIcon aria-hidden="true" />} onClick={() => onPick(!picking)}>
                Pick colour
              </Button>
            </TooltipTrigger>
          </OptionalTooltip>
        </div>
        {note ? (
          <p className="ym-uniform__note" role="status">
            <InfoIcon aria-hidden="true" width={14} height={14} />
            {UNIFORM_TEXT.tooGrey}
          </p>
        ) : null}
        <Slider
          label="Hue"
          value={Math.round(shown.hue_deg) % 360}
          min={0}
          max={359}
          step={1}
          format={dash((v) => `${Math.round(v)}°`)}
          trackBackground={hueGradient()}
          onChange={(v, final) => setTarget("hue_deg", v, final)}
        />
        <Slider
          label="Saturation"
          value={shown.saturation}
          min={SATURATION_MIN}
          max={SATURATION_MAX}
          step={0.1}
          format={dash(satText)}
          onChange={(v, final) => setTarget("saturation", v, final)}
        />
        <Slider
          label="Lightness (EV)"
          value={shown.lightness_ev}
          min={LIGHTNESS_MIN_EV}
          max={LIGHTNESS_MAX_EV}
          step={0.01}
          format={dash(evText)}
          onChange={(v, final) => setTarget("lightness_ev", v, final)}
        />
      </Card>
      <Card
        title="Compress"
        last
        action={amountsSet ? <ResetLink size="card" aria-label="Reset Compress" title={UNIFORM_TEXT.compressReset} onClick={() => send({ compress: { ...NO_AMOUNTS } })} /> : undefined}
      >
        {AMOUNT_KEYS.map(([key, label]) => (
          <Slider
            key={key}
            label={label}
            value={item.compress[key]}
            min={0}
            max={100}
            step={1}
            defaultValue={0}
            format={formats.percent}
            disabled={target === null}
            onChange={(v, final) => setAmount(key, v, final)}
          />
        ))}
        <p className="ym-uniform__line ym-uniform__foot">{target === null ? UNIFORM_TEXT.needTarget : UNIFORM_TEXT.atFull}</p>
      </Card>
    </>
  );
}

/** The node's item from the resolved settings, or null when it is not a Colour Uniform node (the Compress tab is only on one). */
export function uniformItemOf(rs: ResolvedSettings, node: number): ColorUniformItem | null {
  const item = nodeOf(rs.settings, node);
  return item?.kind === "color_uniform" ? item : null;
}
