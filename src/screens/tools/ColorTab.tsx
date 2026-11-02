import { memo } from "react";
import { PipetteIcon } from "lucide-react";
import { AutoToggle, Button, Card, formats, KeyHint, ResetLink, Segmented, Slider, Tag, Wheel, WheelGroup } from "../../components";
import type { ColorBalance, ColorBalancePoint, LightSource } from "../../api/types";
import { gradeOf, inputOf, inputPatch, MAIN_ID, patchOf, resolvedCorrector } from "../../state/settingsGraph";
import { ColorWarpCard } from "./ColorWarpCard";
import type { ColorWarpCardProps } from "./ColorWarpCard";
import { KelvinSlider, SaturationSlider, setWhiteBalance, whiteBalanceState, whiteBalanceToAuto } from "./controls";
import { AUTO, pointOrAuto } from "./shared";
import type { TabProps } from "./shared";

/** The light presets, in the order of the mockup. */
export const LIGHT_OPTIONS: ReadonlyArray<{ value: LightSource; label: string; ariaLabel: string }> = [
  { value: "auto", label: "Auto", ariaLabel: "Auto" },
  { value: "day", label: "Day", ariaLabel: "Daylight" },
  { value: "shade", label: "Shade", ariaLabel: "Shade" },
  { value: "tungsten", label: "Tung.", ariaLabel: "Tungsten" },
  { value: "led", label: "LED", ariaLabel: "LED" },
  { value: "fluorescent", label: "Fluor.", ariaLabel: "Fluorescent" },
];

const WHEELS = [
  ["shadows", "Shadows"],
  ["midtones", "Midtones"],
  ["highlights", "Highlights"],
] as const;

/** The Colour balance card's Reset: the shadows and highlights wheels back to Auto, the midtones wheel to the centre. */
const RESET_BALANCE: ColorBalance = { shadows: AUTO, midtones: [0, 0], highlights: AUTO };
/** The same on a corrector other than Main, where Auto is a manual 0 (`docs/23` §3.2). */
const RESET_BALANCE_OTHER: ColorBalance = { shadows: { kind: "manual", point: [0, 0] }, midtones: [0, 0], highlights: { kind: "manual", point: [0, 0] } };

/**
 * The Colour tab's props: those of every tab, and the state of Pick white and Pick colour, which the Photos screen keeps (it owns the
 * crosshair layer, and ends either mode when the Colour tab is left).
 */
export interface ColorTabProps extends ColorWarpCardProps {
  /** Pick white is on: the row's toggle is pressed and a click on the photo sets the white (`Photos`). */
  pickingWhite?: boolean;
  /** Turns Pick white on or off. Without it the toggle does nothing. */
  onPickWhite?: (on: boolean) => void;
}

/**
 * The White balance card: the light presets, Pick white, Kelvin and Tint, with the AUTO tag, the EDITED tag after a picked white and
 * Reset. All of it is the Input's (`docs/23` §2): the simple view's Colour tab and the node view's Input › Colour tab draw it.
 */
export function WhiteBalanceCard({ store, rs, pickingWhite = false, onPickWhite }: Pick<ColorTabProps, "store" | "rs" | "pickingWhite" | "onPickWhite">) {
  const { settings, estimates } = rs;
  const values = rs.values.input;
  const { picked, isAuto: whiteIsAuto } = whiteBalanceState(rs);
  const toAuto = (): void => whiteBalanceToAuto(store);

  return (
    <Card
      title="White balance"
      tag={
        <>
          {picked ? <Tag variant="edited" title="White picked from the photo" /> : null}
          <AutoToggle isAuto={whiteIsAuto} onAuto={toAuto} />
        </>
      }
      action={<ResetLink size="card" onClick={toAuto} />}
    >
      <div className="ym-stack-12">
        <Segmented
          label="Light type"
          value={inputOf(settings).light_source}
          onChange={(light_source) => void store.update(inputPatch({ light_source }), true)}
          options={LIGHT_OPTIONS.map(({ value, label, ariaLabel }) => ({ value, label, ariaLabel }))}
        />
      </div>
      {/* Pick white (`docs/11` §Pick white): a row of its own, the toggle and, at the right, its key. The chip is for the eye; the key is in
          the toggle's description. */}
      <div className="ym-pick-row">
        <Button
          variant="header"
          size="sm"
          pressed={pickingWhite}
          icon={<PipetteIcon className="size-3.5" aria-hidden="true" />}
          title="Click something neutral on the photo (W)"
          onClick={() => onPickWhite?.(!pickingWhite)}
        >
          Pick white
        </Button>
        <span className="ym-pick-row__keys" aria-hidden="true">
          or press <KeyHint>W</KeyHint>
        </span>
      </div>
      <KelvinSlider store={store} rs={rs} />
      <Slider
        label="Tint"
        value={values.tint}
        min={-50}
        max={50}
        defaultValue={estimates.input.tint}
        format={formats.signed}
        onChange={(tint, final) => setWhiteBalance(store, rs, values.kelvin, tint, final)}
        onReset={() => setWhiteBalance(store, rs, values.kelvin, estimates.input.tint, true)}
      />
    </Card>
  );
}

/**
 * The Colour balance card: three wheels (Shadows, Midtones, Highlights) and Saturation. On Main the shadows and highlights wheels and
 * Saturation keep their Auto semantics; on another corrector (`node`, the node view) they are plain values, centred at 0, and Reset
 * writes manual zeros (`docs/23` §3.2).
 */
export function ColourBalanceCard({ store, rs, node = MAIN_ID }: Pick<TabProps, "store" | "rs"> & { node?: number }) {
  const main = node === MAIN_ID;
  const resolved = resolvedCorrector(rs.values, node);
  if (!resolved) return null;
  const grade = resolved.grade;
  return (
    <Card
      title="Colour balance"
      action={
        <ResetLink
          size="card"
          onClick={() =>
            void store.update(
              patchOf(node, { grade: { color_balance: main ? RESET_BALANCE : RESET_BALANCE_OTHER, saturation: main ? AUTO : { kind: "manual", value: 0 } } }),
              true,
              rs.photo_id,
            )
          }
        />
      }
    >
      <WheelGroup>
        {WHEELS.map(([key, label]) => {
          // On Main the shadows and highlights wheels are Auto or a point (`docs/19` section 5): the puck shows the point in effect, a
          // point dropped on the Auto point is sent as Auto, and Reset (double-click, Home) sends Auto. The midtones wheel is a point.
          // On another corrector every wheel is a point and Reset centres it.
          const auto: ColorBalancePoint = !main ? [0, 0] : key === "shadows" ? rs.estimates.main.balance_shadows : key === "highlights" ? rs.estimates.main.balance_highlights : [0, 0];
          const send = (point: ColorBalancePoint, final: boolean): void => {
            const balance = gradeOf(rs.settings, node).color_balance;
            const manual = (p: ColorBalancePoint): ColorBalance["shadows"] => (main ? pointOrAuto(p, auto) : { kind: "manual", point: p });
            const next: ColorBalance = key === "midtones" ? { ...balance, midtones: point } : { ...balance, [key]: manual(point) };
            void store.update(patchOf(node, { grade: { color_balance: next } }), final, rs.photo_id);
          };
          const reset = (): void => {
            const reset = main ? AUTO : ({ kind: "manual", point: [0, 0] } as const);
            void store.update(patchOf(node, { grade: { color_balance: { ...gradeOf(rs.settings, node).color_balance, [key]: reset } } }), true, rs.photo_id);
          };
          return <Wheel key={key} label={label} value={grade.color_balance[key]} defaultValue={auto} onChange={send} onReset={key === "midtones" ? undefined : reset} />;
        })}
      </WheelGroup>
      <SaturationSlider store={store} rs={rs} node={node} />
    </Card>
  );
}

/**
 * Colour: White balance (light presets, Pick white, Kelvin, Tint), Colour balance (three wheels and Saturation) and Colour warp (the
 * chroma plane with its pins and Pick colour). A picked white shows here with the EDITED tag.
 */
export const ColorTab = memo(function ColorTab({ store, photo, rs, pickingWhite = false, onPickWhite, pickingColor, onPickColor }: ColorTabProps) {
  return (
    <>
      <WhiteBalanceCard store={store} rs={rs} pickingWhite={pickingWhite} onPickWhite={onPickWhite} />
      <ColourBalanceCard store={store} rs={rs} />
      <ColorWarpCard store={store} photo={photo} rs={rs} pickingColor={pickingColor} onPickColor={onPickColor} />
    </>
  );
});
