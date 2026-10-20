import { memo } from "react";
import { AutoToggle, Card, formats, ResetLink, Segmented, Slider, Tag, Wheel, WheelGroup } from "../../components";
import type { ColorBalance, ColorBalancePoint, LightSource } from "../../api/types";
import { ColorWarpCard } from "./ColorWarpCard";
import type { ColorWarpCardProps } from "./ColorWarpCard";
import { KelvinSlider, SaturationSlider, setWhiteBalance, whiteBalanceState, whiteBalanceToAuto } from "./controls";
import { AUTO, pointOrAuto } from "./shared";

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

/** The Colour tab's props: those of every tab, and Pick colour's state, which the Photos screen keeps (it owns the crosshair layer). */
export type ColorTabProps = ColorWarpCardProps;

/**
 * Colour: White balance (light presets, Kelvin, Tint), Colour balance (three wheels and Saturation) and Colour warp (the chroma plane
 * with its pins and Pick colour). A picked white shows here with the EDITED tag.
 */
export const ColorTab = memo(function ColorTab({ store, photo, rs, pickingColor, onPickColor }: ColorTabProps) {
  const { settings, values, estimates } = rs;
  const { picked, isAuto: whiteIsAuto } = whiteBalanceState(rs);
  const toAuto = (): void => whiteBalanceToAuto(store);

  return (
    <>
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
            value={settings.light_source}
            onChange={(light_source) => void store.update({ light_source }, true)}
            options={LIGHT_OPTIONS.map(({ value, label, ariaLabel }) => ({ value, label, ariaLabel }))}
          />
        </div>
        <KelvinSlider store={store} rs={rs} />
        <Slider
          label="Tint"
          value={values.tint}
          min={-50}
          max={50}
          defaultValue={estimates.tint}
          format={formats.signed}
          onChange={(tint, final) => setWhiteBalance(store, rs, values.kelvin, tint, final)}
          onReset={() => setWhiteBalance(store, rs, values.kelvin, estimates.tint, true)}
        />
      </Card>
      <Card
        title="Colour balance"
        action={<ResetLink size="card" onClick={() => void store.update({ color_balance: RESET_BALANCE, saturation: AUTO }, true, rs.photo_id)} />}
      >
        <WheelGroup>
          {WHEELS.map(([key, label]) => {
            // The shadows and highlights wheels are Auto or a point (`docs/19` section 5): the puck shows the point in effect, a point
            // dropped on the Auto point is sent as Auto, and Reset (double-click, Home) sends Auto. The midtones wheel is a point.
            const auto: ColorBalancePoint = key === "shadows" ? estimates.balance_shadows : key === "highlights" ? estimates.balance_highlights : [0, 0];
            const send = (point: ColorBalancePoint, final: boolean): void => {
              const next: ColorBalance =
                key === "midtones" ? { ...settings.color_balance, midtones: point } : { ...settings.color_balance, [key]: pointOrAuto(point, auto) };
              void store.update({ color_balance: next }, final, rs.photo_id);
            };
            return (
              <Wheel
                key={key}
                label={label}
                value={values.color_balance[key]}
                defaultValue={auto}
                onChange={send}
                onReset={key === "midtones" ? undefined : () => void store.update({ color_balance: { ...settings.color_balance, [key]: AUTO } }, true, rs.photo_id)}
              />
            );
          })}
        </WheelGroup>
        <SaturationSlider store={store} rs={rs} />
      </Card>
      <ColorWarpCard store={store} photo={photo} rs={rs} pickingColor={pickingColor} onPickColor={onPickColor} />
    </>
  );
});
