import { memo } from "react";
import { AutoToggle, Card, formats, ResetLink, Segmented, Slider, Tag, Wheel, WheelGroup } from "../../components";
import type { ColorBalance, ColorBalancePoint, LightSource } from "../../api/types";
import { ColorWarpCard } from "./ColorWarpCard";
import type { ColorWarpCardProps } from "./ColorWarpCard";
import { KelvinSlider, SaturationSlider, setWhiteBalance, whiteBalanceState, whiteBalanceToAuto } from "./controls";

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

const ZERO_BALANCE: ColorBalance = { shadows: [0, 0], midtones: [0, 0], highlights: [0, 0] };

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
        action={<ResetLink size="card" onClick={() => void store.update({ color_balance: ZERO_BALANCE, saturation: 0 })} />}
      >
        <WheelGroup>
          {WHEELS.map(([key, label]) => (
            <Wheel
              key={key}
              label={label}
              value={settings.color_balance[key]}
              onChange={(point: ColorBalancePoint, final: boolean) =>
                void store.update({ color_balance: { ...settings.color_balance, [key]: point } }, final, rs.photo_id)
              }
            />
          ))}
        </WheelGroup>
        <SaturationSlider store={store} rs={rs} />
      </Card>
      <ColorWarpCard store={store} photo={photo} rs={rs} pickingColor={pickingColor} onPickColor={onPickColor} />
    </>
  );
});
