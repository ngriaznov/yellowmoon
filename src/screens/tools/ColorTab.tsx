import { AutoToggle, Card, formats, ResetLink, Segmented, Slider, Tag, Wheel, WheelGroup } from "../../components";
import type { ColorBalance, ColorBalancePoint, LightSource } from "../../api/types";
import { KelvinSlider, SaturationSlider, setWhiteBalance, whiteBalanceState, whiteBalanceToAuto } from "./controls";
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

const ZERO_BALANCE: ColorBalance = { shadows: [0, 0], midtones: [0, 0], highlights: [0, 0] };

/**
 * Colour: White balance (light presets, Kelvin, Tint) and Colour balance (three wheels and
 * Saturation). A picked white shows here with the EDITED tag.
 */
export function ColorTab({ store, rs }: TabProps) {
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
                void store.update({ color_balance: { ...settings.color_balance, [key]: point } }, final)
              }
            />
          ))}
        </WheelGroup>
        <SaturationSlider store={store} rs={rs} />
      </Card>
    </>
  );
}
