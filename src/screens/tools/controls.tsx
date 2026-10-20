import { formats, Slider } from "../../components";
import type { ResolvedSettings } from "../../api/types";
import type { AppStore } from "../../state/store";
import { AUTO, manualOrAuto } from "./shared";

/**
 * Controls that live on two tabs. Colour › White balance › Kelvin, Light › Exposure › Contrast and
 * Colour › Colour balance › Saturation are also the Temperature, Contrast and Saturation rows of
 * the Look tab's Tone card (owner's decision of 2026-10-13, `docs/16`). All three are Auto or a
 * value; contrast and saturation since Apple Auto (`docs/19` section 5). They are one setting each:
 * both copies are built from the code below, read the same resolved value and send the same patch
 * through `store.update`, so moving one moves the other, and undo, redo and Reset treat them as
 * any other field.
 */

/** What the White balance card and the Tone card both need to know about the white balance. */
export interface WhiteBalanceState {
  /** A white was picked from the photo (the EDITED tag until Reset). */
  picked: boolean;
  /** White balance counts as automatic only when it is Auto, the light source is Auto and no white was picked (`docs/11` Auto semantics). */
  isAuto: boolean;
}

/** Whether the white balance is automatic, and whether a white was picked. */
export function whiteBalanceState(rs: ResolvedSettings): WhiteBalanceState {
  const picked = rs.settings.white_pick !== null;
  return { picked, isAuto: rs.auto.white_balance && rs.auto.light_source && !picked };
}

/** Back to Auto: white balance, light preset and picked white (the AUTO button and the cards' Reset). */
export function whiteBalanceToAuto(store: AppStore): void {
  void store.update({ white_balance: AUTO, light_source: "auto", white_pick: null });
}

/**
 * A manual white balance at `kelvin` and `tint`. A pair equal to what Automatic resolves to *is*
 * Automatic, so dropping on the detent sends `auto`. Moving only Kelvin on an Auto white balance
 * keeps the Auto tint, which is what moving one slider means.
 */
export function setWhiteBalance(store: AppStore, rs: ResolvedSettings, kelvin: number, tint: number, final: boolean): void {
  const { estimates } = rs;
  void store.update({ white_balance: manualOrAuto({ kelvin, tint }, { kelvin: estimates.kelvin, tint: estimates.tint }) }, final, rs.photo_id);
}

interface ControlProps {
  store: AppStore;
  rs: ResolvedSettings;
}

/**
 * The Kelvin slider: 2000 to 11000 K in steps of 50, with its detent on the Auto Kelvin. Reset goes back to
 * the estimated light (Auto when the tint is Auto too), not to a manual copy of it. `label` is "Kelvin" on
 * the Colour tab and "Temperature" on the Look tab's Tone card.
 */
export function KelvinSlider({ store, rs, label = "Kelvin" }: ControlProps & { label?: string }) {
  const { values, estimates } = rs;
  return (
    <Slider
      label={label}
      value={values.kelvin}
      min={2000}
      max={11000}
      step={50}
      defaultValue={estimates.kelvin}
      format={formats.kelvin}
      onChange={(kelvin, final) => setWhiteBalance(store, rs, kelvin, values.tint, final)}
      onReset={() =>
        void store.update({
          white_balance: manualOrAuto({ kelvin: estimates.kelvin, tint: values.tint }, { kelvin: estimates.kelvin, tint: estimates.tint }),
          light_source: "auto",
          white_pick: null,
        })
      }
    />
  );
}

/** The settings that are Auto or a value from −100 to 100 (`PercentSetting`, `docs/19` section 5). */
export type PercentField = "contrast" | "saturation" | "highlights" | "shadows" | "black";

/**
 * Contrast, Saturation, Highlights, Shadows or Black: −100 to 100, Auto or a value (`docs/19` section 10). They behave like the
 * Exposure slider: the detent and Reset are at the photo's Auto value (0, or Apple Auto's answer, `AutoEstimates.contrast`, …), a
 * value equal to it is sent as Auto (`manualOrAuto`) and any other value as Manual, and Reset sends Auto, not a manual copy of the
 * Auto value.
 */
export function PercentSlider({ store, rs, field, label }: ControlProps & { field: PercentField; label: string }) {
  const auto = rs.estimates[field];
  return (
    <Slider
      label={label}
      value={rs.values[field]}
      min={-100}
      max={100}
      defaultValue={auto}
      format={formats.signed}
      onChange={(value, final) => void store.update({ [field]: manualOrAuto({ value }, { value: auto }) }, final, rs.photo_id)}
      onReset={() => void store.update({ [field]: AUTO }, true, rs.photo_id)}
    />
  );
}

/** Contrast (Light › Exposure and Look › Tone): Auto or −100…100, the detent at the Auto value. */
export function ContrastSlider({ store, rs }: ControlProps) {
  return <PercentSlider store={store} rs={rs} field="contrast" label="Contrast" />;
}

/** Saturation (Colour › Colour balance and Look › Tone): Auto or −100…100, the detent at the Auto value. It keeps the luminance (`docs/02` C9). */
export function SaturationSlider({ store, rs }: ControlProps) {
  return <PercentSlider store={store} rs={rs} field="saturation" label="Saturation" />;
}
