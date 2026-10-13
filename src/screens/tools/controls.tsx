import { formats, Slider } from "../../components";
import type { ResolvedSettings } from "../../api/types";
import type { AppStore } from "../../state/store";
import { AUTO, manualOrAuto } from "./shared";

/**
 * Controls that live on two tabs. Colour › White balance › Kelvin, Light › Exposure › Contrast and
 * Colour › Colour balance › Saturation are also the Temperature, Contrast and Saturation rows of
 * the Look tab's Tone card (owner's decision of 2026-10-13, `docs/16`). They are one setting each:
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
  void store.update({ white_balance: manualOrAuto({ kelvin, tint }, { kelvin: estimates.kelvin, tint: estimates.tint }) }, final);
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

/** Contrast, -100 to 100, default 0 (Light › Exposure and Look › Tone). */
export function ContrastSlider({ store, rs }: ControlProps) {
  return (
    <Slider
      label="Contrast"
      value={rs.values.contrast}
      min={-100}
      max={100}
      defaultValue={0}
      format={formats.signed}
      onChange={(contrast, final) => void store.update({ contrast }, final)}
    />
  );
}

/** Saturation, -100 to 100, default 0 (Colour › Colour balance and Look › Tone). It is subtractive (`docs/02` C9). */
export function SaturationSlider({ store, rs }: ControlProps) {
  return (
    <Slider
      label="Saturation"
      value={rs.values.saturation}
      min={-100}
      max={100}
      defaultValue={0}
      format={formats.signed}
      onChange={(saturation, final) => void store.update({ saturation }, final)}
    />
  );
}
