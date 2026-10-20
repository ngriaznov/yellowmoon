import { memo } from "react";
import { Card, formats, InfoRow, ResetLink, RowNote, Rows, Slider, SwitchRow } from "../../components";
import type { CameraInfo, DecodeOptions } from "../../api/types";
import type { TabProps } from "./shared";

/** "Sony A1" gives "A1", the name the camera card's sentences use. */
export const shortBody = (body: string): string => body.replace(/^Sony\s+/i, "");

/** "Sony A1 (ILCE-1)": the body and, when the file says it, the model code. */
export function detectedLabel(camera: CameraInfo): string {
  return camera.model && camera.model !== camera.body ? `${camera.body} (${camera.model})` : camera.body;
}

/** The sentence under the Camera card's rows (`docs/11-ui.md` §Photos, "Camera card copy"). */
export function cameraNote(camera: CameraInfo): string {
  const body = shortBody(camera.body);
  switch (camera.match_kind.kind) {
    case "default":
      return `The ${body} has no measured profile yet, so it borrows the ${camera.profile}. Eleven Sony cameras have their own, and cameras that share their sensors use them.`;
    case "sensor_family":
      return `The ${body} shares its sensor with the ${shortBody(camera.match_kind.via)}, so it uses that profile.`;
    case "exact":
      return `The ${body} has its own measured profile.`;
  }
}

/** The note for a Lossless M or S file (`docs/04-raw-decoding.md`). */
export const REDUCED_NOTE =
  "This file was saved in the camera as Lossless M or S. The camera has already processed it, so colour emulation is approximate. Use Lossless L or uncompressed for best results.";

/** The note under the Range card's sliders (`docs/11-ui.md` §Photos, "Range card"). */
export const RANGE_NOTE =
  "Highlights moves the brightest tones the camera recorded: 0 keeps them where they were, 1 takes them to white. Low light softens the noise near black.";

const SWITCHES: ReadonlyArray<[keyof DecodeOptions, string]> = [
  ["neural_demosaic", "Neural demosaic"],
  ["highlight_recovery", "Highlight recovery"],
  ["lens_corrections", "Lens corrections from camera"],
  ["chromatic_aberration", "Chromatic aberration"],
];

/** RAW: the camera and its profile, the decode switches and the range sliders. */
export const RawTab = memo(function RawTab({ store, photo, rs }: TabProps) {
  const { camera } = photo;
  const { decode, rolloff } = rs.settings;
  return (
    <>
      <Card title="Camera">
        <Rows>
          <InfoRow label="Detected" value={detectedLabel(camera)} />
          <InfoRow label="Profile" value={camera.profile} tag={camera.match_kind.kind === "exact" ? undefined : "stand-in"} />
          {photo.reduced ? <InfoRow label="Saved as" value="Lossless M or S" tag="reduced" /> : null}
        </Rows>
        <RowNote>{cameraNote(camera)}</RowNote>
        {photo.reduced ? <RowNote>{REDUCED_NOTE}</RowNote> : null}
      </Card>
      <Card title="Decode">
        <Rows spaced>
          {SWITCHES.map(([key, label]) => (
            <SwitchRow
              key={key}
              label={label}
              checked={decode[key]}
              onChange={(on) => void store.update({ decode: { ...decode, [key]: on } }, true)}
            />
          ))}
        </Rows>
      </Card>
      <Card title="Range" action={<ResetLink size="card" onClick={() => void store.update({ rolloff: { highlights: 1, low_light: 0.5 } })} />}>
        <Slider
          label="Highlights"
          value={rolloff.highlights}
          min={0}
          max={1}
          step={0.01}
          defaultValue={1}
          format={formats.fixed2}
          onChange={(highlights, final) => void store.update({ rolloff: { ...rolloff, highlights } }, final, rs.photo_id)}
        />
        <Slider
          label="Low light"
          value={rolloff.low_light}
          min={0}
          max={1}
          step={0.01}
          defaultValue={0.5}
          format={formats.fixed2}
          onChange={(low_light, final) => void store.update({ rolloff: { ...rolloff, low_light } }, final, rs.photo_id)}
        />
        <RowNote>{RANGE_NOTE}</RowNote>
      </Card>
    </>
  );
});
