import { useEffect, useState } from "react";
import { ToggleGroup, ToggleGroupItem } from "@/components/glass/toggle-group";
import { Button, Card, formats, MINUS, ResetLink, RowNote, Slider, WarpPlane, WARP_MAX_PINS } from "../../components";
import type { WarpTool } from "../../components";
import type { ColorWarpPin } from "../../api/types";
import { useAppState } from "../../state/context";
import type { TabProps } from "./shared";

/** What the card says while no pin is selected. */
export const WARP_HINT = "Drag a colour to move it. Lock holds a colour in place.";
/** What it says at sixteen pins, when a press on empty plane adds nothing. */
export const WARP_LIMIT_HINT = `${WARP_MAX_PINS} pins at most.`;

/** A pivot in stops: "+1.0 EV", "−2.5 EV", "0.0 EV". */
export const formatPivot = (value: number): string => {
  const text = Math.abs(value).toFixed(1);
  return `${text === "0.0" ? "" : value > 0 ? "+" : MINUS}${text} EV`;
};
/** An exposure in stops: "+0.30 EV", "0.00 EV". */
export const formatWarpExposure = (value: number): string => `${formats.ev(value)} EV`;

export interface ColorWarpCardProps extends TabProps {
  /** Pick colour is on: a click on the photo adds a pin (the crosshair layer of Pick white, `Photos`). */
  pickingColor?: boolean;
  /** Turns Pick colour on or off. Without it the toggle does nothing. */
  onPickColor?: (on: boolean) => void;
}

/**
 * Colour › Colour warp (`docs/11-ui.md` §Colour warp, `docs/02` C9b): the tools (Move or Lock, Pick colour, Delete pin), the chroma
 * plane with its pins (`WarpPlane`), and the selected pin's sliders, or one line of help while none is selected. The pins are one
 * setting, `color_warp`: every change sends the whole list through `store.update`, a drag one patch per frame and a final one, so a
 * drag is one undo step. The selected pin is the store's `warpPin`, which is window state and not a setting.
 */
export function ColorWarpCard({ store, photo, rs, pickingColor = false, onPickColor }: ColorWarpCardProps) {
  const pins = rs.settings.color_warp.pins;
  const stored = useAppState((s) => s.warpPin);
  const selected = stored !== null && stored < pins.length ? stored : null;
  const scatter = useAppState((s) => s.chromaScatter);
  const [tool, setTool] = useState<WarpTool>("move");

  // The photo's colours behind the plane follow the frames the viewport draws, at most five times a second (`store.watchChromaScatter`).
  useEffect(() => store.watchChromaScatter(), [store, photo.id]);

  const send = (next: ColorWarpPin[], final: boolean): void => void store.update({ color_warp: { pins: next } }, final, rs.photo_id);
  const pin = selected === null ? undefined : pins[selected];
  /** Changes a field of the selected pin. */
  const setPin = (patch: Partial<ColorWarpPin>, final: boolean): void => {
    if (selected === null || !pin) return;
    send(
      pins.map((p, i) => (i === selected ? { ...p, ...patch } : p)),
      final,
    );
  };
  const removeSelected = (): void => {
    if (selected === null) return;
    store.selectWarpPin(null);
    send(
      pins.filter((_, i) => i !== selected),
      true,
    );
  };
  const reset = (): void => {
    store.selectWarpPin(null);
    void store.update({ color_warp: { pins: [] } });
  };

  return (
    <Card title="Colour warp" action={pins.length > 0 ? <ResetLink size="card" onClick={reset} /> : null}>
      <div className="ym-warp">
        <div className="ym-warp__tools">
          <ToggleGroup
            type="single"
            variant="outline"
            size="sm"
            aria-label="Pin tool"
            value={tool}
            onValueChange={(value) => {
              if (value === "move" || value === "lock") setTool(value);
            }}
          >
            <ToggleGroupItem value="move" className="px-3">
              Move
            </ToggleGroupItem>
            <ToggleGroupItem value="lock" className="px-3">
              Lock
            </ToggleGroupItem>
          </ToggleGroup>
          <Button variant="header" size="sm" pressed={pickingColor} onClick={() => onPickColor?.(!pickingColor)}>
            Pick colour
          </Button>
          <Button variant="secondary" size="sm" className="ym-warp__delete" disabled={selected === null} onClick={removeSelected}>
            Delete pin
          </Button>
        </div>
        <WarpPlane
          pins={pins}
          selected={selected}
          onSelect={(index) => store.selectWarpPin(index)}
          onChange={send}
          tool={tool}
          scatter={scatter?.points ?? null}
        />
      </div>
      {!pin ? (
        <RowNote>{pins.length >= WARP_MAX_PINS ? WARP_LIMIT_HINT : WARP_HINT}</RowNote>
      ) : (
        <div className="ym-warp__rows">
          <Slider
            label="Range"
            value={Math.round(pin.range * 100)}
            min={1}
            max={25}
            defaultValue={4}
            format={formats.integer}
            onChange={(v, final) => setPin({ range: v / 100 }, final)}
          />
          {pin.locked ? null : (
            <>
              <Slider
                label="Shadows"
                value={pin.shadows * 100}
                min={0}
                max={100}
                defaultValue={100}
                format={formats.percent}
                onChange={(v, final) => setPin({ shadows: v / 100 }, final)}
              />
              <Slider
                label="Highlights"
                value={pin.highlights * 100}
                min={0}
                max={100}
                defaultValue={100}
                format={formats.percent}
                onChange={(v, final) => setPin({ highlights: v / 100 }, final)}
              />
              <Slider
                label="Pivot"
                value={pin.pivot_ev}
                min={-6}
                max={6}
                step={0.1}
                defaultValue={0}
                format={formatPivot}
                onChange={(v, final) => setPin({ pivot_ev: v }, final)}
              />
              <Slider
                label="Exposure"
                value={pin.exposure_ev}
                min={-2}
                max={2}
                step={0.01}
                defaultValue={0}
                format={formatWarpExposure}
                onChange={(v, final) => setPin({ exposure_ev: v }, final)}
              />
            </>
          )}
        </div>
      )}
    </Card>
  );
}
