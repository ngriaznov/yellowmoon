/**
 * The Photos screen of `design/mockups/photos.html`, rebuilt from the component library with the
 * mockup's data. It is not the app's Photos screen (that is task M3-06) but a reference that
 * proves the components reproduce the mockup: `e2e/components.spec.ts` compares its rendering
 * with the mockup's. Open it at `/components.html?replica=photos&tab=light`.
 */
import { useMemo, useState } from "react";
import type { ComponentProps, ReactNode } from "react";
import type { CurvePoint } from "../api/types";
import {
  AutoRow,
  AutoToggle,
  Button,
  Card,
  ChoiceList,
  CurvePlot,
  CURVE_CHANNEL_LABELS,
  Field,
  Filmstrip,
  formats,
  Histogram,
  InfoRow,
  KeyHint,
  LinkButton,
  LookTile,
  MoonMark,
  PhotoCell,
  ResetLink,
  RowNote,
  Rows,
  SearchField,
  Segmented,
  Slider,
  SwitchRow,
  TabBar,
  tabPanelProps,
  TitleBlock,
  UpdatePill,
  useTabShortcuts,
  Wheel,
  WheelGroup,
} from "../components";
import type { CurveChannel, TabId } from "../components";
import type { ColorBalancePoint } from "../api/types";
import { MOCK_LENSES } from "../api.mock";
import { SCENES } from "../fixtures/scenes";
import { CURVE_DEMO_POINTS, DEMO_LOOKS, demoHistogram, gradeScene, lisbonPhotos, logScene, lookThumbnail, sceneDataUrl } from "./demoData";

interface SliderDef {
  id: string;
  label: string;
  min: number;
  max: number;
  step: number;
  def: number;
  fmt: (v: number) => string;
  start?: number;
}

const S = (id: string, label: string, min: number, max: number, step: number, def: number, fmt: (v: number) => string, start?: number): SliderDef => ({
  id,
  label,
  min,
  max,
  step,
  def,
  fmt,
  start,
});

const ALL: Record<string, SliderDef> = {};
const reg = (...defs: SliderDef[]): SliderDef[] => {
  for (const d of defs) ALL[d.id] = d;
  return defs;
};

const INTENSITY = reg(S("int", "Intensity", 0, 100, 1, 100, formats.percent));
const EXPOSURE = reg(
  S("exp", "Exposure", -4, 4, 0.01, 0.3, formats.ev),
  S("con", "Contrast", -100, 100, 1, 0, formats.signed),
  S("bri", "Brightness", -100, 100, 1, 0, formats.signed),
);
const HDR = reg(
  S("hi", "Highlights", -100, 100, 1, 0, formats.signed, -18),
  S("sh", "Shadows", -100, 100, 1, 0, formats.signed, 12),
  S("wh", "White", -100, 100, 1, 0, formats.signed),
  S("bl", "Black", -100, 100, 1, 0, formats.signed),
);
const WB = reg(S("kel", "Kelvin", 2000, 11000, 50, 5450, formats.kelvin), S("tint", "Tint", -50, 50, 1, 0, formats.signed));
const SAT = reg(S("sat", "Saturation", -100, 100, 1, 0, formats.signed, 6));
const SHARP = reg(S("amt", "Amount", 0, 300, 1, 120, formats.integer), S("rad", "Radius", 0.3, 2.5, 0.1, 0.8, formats.px1));
const NR = reg(
  S("nrl", "Luminance", 0, 100, 1, 30, formats.integer),
  S("nrc", "Colour", 0, 100, 1, 50, formats.integer),
  S("nrd", "Detail", 0, 100, 1, 50, formats.integer),
);
const LENS_AMOUNTS = reg(
  S("fla", "Flares", 0, 100, 1, 0, formats.integer),
  S("edg", "Edges", 0, 100, 1, 0, formats.integer),
  S("ana", "Anamorphic", 0, 100, 1, 0, formats.integer),
);
const ROLL = reg(S("hro", "Highlights", 0, 1, 0.01, 1, formats.fixed2), S("lro", "Low light", 0, 1, 0.01, 0.5, formats.fixed2));

const LIGHTS = [
  { value: "auto", label: "Auto" },
  { value: "day", label: "Day" },
  { value: "shade", label: "Shade" },
  { value: "tung", label: "Tung." },
  { value: "led", label: "LED" },
  { value: "fluo", label: "Fluor." },
] as const;

const WHEEL_DEFAULTS: Record<"shadows" | "midtones" | "highlights", ColorBalancePoint> = {
  // The mockup's puck positions, as points of the unit disc (percent of the wheel from its centre).
  shadows: [-0.08, -0.12],
  midtones: [0, 0],
  highlights: [0.12, 0.12],
};

const SWITCHES = [
  ["neural", "Neural demosaic"],
  ["hlr", "Highlight recovery"],
  ["lens", "Lens corrections from camera"],
  ["ca", "Chromatic aberration"],
] as const;

export interface PhotosReplicaProps {
  initialTab?: TabId;
  initialBefore?: boolean;
  initialQuery?: string;
  initialLook?: string;
  initialCurveChannel?: CurveChannel;
}

const initialValues = (): Record<string, number> => Object.fromEntries(Object.values(ALL).map((d) => [d.id, d.start ?? d.def]));

/** The Photos screen as the mockup draws it. */
export function PhotosReplica({
  initialTab = "look",
  initialBefore = false,
  initialQuery = "",
  initialLook = "neutral",
  initialCurveChannel = "luma",
}: PhotosReplicaProps) {
  const [tab, setTab] = useState<TabId>(initialTab);
  const [before, setBefore] = useState(initialBefore);
  const [current, setCurrent] = useState(3);
  const [picked, setPicked] = useState<number[]>([2, 3, 4]);
  const [vals, setVals] = useState<Record<string, number>>(initialValues);
  const [light, setLight] = useState<string>("auto");
  const [look, setLook] = useState(initialLook);
  const [scope, setScope] = useState<"this" | "picked" | "all">("all");
  const [query, setQuery] = useState(initialQuery);
  const [channel, setChannel] = useState<CurveChannel>(initialCurveChannel);
  const [curves, setCurves] = useState<Record<CurveChannel, CurvePoint[]>>({
    luma: CURVE_DEMO_POINTS.luma,
    red: CURVE_DEMO_POINTS.red,
    green: CURVE_DEMO_POINTS.green,
    blue: CURVE_DEMO_POINTS.blue,
  });
  const [wheels, setWheels] = useState(WHEEL_DEFAULTS);
  const [lens, setLens] = useState<string>("default");
  const [switches, setSwitches] = useState<Record<string, boolean>>({ neural: true, hlr: true, lens: true, ca: true });
  useTabShortcuts(setTab);

  const scene = SCENES[current] ?? SCENES[3]!;
  const histogram = useMemo(() => demoHistogram(), []);
  const lookDef = DEMO_LOOKS.find((l) => l.id === look) ?? DEMO_LOOKS[0]!;
  const intensity = vals.int ?? 100;
  const photos = useMemo(() => lisbonPhotos({ picked }), [picked]);

  const setVal = (id: string, v: number): void => setVals((s) => ({ ...s, [id]: v }));
  const slider = (def: SliderDef) => (
    <Slider
      key={def.id}
      label={def.label}
      value={vals[def.id] ?? def.def}
      min={def.min}
      max={def.max}
      step={def.step}
      defaultValue={def.def}
      format={def.fmt}
      onChange={(v) => setVal(def.id, v)}
    />
  );
  const isAuto = (defs: SliderDef[]): boolean => defs.every((d) => Math.abs((vals[d.id] ?? d.def) - d.def) < 1e-9);
  const reset = (defs: SliderDef[]): void => setVals((s) => ({ ...s, ...Object.fromEntries(defs.map((d) => [d.id, d.def])) }));
  const near = (id: string): boolean => Math.abs((vals[id] ?? 0) - (ALL[id]?.def ?? 0)) < 1e-9;

  const nPicked = picked.length;
  const lookChip = `${lookDef.name} · ${Math.round(intensity)}%`;
  const q = query.trim().toLowerCase();
  const groups = [
    { id: "builtin", label: "Built in" },
    { id: "arri", label: "ARRI looks · imported" },
  ]
    .map((g) => {
      const inGroup = DEMO_LOOKS.filter((l) => l.group === g.id);
      const shown = inGroup.filter((l) => !q || l.name.toLowerCase().includes(q));
      return { ...g, shown, count: q ? `${shown.length} of ${inGroup.length}` : String(inGroup.length) };
    })
    .filter((g) => g.shown.length > 0);

  // The replica matches design/mockups, whose Before is still the camera log (docs/11: the app shows the camera default).
  const photoStyle = before ? logScene(scene) : gradeScene(scene, lookDef.grade);
  const photoSrc = sceneDataUrl(photoStyle, 0.09);

  const card = (title: string, body: ReactNode, props: Partial<ComponentProps<typeof Card>> = {}) => (
    <Card title={title} {...props}>
      {body}
    </Card>
  );

  let panel: ReactNode;
  switch (tab) {
    case "look":
      panel = (
        <>
          {card(
            "Automatic",
            <Rows>
              <AutoRow label="Exposure" value={`${formats.ev(vals.exp ?? 0.3)} EV`} isAuto={near("exp")} tabLabel="Light" onOpen={() => setTab("light")} />
              <AutoRow
                label="White balance"
                value={`${vals.kel} K · tint ${vals.tint === 0 ? "0" : formats.signed(vals.tint ?? 0)}`}
                isAuto={near("kel") && near("tint") && light === "auto"}
                tabLabel="Colour"
                onOpen={() => setTab("color")}
              />
              <AutoRow label="Noise" value={`Luminance ${Math.round(vals.nrl ?? 30)}`} isAuto={near("nrl") && near("nrc") && near("nrd")} tabLabel="Detail" onOpen={() => setTab("detail")} />
              <AutoRow label="Camera" value="A1 · A7R IV profile" isAuto tabLabel="RAW" onOpen={() => setTab("raw")} />
            </Rows>,
            { action: <ResetLink label="Re-run" size="card" onClick={() => reset([...EXPOSURE.slice(0, 1), ...WB, ...NR])} /> },
          )}
          {card(
            "Look",
            <>
              {INTENSITY.map(slider)}
              <Field label="Apply to">
                <Segmented
                  label="Apply look to"
                  value={scope}
                  onChange={setScope}
                  options={[
                    { value: "this", label: "This photo" },
                    { value: "picked", label: `Picked · ${nPicked}` },
                    { value: "all", label: `All · ${SCENES.length}` },
                  ]}
                />
              </Field>
              <div style={{ marginTop: 16 }}>
                <div style={{ marginBottom: 16 }}>
                  <SearchField value={query} onChange={setQuery} label="Search looks" />
                </div>
                {groups.map((g) => (
                  <div className="ym-look-group" key={g.id}>
                    <div className="ym-look-group__header">
                      <span>{g.label}</span>
                      <span className="ym-look-group__count">{g.count}</span>
                    </div>
                    <div className="ym-look-grid">
                      {g.shown.map((l) => (
                        <LookTile
                          key={l.id}
                          name={l.name}
                          src={lookThumbnail(scene.name, l)}
                          index={DEMO_LOOKS.indexOf(l) + 1}
                          isDefault={l.id === "neutral"}
                          selected={l.id === look}
                          onSelect={() => setLook(l.id)}
                        />
                      ))}
                    </div>
                  </div>
                ))}
                {groups.length === 0 ? <p style={{ margin: "0 0 16px", fontSize: 14 }}>No looks match “{query}”.</p> : null}
                <LinkButton>Import ARRI looks →</LinkButton>
              </div>
            </>,
            { action: <ResetLink size="card" onClick={() => { setLook("neutral"); reset(INTENSITY); }} /> },
          )}
        </>
      );
      break;
    case "lens":
      panel = (
        <>
          {card("Lens", <ChoiceList label="Lens" value={lens} onChange={setLens} options={MOCK_LENSES.map((l) => ({ value: l.kind, name: l.name, line: l.line }))} />)}
          {card("Amounts", LENS_AMOUNTS.map(slider), { action: <ResetLink size="card" onClick={() => reset(LENS_AMOUNTS)} /> })}
        </>
      );
      break;
    case "light":
      panel = (
        <>
          <Histogram data={histogram} left="R 182  G 164  B 140" right="ISO 400 · 1/125 · f/2.8" />
          {card("Exposure", EXPOSURE.map(slider), {
            tag: <AutoToggle isAuto={isAuto(EXPOSURE)} onAuto={() => reset(EXPOSURE)} />,
            action: <ResetLink size="card" onClick={() => reset(EXPOSURE)} />,
          })}
          {card("High dynamic range", HDR.map(slider), { action: <ResetLink size="card" onClick={() => reset(HDR)} /> })}
        </>
      );
      break;
    case "color":
      panel = (
        <>
          {card(
            "White balance",
            <>
              <div style={{ marginBottom: 12 }}>
                <Segmented label="Light type" value={light} onChange={setLight} options={LIGHTS.map((l) => ({ value: l.value, label: l.label }))} />
              </div>
              {WB.map(slider)}
            </>,
            {
              tag: <AutoToggle isAuto={isAuto(WB) && light === "auto"} onAuto={() => { reset(WB); setLight("auto"); }} />,
              action: <ResetLink size="card" onClick={() => { reset(WB); setLight("auto"); }} />,
            },
          )}
          {card(
            "Colour balance",
            <>
              <WheelGroup>
                {(["shadows", "midtones", "highlights"] as const).map((key) => (
                  <Wheel
                    key={key}
                    label={key[0]!.toUpperCase() + key.slice(1)}
                    value={wheels[key]}
                    onChange={(v) => setWheels((s) => ({ ...s, [key]: v }))}
                  />
                ))}
              </WheelGroup>
              {SAT.map(slider)}
            </>,
            { action: <ResetLink size="card" onClick={() => { reset(SAT); setWheels({ shadows: [0, 0], midtones: [0, 0], highlights: [0, 0] }); }} /> },
          )}
        </>
      );
      break;
    case "detail":
      panel = (
        <>
          {card("Sharpening", SHARP.map(slider), { action: <ResetLink size="card" onClick={() => reset(SHARP)} /> })}
          {card("Noise reduction", NR.map(slider), {
            tag: <AutoToggle isAuto={isAuto(NR)} onAuto={() => reset(NR)} />,
            action: <ResetLink size="card" onClick={() => reset(NR)} />,
          })}
        </>
      );
      break;
    case "curve":
      panel = card(
        "Curve",
        <div style={{ display: "flex", flexDirection: "column", gap: 10, marginBottom: 12 }}>
          <Segmented
            label="Curve channel"
            value={channel}
            onChange={setChannel}
            options={(Object.keys(CURVE_CHANNEL_LABELS) as CurveChannel[]).map((c) => ({ value: c, label: CURVE_CHANNEL_LABELS[c] }))}
          />
          <CurvePlot key={channel} channel={channel} points={curves[channel]} defaultSelected={channel === "green" ? 1 : 2} onChange={(points) => setCurves((s) => ({ ...s, [channel]: points }))} />
        </div>,
        { action: <ResetLink size="card" onClick={() => setCurves({ luma: [[0, 0], [1, 1]], red: [[0, 0], [1, 1]], green: [[0, 0], [1, 1]], blue: [[0, 0], [1, 1]] })} /> },
      );
      break;
    case "raw":
      panel = (
        <>
          {card(
            "Camera",
            <>
              <Rows>
                <InfoRow label="Detected" value="Sony A1 (ILCE-1)" />
                <InfoRow label="Profile" value="A7R IV" tag="stand-in" />
              </Rows>
              <RowNote>The A1 has no measured profile yet, so it borrows the A7R IV. Eleven Sony cameras have their own, and cameras that share their sensors use them.</RowNote>
            </>,
          )}
          {card(
            "Decode",
            <Rows spaced>
              {SWITCHES.map(([id, label]) => (
                <SwitchRow key={id} label={label} checked={!!switches[id]} onChange={(on) => setSwitches((s) => ({ ...s, [id]: on }))} />
              ))}
            </Rows>,
          )}
          {card(
            "Range",
            <>
              {ROLL.map(slider)}
              <RowNote>Highlights moves the brightest tones the camera recorded: 0 keeps them where they were, 1 takes them to white. Low light softens the noise near black.</RowNote>
            </>,
            { action: <ResetLink size="card" onClick={() => reset(ROLL)} /> },
          )}
        </>
      );
      break;
  }

  return (
    <div className="rp-screen rp-photos">
      <header className="ym-header rp-header" style={{ gridColumn: "1 / -1", gridRow: 1 }}>
        <div className="ym-brand">
          <span className="ym-brand-title">
            <MoonMark />
            Lisbon
          </span>
          <span className="ym-brand-sub">2026-10-04 · {SCENES.length} photos</span>
        </div>
        <nav aria-label="View" className="rp-nav">
          <Button variant="header" pressed={before} onClick={() => setBefore((b) => !b)}>
            Before
          </Button>
          <Button variant="header">Crop</Button>
          <Button variant="header">Pick white</Button>
          <Button variant="header" aria-haspopup="menu" className="rp-fit">
            Fit 34%
            <span className="rp-caret" aria-hidden="true" />
          </Button>
          <span className="rp-status">Neural · 42 ms · Display P3</span>
        </nav>
        <div className="rp-header-right">
          <UpdatePill state={{ phase: "ready", version: "0.1.1" }} />
          <a className="rp-export" href="components.html?replica=export">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="square" aria-hidden="true">
              <path d="M12 3v12" />
              <path d="M6.5 9.5L12 15l5.5-5.5" />
              <path d="M4 20h16" />
            </svg>
            {nPicked > 1 ? `Export ${nPicked}` : "Export"}
            <KeyHint>⌘E</KeyHint>
          </a>
        </div>
      </header>

      <PhotoCell
        style={{ gridColumn: 1, gridRow: 2 }}
        imageSrc={photoSrc}
        imageLabel={`Photo ${scene.name}, ${lookChip}`}
        lookChip={lookChip}
        before={before}
        beforeLabel="Before · camera log"
        cameraLine={`Sony A1 · ${scene.name} · ISO 400 · 1/125 · f/2.8`}
        native={false}
      />

      <Filmstrip
        className="rp-filmstrip"
        photos={photos}
        currentId={current + 1}
        onSelect={(id) => setCurrent(Number(id) - 1)}
        onTogglePick={(id) => setPicked((p) => (p.includes(Number(id) - 1) ? p.filter((x) => x !== Number(id) - 1) : [...p, Number(id) - 1]))}
      />

      <aside aria-label="Tools" className="rp-tools">
        <TabBar value={tab} onChange={setTab} />
        <div className="rp-tools-scroll" {...tabPanelProps(tab)}>
          <TitleBlock tab={tab} />
          {panel}
        </div>
      </aside>
    </div>
  );
}

