/**
 * Every component in every state (docs/15 M3-01: "a Storybook-free `components.html` dev page shows
 * every component state"). Run `npm run dev` and open `/components.html`. Nothing here talks to
 * the core; the interactive demos keep their own state and print what the components report.
 */
import { useState } from "react";
import type { ReactNode } from "react";
import type { CurvePoint, UpdateState } from "../api/types";
import {
  AutoRow,
  AutoToggle,
  Button,
  Card,
  ChoiceList,
  CURVE_CHANNEL_LABELS,
  CurvePlot,
  DevelopList,
  DropZone,
  ExportQueueList,
  Field,
  Filmstrip,
  filterArwPaths,
  formats,
  Histogram,
  InfoRow,
  KeyHint,
  LinkButton,
  LookTile,
  MAX_CURVE_POINTS,
  MoonMark,
  PhotoCell,
  ProgressBar,
  ResetLink,
  RowNote,
  Rows,
  SearchField,
  Segmented,
  Slider,
  StepRail,
  Switch,
  SwitchRow,
  Tag,
  TAG_LABELS,
  TabBar,
  TABS,
  TitleBlock,
  UpdatePill,
  useTabShortcuts,
  useWindowDrop,
  ValueBox,
  WarpPlane,
  Wheel,
  WheelGroup,
  lockedPin,
  movingPin,
  pinValueText,
} from "../components";
import type { CurveChannel, ExportQueueItem, TabId, TagVariant, WarpTool } from "../components";
import type { ColorBalancePoint, ColorWarpPin } from "../api/types";
import { CURVE_DEMO_POINTS, DEMO_LOOKS, demoHistogram, demoLumaHistogram, lisbonPhotos, lookThumbnail, photoUrl } from "./demoData";

// ---- Layout helpers --------------------------------------------------------------------------

function Section({ id, title, lead, children }: { id: string; title: string; lead?: string; children: ReactNode }) {
  return (
    <section id={id} className="dev-section" aria-labelledby={`${id}-h`}>
      <h2 id={`${id}-h`}>{title}</h2>
      {lead ? <p>{lead}</p> : null}
      <div className="dev-grid">{children}</div>
    </section>
  );
}

function Demo({ title, frame = "panel", tab, children }: { title: string; frame?: "panel" | "bare" | "wide" | "free"; tab?: TabId; children: ReactNode }) {
  return (
    <figure className="dev-demo">
      <figcaption>{title}</figcaption>
      <div className="dev-stage" data-frame={frame} data-tab={tab}>
        {children}
      </div>
    </figure>
  );
}

const SECTIONS = [
  ["foundations", "Foundations"],
  ["slider", "Slider"],
  ["valuebox", "ValueBox"],
  ["segmented", "Segmented"],
  ["choice", "ChoiceList"],
  ["switch", "Switch"],
  ["card", "Card"],
  ["tabbar", "TabBar"],
  ["look", "LookTile"],
  ["wheel", "Wheel"],
  ["warp", "WarpPlane"],
  ["curve", "CurvePlot"],
  ["filmstrip", "Filmstrip"],
  ["photo", "PhotoCell"],
  ["update", "UpdatePill"],
  ["queue", "Queues"],
  ["drop", "DropZone"],
  ["steps", "StepRail"],
] as const;

const TAG_VARIANTS = Object.keys(TAG_LABELS) as TagVariant[];

/** The dev page. */
export function ComponentsPage() {
  return (
    <main className="dev-page">
      <header>
        <h1>
          <MoonMark />
          Yellowmoon components
        </h1>
        <p className="dev-lead">
          Every component of <code>docs/11-ui.md</code> in every state, drawn from the tokens. The sliders, wheels and curves are live. The
          screens of the mockups rebuilt from these parts open at <a href="?replica=photos">?replica=photos</a>, <a href="?replica=start">start</a>,{" "}
          <a href="?replica=export">export</a> and <a href="?replica=wizard">wizard</a>.
        </p>
        <nav className="dev-nav" aria-label="Sections">
          {SECTIONS.map(([id, label]) => (
            <a key={id} href={`#${id}`}>
              {label}
            </a>
          ))}
        </nav>
      </header>
      <Foundations />
      <SliderSection />
      <ValueBoxSection />
      <SegmentedSection />
      <ChoiceListSection />
      <SwitchSection />
      <CardSection />
      <TabBarSection />
      <LookSection />
      <WheelSection />
      <WarpSection />
      <CurveSection />
      <FilmstripSection />
      <PhotoSection />
      <UpdateSection />
      <QueueSection />
      <DropSection />
      <StepsSection />
    </main>
  );
}

// ---- Foundations -------------------------------------------------------------------------------

function Foundations() {
  const [pressed, setPressed] = useState(true);
  const [auto, setAuto] = useState(true);
  const [autoCrop, setAutoCrop] = useState(false);
  return (
    <Section id="foundations" title="Foundations" lead="MoonMark, Tag, buttons, links, key hints and the progress bar.">
      <Demo title="MoonMark" frame="free">
        <MoonMark />
        <MoonMark size={24} />
        <MoonMark size={48} />
      </Demo>
      <Demo title="Tag · all variants" frame="free">
        {TAG_VARIANTS.map((variant) => (
          <Tag key={variant} variant={variant} />
        ))}
        <Tag variant="active">45%</Tag>
      </Demo>
      <Demo title="Tag · small, on a filmstrip thumbnail" frame="free">
        <Tag variant="cant-open" size="sm" />
      </Demo>
      <Demo title="AutoToggle · automatic, edited" frame="free">
        <AutoToggle isAuto={auto} onAuto={() => setAuto(true)} />
        <Button variant="secondary" onClick={() => setAuto((a) => !a)}>
          Toggle
        </Button>
        <AutoToggle isAuto={false} onAuto={() => undefined} />
      </Demo>
      <Demo title="AutoToggle · both ways (the Crop panel's Auto crop)" frame="free">
        <AutoToggle aria-label="Auto crop" isAuto={autoCrop} onAuto={() => setAutoCrop(true)} onOff={() => setAutoCrop(false)} />
      </Demo>
      <Demo title="Button · primary, secondary, sizes" frame="free">
        <Button>Open files…</Button>
        <Button variant="secondary">Open folder…</Button>
        <Button size="lg">Next</Button>
        <Button size="lg" variant="secondary">
          Cancel
        </Button>
      </Demo>
      <Demo title="Button · disabled keeps its focus; split with aside" frame="free">
        <Button disabled>Next</Button>
        <Button variant="secondary" disabled>
          Cancel
        </Button>
        <Button size="lg" aside="≈ 450 MB · ⌘↩">
          Export 3 photos
        </Button>
      </Demo>
      <Demo title="Button · header variant, toggle, shortcut hint" frame="free">
        <Button variant="header" pressed={pressed} onClick={() => setPressed((p) => !p)}>
          Before
        </Button>
        <Button variant="header">Crop</Button>
        <Button variant="header" shortcut="⌘O">
          Open files
        </Button>
        <Button variant="header" shortcut="⇧⌘O">
          Open folder
        </Button>
        <Button variant="header" disabled shortcut="⌘E">
          Export
        </Button>
        <KeyHint>⌘E</KeyHint>
      </Demo>
      <Demo title="LinkButton · md, sm, bold, disabled" frame="free">
        <LinkButton>Import →</LinkButton>
        <LinkButton size="sm">Choose…</LinkButton>
        <LinkButton weight="bold">Start over</LinkButton>
        <LinkButton disabled>← Back</LinkButton>
      </Demo>
      <Demo title="ResetLink · row, card, Re-run" frame="free">
        <ResetLink />
        <ResetLink size="card" />
        <ResetLink size="card" label="Re-run" />
      </Demo>
      <Demo title="ProgressBar · 0, 40, 100 · md, lg" frame="bare">
        <div style={{ display: "grid", gap: 12 }}>
          <ProgressBar value={0} label="Empty" />
          <ProgressBar value={40} label="Forty per cent" />
          <ProgressBar value={100} label="Full" />
          <ProgressBar value={46} size="lg" label="Large" />
        </div>
      </Demo>
    </Section>
  );
}

// ---- Slider ------------------------------------------------------------------------------------

function SliderSection() {
  const [exposure, setExposure] = useState(0.3);
  const [highlights, setHighlights] = useState(-18);
  const [kelvin, setKelvin] = useState(5450);
  const [quality, setQuality] = useState(92);
  const [intensity, setIntensity] = useState(100);
  const [tabs, setTabs] = useState<Record<string, number>>({ look: 60, light: -18, color: 12, detail: 80, curve: 25, raw: 0.8 });
  const [log, setLog] = useState<string[]>([]);
  const [auto, setAuto] = useState(true);
  const say = (v: number, final: boolean): void => setLog((l) => [...l.slice(-7), `${final ? "final " : "drag  "} ${v}`]);

  return (
    <Section
      id="slider"
      title="Slider"
      lead="Drag, or use the arrows (Shift ×10), PageUp/PageDown, Home and End. Double-click the track to reset. The box takes typed values. The log shows what onChange reports: one call per animation frame while dragging, then a final one."
    >
      <Demo title="Default (the thumb sits on the detent)">
        <Slider label="Exposure" value={0.3} min={-4} max={4} step={0.01} defaultValue={0.3} format={formats.ev} onChange={() => undefined} tab="light" />
      </Demo>
      <Demo title="Changed: fill between detent and thumb, tab colour box, Reset">
        <Slider label="Highlights" value={-18} min={-100} max={100} step={1} defaultValue={0} format={formats.signed} onChange={() => undefined} tab="light" />
        <Slider label="Shadows" value={12} min={-100} max={100} step={1} defaultValue={0} format={formats.signed} onChange={() => undefined} tab="light" />
      </Demo>
      <Demo title="Live · exposure with Auto (reset returns to Auto)" tab="light">
        <Slider
          label="Exposure"
          value={auto ? 0.3 : exposure}
          min={-4}
          max={4}
          step={0.01}
          defaultValue={0.3}
          changed={!auto}
          format={formats.ev}
          onChange={(v, final) => {
            setAuto(false);
            setExposure(v);
            say(v, final);
          }}
          onReset={() => {
            setAuto(true);
            setExposure(0.3);
            setLog((l) => [...l.slice(-7), "reset to Auto"]);
          }}
        />
        <div className="dev-output" aria-live="off">
          {log.join("\n") || "no changes yet"}
        </div>
      </Demo>
      <Demo title="Live · highlights, kelvin" tab="color">
        <Slider label="Highlights" value={highlights} min={-100} max={100} step={1} defaultValue={0} format={formats.signed} onChange={(v) => setHighlights(v)} />
        <Slider label="Kelvin" value={kelvin} min={2000} max={11000} step={50} defaultValue={5450} format={formats.kelvin} onChange={(v) => setKelvin(v)} />
      </Demo>
      <Demo title="Intensity · default at the right end" tab="look">
        <Slider label="Intensity" value={intensity} min={0} max={100} step={1} defaultValue={100} format={formats.percent} onChange={(v) => setIntensity(v)} />
      </Demo>
      <Demo title="No default, no reset column (Export › Quality)" tab="look">
        <Slider label="Quality" value={quality} min={50} max={100} step={1} resetColumn={false} onChange={(v) => setQuality(v)} />
      </Demo>
      <Demo title="Disabled">
        <Slider label="Contrast" value={20} min={-100} max={100} step={1} defaultValue={0} format={formats.signed} onChange={() => undefined} disabled tab="light" />
      </Demo>
      <Demo title="Changed · one per tab colour" frame="bare">
        {TABS.map((tab) => (
          <div key={tab.id} data-tab={tab.id} style={{ padding: "0 24px" }}>
            <Slider
              label={tab.label}
              value={tabs[tab.id] ?? 0}
              min={tab.id === "raw" ? 0 : -100}
              max={tab.id === "raw" ? 1 : 100}
              step={tab.id === "raw" ? 0.01 : 1}
              defaultValue={tab.id === "raw" ? 0.5 : 0}
              format={tab.id === "raw" ? formats.fixed2 : formats.signed}
              onChange={(v) => setTabs((s) => ({ ...s, [tab.id]: v }))}
            />
          </div>
        ))}
      </Demo>
    </Section>
  );
}

// ---- ValueBox ----------------------------------------------------------------------------------

function ValueBoxSection() {
  const [value, setValue] = useState(5450);
  const [committed, setCommitted] = useState("");
  return (
    <Section id="valuebox" title="ValueBox" lead="Click to type. Return commits, Esc cancels, leaving the box commits. Out-of-range values clamp (try 99999 or 100) and snap to the step.">
      <Demo title="Default and changed" frame="free">
        <ValueBox value={0} min={-100} max={100} label="Contrast value" format={formats.signed} onCommit={() => undefined} />
        <ValueBox value={-18} min={-100} max={100} label="Highlights value" format={formats.signed} changed tab="light" onCommit={() => undefined} />
      </Demo>
      <Demo title="Changed · every tab colour" frame="free">
        {TABS.map((tab) => (
          <ValueBox key={tab.id} value={12} min={-100} max={100} label={`${tab.label} value`} format={formats.signed} changed tab={tab.id} onCommit={() => undefined} />
        ))}
      </Demo>
      <Demo title="Disabled" frame="free">
        <ValueBox value={0} min={-100} max={100} label="Disabled value" format={formats.signed} disabled onCommit={() => undefined} />
      </Demo>
      <Demo title="Live · kelvin (2000–11000, step 50)" frame="free">
        <ValueBox
          value={value}
          min={2000}
          max={11000}
          step={50}
          label="Kelvin value"
          format={formats.kelvin}
          changed={value !== 5450}
          tab="color"
          onCommit={(v) => {
            setValue(v);
            setCommitted(`committed ${v}`);
          }}
        />
        <span className="dev-note">{committed}</span>
      </Demo>
    </Section>
  );
}

// ---- Segmented ---------------------------------------------------------------------------------

function SegmentedSection() {
  const [light, setLight] = useState("auto");
  const [scope, setScope] = useState("all");
  const [fmt, setFmt] = useState("jpeg");
  const [bits, setBits] = useState("16");
  return (
    <Section id="segmented" title="Segmented" lead="A radiogroup. One option is in the tab order; arrows move the selection, Home and End jump.">
      <Demo title="Light presets · six columns, 32 px">
        <Segmented
          label="Light type"
          value={light}
          onChange={setLight}
          options={[
            { value: "auto", label: "Auto" },
            { value: "day", label: "Day" },
            { value: "shade", label: "Shade" },
            { value: "tung", label: "Tung." },
            { value: "led", label: "LED" },
            { value: "fluo", label: "Fluor." },
          ]}
        />
      </Demo>
      <Demo title="Apply to · three columns, with a label">
        <Field label="Apply to">
          <Segmented
            label="Apply look to"
            value={scope}
            onChange={setScope}
            options={[
              { value: "this", label: "This photo" },
              { value: "picked", label: "Picked · 3" },
              { value: "all", label: "All · 12" },
            ]}
          />
        </Field>
      </Demo>
      <Demo title="Format · large, 36 px">
        <Segmented
          label="Format"
          size="lg"
          value={fmt}
          onChange={setFmt}
          options={[
            { value: "jpeg", label: "JPEG" },
            { value: "tiff", label: "TIFF" },
            { value: "png", label: "PNG" },
            { value: "dng", label: "DNG" },
          ]}
        />
      </Demo>
      <Demo title="Bit depth · small, 30 px">
        <Field label="Bit depth">
          <Segmented
            label="Bit depth"
            size="sm"
            value={bits}
            onChange={setBits}
            options={[
              { value: "8", label: "8-bit" },
              { value: "16", label: "16-bit" },
            ]}
          />
        </Field>
      </Demo>
      <Demo title="A disabled option">
        <Segmented
          label="Downloaded zip"
          size="lg"
          value="keep"
          onChange={() => undefined}
          options={[
            { value: "keep", label: "Keep it" },
            { value: "trash", label: "Move to Trash", disabled: true },
          ]}
        />
      </Demo>
    </Section>
  );
}

// ---- ChoiceList --------------------------------------------------------------------------------

const DEMO_CHOICES = [
  { value: "default", name: "Default", line: "Clean and modern: even sharpness, faint glow." },
  { value: "classic", name: "Classic", line: "Warm and gentle: soft glow, amber flares, smooth falloff." },
  { value: "vintage", name: "Vintage", line: "Glows wide open, lower contrast, soft edges, a little swirl." },
  { value: "petzval", name: "Petzval", line: "Very sharp centre, strong swirl, fast falloff." },
] as const;

function ChoiceListSection() {
  const [choice, setChoice] = useState<string>("classic");
  return (
    <Section id="choice" title="ChoiceList" lead="glasscn's RadioGroup of Item rows, a name with a line under it. One row is in the tab order; arrows move and choose. The chosen row is ringed in the tab colour.">
      <Demo title="The Lens card's lenses · Lens tab colour" frame="panel" tab="lens">
        <ChoiceList label="Lens" value={choice} onChange={setChoice} options={DEMO_CHOICES} />
      </Demo>
      <Demo title="In another tab's colour, and a row with no line" frame="panel" tab="light">
        <ChoiceList label="Light" value="b" onChange={() => undefined} options={[{ value: "a", name: "Plain row" }, { value: "b", name: "Chosen row", line: "A line under the name, in the quieter colour." }]} />
      </Demo>
      <Demo title="Off (the lens data are missing)" frame="panel" tab="lens">
        <ChoiceList label="Lens, off" value="default" onChange={() => undefined} options={[{ value: "default", name: "Default" }]} disabled />
      </Demo>
    </Section>
  );
}

// ---- Switch ------------------------------------------------------------------------------------

function SwitchSection() {
  const [on, setOn] = useState(true);
  return (
    <Section id="switch" title="Switch" lead="Space or Return toggles. On takes the tab colour, off is paper.">
      <Demo title="Off, on, disabled" frame="free">
        <Switch label="Off" checked={false} onChange={() => undefined} />
        <Switch label="On" checked onChange={() => undefined} />
        <Switch label="Disabled" checked disabled onChange={() => undefined} />
        <Switch label="Live" checked={on} onChange={setOn} />
      </Demo>
      <Demo title="On · every tab colour" frame="free">
        {TABS.map((tab) => (
          <Switch key={tab.id} label={`${tab.label} switch`} checked tab={tab.id} onChange={() => undefined} />
        ))}
      </Demo>
    </Section>
  );
}

// ---- Card --------------------------------------------------------------------------------------

function CardSection() {
  const [auto, setAuto] = useState(false);
  const [decode, setDecode] = useState([true, true, false, true]);
  const labels = ["Neural demosaic", "Highlight recovery", "Lens corrections from camera", "Chromatic aberration"];
  return (
    <Section id="card" title="Card" lead="A caps title with a tag and a Reset or Re-run link, rows separated by 2 px rules, 3 px rule between cards.">
      <Demo title="Tool card · AUTO toggle (automatic) and Reset" frame="bare">
        <Card title="Exposure" tab="light" tag={<AutoToggle isAuto onAuto={() => undefined} />} action={<ResetLink size="card" />}>
          <Slider label="Exposure" value={0.3} min={-4} max={4} step={0.01} defaultValue={0.3} format={formats.ev} onChange={() => undefined} />
        </Card>
      </Demo>
      <Demo title="Tool card · edited (press AUTO to restore)" frame="bare">
        <Card title="White balance" tab="color" tag={<AutoToggle isAuto={auto} onAuto={() => setAuto(true)} />} action={<ResetLink size="card" onClick={() => setAuto(true)} />}>
          <Slider
            label="Kelvin"
            value={auto ? 5450 : 6200}
            min={2000}
            max={11000}
            step={50}
            defaultValue={5450}
            format={formats.kelvin}
            onChange={() => setAuto(false)}
          />
        </Card>
      </Demo>
      <Demo title="Automatic card · AUTO and EDITED rows, Re-run" frame="bare">
        <Card title="Automatic" tab="look" action={<ResetLink size="card" label="Re-run" />}>
          <Rows>
            <AutoRow label="Exposure" value="+0.30 EV" isAuto tabLabel="Light" onOpen={() => undefined} />
            <AutoRow label="White balance" value="6200 K · tint +4" isAuto={false} tabLabel="Colour" onOpen={() => undefined} />
            <AutoRow label="Noise" value="Luminance 30" isAuto tabLabel="Detail" onOpen={() => undefined} />
            <AutoRow label="Camera" value="A1 · A7R IV profile" isAuto tabLabel="RAW" onOpen={() => undefined} />
          </Rows>
        </Card>
      </Demo>
      <Demo title="Camera card · STAND-IN and REDUCED" frame="bare">
        <Card title="Camera" tab="raw">
          <Rows>
            <InfoRow label="Detected" value="Sony A1 (ILCE-1)" />
            <InfoRow label="Profile" value="A7R IV" tag="stand-in" />
            <InfoRow label="Decode" value="Lossless M" tag="reduced" />
          </Rows>
          <RowNote>The A1 has no measured profile yet, so it borrows the A7R IV. Eleven Sony cameras have their own, and cameras that share their sensors use them.</RowNote>
        </Card>
      </Demo>
      <Demo title="Decode card · switches in the RAW colour" frame="bare">
        <Card title="Decode" tab="raw">
          <Rows spaced>
            {labels.map((label, i) => (
              <SwitchRow key={label} label={label} checked={!!decode[i]} onChange={(on) => setDecode((d) => d.map((v, j) => (j === i ? on : v)))} />
            ))}
          </Rows>
        </Card>
      </Demo>
      <Demo title="Option card (Export panel)" frame="bare">
        <Card density="option" title="Colour space">
          <Segmented label="Colour space" size="lg" value="srgb" onChange={() => undefined} options={[{ value: "srgb", label: "sRGB" }, { value: "p3", label: "Display P3" }, { value: "rec2020", label: "Rec.2020" }]} />
          <RowNote>Best for sharing and the web.</RowNote>
        </Card>
      </Demo>
      <Demo title="Search field" frame="bare">
        <SearchFieldDemo />
      </Demo>
    </Section>
  );
}

function SearchFieldDemo() {
  const [q, setQ] = useState("");
  return (
    <div style={{ padding: "12px 24px" }}>
      <SearchField label="Search looks" value={q} onChange={setQ} />
    </div>
  );
}

// ---- TabBar and TitleBlock ----------------------------------------------------------------------

function TabBarSection() {
  const [tab, setTab] = useState<TabId>("look");
  const [shortcutTab, setShortcutTab] = useState<TabId>("light");
  useTabShortcuts(setShortcutTab);
  return (
    <Section id="tabbar" title="TabBar and TitleBlock" lead="←/→ move between tabs inside the bar. ⌘1–⌘7 switch tabs from anywhere on this page (the second bar follows the shortcuts).">
      <Demo title="Live" frame="bare">
        <TabBar value={tab} onChange={setTab} />
        <TitleBlock tab={tab} />
      </Demo>
      <Demo title="Follows ⌘1–⌘7" frame="bare">
        <TabBar value={shortcutTab} onChange={setShortcutTab} label="Shortcut tabs" />
      </Demo>
      {TABS.map((t) => (
        <Demo key={t.id} title={`Active: ${t.label}`} frame="bare">
          <TabBar value={t.id} onChange={() => undefined} label={`${t.label} active`} />
        </Demo>
      ))}
      <Demo title="TitleBlock · every tab, and the Export panel's ink block" frame="bare">
        {TABS.map((t) => (
          <TitleBlock key={t.id} tab={t.id} />
        ))}
        <TitleBlock tone="ink" title="Export" blurb="3 photos from Lisbon. Nothing is saved besides the exported files." />
      </Demo>
    </Section>
  );
}

// ---- LookTile ----------------------------------------------------------------------------------

function LookSection() {
  const [look, setLook] = useState("neutral");
  return (
    <Section id="look" title="LookTile" lead="Thumbnail of the current photo through the look, a 1–9 badge, the DEFAULT tag, the name below. Selected is a 3 px ring and the name on a tab-colour label.">
      <Demo title="Live grid · built-in and imported" frame="bare" tab="look">
        <div style={{ padding: "12px 24px" }}>
          <div className="ym-look-grid">
            {DEMO_LOOKS.map((l, i) => (
              <LookTile key={l.id} name={l.name} src={lookThumbnail("DSC04204", l)} index={i + 1} isDefault={l.id === "neutral"} selected={l.id === look} onSelect={() => setLook(l.id)} />
            ))}
          </div>
        </div>
      </Demo>
      <Demo title="States · selected, default, no badge (10th), disabled (file missing)" frame="bare" tab="look">
        <div style={{ padding: "12px 24px" }}>
          <div className="ym-look-grid">
            <LookTile name="Selected" src={lookThumbnail("DSC04204", DEMO_LOOKS[5]!)} index={1} selected onSelect={() => undefined} />
            <LookTile name="Default" src={lookThumbnail("DSC04204", DEMO_LOOKS[0]!)} index={2} isDefault onSelect={() => undefined} />
            <LookTile name="Tenth look" src={lookThumbnail("DSC04204", DEMO_LOOKS[3]!)} index={10} onSelect={() => undefined} />
            <LookTile name="Missing" src={lookThumbnail("DSC04204", DEMO_LOOKS[2]!)} index={4} disabled note="File missing or changed" onSelect={() => undefined} />
            <LookTile name="A look with a very long name" src={lookThumbnail("DSC04204", DEMO_LOOKS[7]!)} index={5} onSelect={() => undefined} />
            <LookTile name="No picture yet" index={6} onSelect={() => undefined} />
          </div>
        </div>
      </Demo>
      <Demo title="Selected in each tab colour" frame="bare">
        {TABS.map((t) => (
          <div key={t.id} data-tab={t.id} style={{ display: "inline-block", width: 118, margin: "10px 0 10px 24px" }}>
            <LookTile name={t.label} src={lookThumbnail("DSC04204", DEMO_LOOKS[4]!)} selected onSelect={() => undefined} />
          </div>
        ))}
      </Demo>
    </Section>
  );
}

// ---- Wheel -------------------------------------------------------------------------------------

function WheelSection() {
  const [points, setPoints] = useState<Record<"shadows" | "midtones" | "highlights", ColorBalancePoint>>({ shadows: [-0.3, -0.4], midtones: [0, 0], highlights: [0.45, 0.35] });
  const [log, setLog] = useState("");
  return (
    <Section id="wheel" title="Wheel" lead="Drag the puck; double-click resets; arrows nudge (Shift ×10); Home, Delete or Backspace centre it. The puck takes the hue of its direction.">
      <Demo title="Live · three wheels" frame="bare">
        <div style={{ padding: "12px 24px" }}>
          <WheelGroup>
            {(["shadows", "midtones", "highlights"] as const).map((key) => (
              <Wheel
                key={key}
                label={key[0]!.toUpperCase() + key.slice(1)}
                value={points[key]}
                onChange={(v, final) => {
                  setPoints((s) => ({ ...s, [key]: v }));
                  setLog(`${key} ${v[0].toFixed(2)}, ${v[1].toFixed(2)}${final ? " (final)" : ""}`);
                }}
              />
            ))}
          </WheelGroup>
          <div className="dev-output">{log || "drag a puck"}</div>
        </div>
      </Demo>
      <Demo title="Hues around the wheel" frame="free">
        {[
          [0.9, 0],
          [0.65, -0.65],
          [0, -0.9],
          [-0.65, -0.65],
          [-0.9, 0],
          [-0.65, 0.65],
          [0, 0.9],
          [0.65, 0.65],
        ].map(([a, b], i) => (
          <Wheel key={i} label={`${i * 45}°`} value={[a as number, b as number]} onChange={() => undefined} size={72} />
        ))}
      </Demo>
      <Demo title="Centred, disabled" frame="free">
        <Wheel label="Centred" value={[0, 0]} onChange={() => undefined} />
        <Wheel label="Disabled" value={[0.3, 0.2]} onChange={() => undefined} disabled />
      </Demo>
    </Section>
  );
}

// ---- WarpPlane ---------------------------------------------------------------------------------

/** A photo's colours for the demo: a cloud of blues around the first pin's source. */
const WARP_DEMO_SCATTER = Array.from({ length: 400 }, (_, i) => [-0.09 + 0.02 * Math.sin(i * 1.9), -0.06 + 0.02 * Math.cos(i * 2.7)]).flat();

function WarpSection() {
  const [pins, setPins] = useState<ColorWarpPin[]>([movingPin([-0.09, -0.06], [-0.13, 0.03]), movingPin([0.05, 0.05], [0.12, 0.07]), lockedPin([0.12, -0.11])]);
  const [selected, setSelected] = useState<number | null>(0);
  const [tool, setTool] = useState<WarpTool>("move");
  const [log, setLog] = useState("");
  return (
    <Section
      id="warp"
      title="WarpPlane"
      lead="Colour › Colour warp's chroma plane. Press on empty plane to add a pin (Move or Lock) and drag it; press a pin to select and move it; double-click puts a target back on its source. Tab reaches each pin: arrows move it (Shift ×10), Home resets its target, Delete removes it. Sixteen pins at most."
    >
      <Demo title="Live · two moves and a lock" tab="color">
        <div style={{ display: "flex", flexDirection: "column", gap: 10, width: 352 }}>
          <Segmented
            label="Pin tool"
            value={tool}
            onChange={setTool}
            options={[
              { value: "move", label: "Move" },
              { value: "lock", label: "Lock" },
            ]}
          />
          <WarpPlane
            pins={pins}
            selected={selected}
            onSelect={setSelected}
            tool={tool}
            scatter={WARP_DEMO_SCATTER}
            onChange={(next, final) => {
              setPins(next);
              const pin = selected === null ? next.at(-1) : (next[selected] ?? next.at(-1));
              setLog(`${next.length} pins${pin ? ` · ${pinValueText(pin)}` : ""}${final ? " (final)" : ""}`);
            }}
          />
          <div className="dev-output">{log || "drag a colour"}</div>
        </div>
      </Demo>
    </Section>
  );
}

// ---- CurvePlot ---------------------------------------------------------------------------------

function CurveSection() {
  const [channel, setChannel] = useState<CurveChannel>("luma");
  const [curves, setCurves] = useState<Record<CurveChannel, CurvePoint[]>>({
    luma: CURVE_DEMO_POINTS.luma,
    red: CURVE_DEMO_POINTS.red,
    green: CURVE_DEMO_POINTS.green,
    blue: CURVE_DEMO_POINTS.blue,
  });
  const [full, setFull] = useState<CurvePoint[]>(
    Array.from({ length: MAX_CURVE_POINTS }, (_, i): CurvePoint => [i / (MAX_CURVE_POINTS - 1), Math.min(1, Math.max(0, i / (MAX_CURVE_POINTS - 1) + 0.08 * Math.sin(i * 1.7)))]),
  );
  const [identity, setIdentity] = useState<CurvePoint[]>([[0, 0], [1, 1]]);
  const [histogramCurve, setHistogramCurve] = useState<CurvePoint[]>([[0, 0], [0.3, 0.2], [0.7, 0.85], [1, 1]]);
  return (
    <Section id="curve" title="CurvePlot and Histogram" lead="Press anywhere on the plot to add a point on the curve and drag it. Drag a point off the plot to delete it (the ends stay). With the keyboard: Tab to a point, arrows move it, Delete removes it, Return on the plot adds one.">
      <Demo title="Live · all four channels" frame="bare">
        <div style={{ padding: "12px 24px", display: "flex", flexDirection: "column", gap: 10 }}>
          <Segmented
            label="Curve channel"
            value={channel}
            onChange={setChannel}
            options={(Object.keys(CURVE_CHANNEL_LABELS) as CurveChannel[]).map((c) => ({ value: c, label: CURVE_CHANNEL_LABELS[c] }))}
          />
          <CurvePlot key={channel} channel={channel} points={curves[channel]} defaultSelected={1} onChange={(points) => setCurves((s) => ({ ...s, [channel]: points }))} />
        </div>
      </Demo>
      <Demo title="Identity (two end points)" frame="bare">
        <div style={{ padding: "12px 24px" }}>
          <CurvePlot points={identity} onChange={setIdentity} />
        </div>
      </Demo>
      <Demo title="With a histogram" frame="bare">
        <div style={{ padding: "12px 24px" }}>
          <CurvePlot channel="blue" points={histogramCurve} histogram={demoLumaHistogram()} onChange={setHistogramCurve} />
        </div>
      </Demo>
      <Demo title={`${MAX_CURVE_POINTS} points: the most a curve may have`} frame="bare">
        <div style={{ padding: "12px 24px" }}>
          <CurvePlot channel="green" points={full} onChange={setFull} />
        </div>
      </Demo>
      <Demo title="Histogram (Light tab)" frame="bare">
        <Histogram data={demoHistogram()} left="R 182  G 164  B 140" right="ISO 400 · 1/125 · f/2.8" />
      </Demo>
    </Section>
  );
}

// ---- Filmstrip ---------------------------------------------------------------------------------

function FilmstripSection() {
  const [current, setCurrent] = useState<number | string>(4);
  const [picked, setPicked] = useState<number[]>([2, 3, 4]);
  const base = lisbonPhotos({ picked });
  const photos = base.map((p, i) => (i === 6 ? { ...p, status: "failed" as const } : i === 9 ? { ...p, status: "developing" as const } : p));
  return (
    <Section id="filmstrip" title="Filmstrip" lead="Click selects, ⌘-click or P picks, ←/→ move the selection. The seventh file can't be opened and the tenth is still developing.">
      <Demo title="Browse · live" frame="wide">
        <div style={{ border: "2px solid var(--ink)", height: 124, display: "grid" }}>
          <Filmstrip
            photos={photos}
            currentId={current}
            onSelect={setCurrent}
            onTogglePick={(id) => setPicked((p) => (p.includes(Number(id) - 1) ? p.filter((x) => x !== Number(id) - 1) : [...p, Number(id) - 1]))}
          />
        </div>
      </Demo>
      <Demo title="Export · 3 of 12 included, the rest dimmed" frame="wide">
        <div style={{ border: "2px solid var(--ink)", height: 124, display: "grid" }}>
          <Filmstrip mode="export" photos={lisbonPhotos({ included: [2, 3, 4] })} currentId={4} />
        </div>
      </Demo>
    </Section>
  );
}

// ---- PhotoCell ---------------------------------------------------------------------------------

function PhotoSection() {
  const [rect, setRect] = useState("");
  const [probe, setProbe] = useState<string | null>(null);
  const camera = "Sony A1 · DSC04204 · ISO 400 · 1/125 · f/2.8";
  return (
    <Section id="photo" title="PhotoCell" lead="Always 3:2. It reports its rectangle (resize the window to see it change). Hold ⌥ over the photo for the pixel readout. In the app the native viewport leaves the cell transparent and draws behind it; the fallback viewport sends a picture of the whole cell.">
      <Demo title="Look chip and camera line · reports its rect" frame="free">
        <div className="dev-photo-frame">
          <PhotoCell imageSrc={photoUrl("DSC04204")} imageLabel="Photo DSC04204, Neutral · 100%" lookChip="Neutral · 100%" cameraLine={camera} native={false} onRect={(r) => setRect(`x ${r.x.toFixed(1)}  y ${r.y.toFixed(1)}  ${r.width.toFixed(1)} × ${r.height.toFixed(1)}  @${r.scale}×`)} />
        </div>
        <div className="dev-output" style={{ width: "100%" }}>
          onRect: {rect}
        </div>
      </Demo>
      <Demo title="Before mode" frame="free">
        <div className="dev-photo-frame">
          <PhotoCell imageSrc={photoUrl("DSC04206")} imageLabel="Photo DSC04206, before" lookChip="Neutral · 100%" before cameraLine={camera} native={false} />
        </div>
      </Demo>
      <Demo title="⌥ pixel readout (hover with Option held)" frame="free">
        <div className="dev-photo-frame">
          <PhotoCell
            imageSrc={photoUrl("DSC04207")}
            imageLabel="Photo DSC04207"
            lookChip="Signature · 80%"
            cameraLine="Sony A1 · DSC04207 · ISO 100"
            native={false}
            pixelReadout={probe}
            onPixelProbe={(p) => setProbe(p ? `R ${Math.round(p.x * 255)} G ${Math.round(p.y * 255)} B 140` : null)}
          />
        </div>
      </Demo>
      <Demo title="Fallback viewport: the shell's picture of the whole cell, drawn pixel for pixel on paper" frame="free">
        <div className="dev-photo-frame">
          <PhotoCell
            frame={{ src: photoUrl("DSC04204"), widthPx: 1000, heightPx: 667 }}
            imageLabel="Photo DSC04204, drawn by the fallback viewport"
            lookChip="Neutral · 100%"
            cameraLine={camera}
            native={false}
          />
        </div>
      </Demo>
      <Demo title="Native mode: transparent cell over a checkerboard (the photo is drawn behind)" frame="free">
        <div className="dev-photo-frame" style={{ background: "repeating-conic-gradient(#d9d9d9 0 25%, #fff 0 50%) 0 0 / 24px 24px" }}>
          <PhotoCell imageLabel="Photo DSC04204, drawn natively" lookChip="Neutral · 100%" cameraLine={camera} native />
        </div>
      </Demo>
    </Section>
  );
}

// ---- UpdatePill --------------------------------------------------------------------------------

function UpdateSection() {
  const states: Array<[string, UpdateState, boolean]> = [
    ["Idle: nothing is drawn", { phase: "idle" }, false],
    ["Update available (click downloads)", { phase: "available", version: "0.2.0" }, false],
    ["Downloading, no length yet", { phase: "downloading", version: "0.2.0" }, false],
    ["Downloading 42%", { phase: "downloading", version: "0.2.0", percent: 42 }, false],
    ["Update ready (click restarts)", { phase: "ready", version: "0.2.0" }, false],
    ["Ready while exporting: disabled, hover for the reason", { phase: "ready", version: "0.2.0" }, true],
    ["Update failed (reason on hover)", { phase: "error", message: "Could not reach the update server. Try again later." }, false],
  ];
  const [clicks, setClicks] = useState(0);
  return (
    <Section id="update" title="UpdatePill" lead="States come from updater.ts: idle, available, downloading, ready and error.">
      {states.map(([title, state, exporting]) => (
        <Demo key={title} title={title} frame="bare">
          <div style={{ display: "flex", height: 75, border: "3px solid var(--ink)" }}>
            <UpdatePill state={state} exportRunning={exporting} onActivate={() => setClicks((c) => c + 1)} />
            <span style={{ width: 150 }} />
          </div>
        </Demo>
      ))}
      <Demo title="Activations" frame="free">
        <span className="dev-note">onActivate called {clicks} times</span>
      </Demo>
    </Section>
  );
}

// ---- Queues ------------------------------------------------------------------------------------

function QueueSection() {
  const queue: ExportQueueItem[] = [
    { id: 1, file: "DSC04203_neutral.jpg", state: "done" },
    { id: 2, file: "DSC04204_neutral.jpg", state: "running", percent: 45 },
    { id: 3, file: "DSC04205_neutral.jpg", state: "waiting" },
    { id: 4, file: "DSC04206_neutral.jpg", state: "failed", note: "The folder is full. Free some space and try again." },
    { id: 5, file: "DSC04207_neutral.jpg", state: "skipped" },
  ];
  return (
    <Section id="queue" title="ExportQueueList and DevelopList" lead="The Export panel's queue and the Start screen's list of photos being developed.">
      <Demo title="ExportQueueList · done, running, waiting, failed, skipped" frame="bare">
        <div style={{ padding: "12px 24px" }}>
          <ExportQueueList items={queue} />
        </div>
      </Demo>
      <Demo title="DevelopList · ready, developing, waiting, skipped" frame="free">
        <div style={{ width: 560, padding: "12px 0" }}>
          <DevelopList
            items={[
              { id: 1, name: "DSC04201", state: "ready" },
              { id: 2, name: "DSC04202", state: "ready" },
              { id: 3, name: "DSC04203", state: "skipped", note: "ARW 6.0 compressed: not supported yet." },
              { id: 4, name: "DSC04204", state: "developing" },
              { id: 5, name: "DSC04205", state: "waiting" },
            ]}
          />
        </div>
      </Demo>
    </Section>
  );
}

// ---- DropZone ----------------------------------------------------------------------------------

function DropSection() {
  const [log, setLog] = useState("Drop files on the zone below, or anywhere on this page.");
  const [windowLog, setWindowLog] = useState("");
  const { dragging } = useWindowDrop({
    onDrop: (paths) => {
      const { accepted, ignored } = filterArwPaths(paths);
      setWindowLog(`window drop: ${accepted.length} to open, ${ignored.length} ignored${accepted.length === 0 ? " · “No ARW files in what you dropped.”" : ""}`);
    },
  });
  return (
    <Section id="drop" title="DropZone" lead="Dashed 3 px box. It highlights while a drag is over the window (useWindowDrop) and reports drops onto itself through onDropPaths. Non-ARW files are ignored.">
      <Demo title="Large · idle" frame="bare">
        <div style={{ display: "flex", height: 380, padding: 12 }}>
          <DropZone title="Drop files or a folder" hint="ARW only, as many as you like">
            <Button>Open files…</Button>
            <Button variant="secondary">Open folder…</Button>
          </DropZone>
        </div>
      </Demo>
      <Demo title="Large · a drag is over the window" frame="bare">
        <div style={{ display: "flex", height: 380, padding: 12 }}>
          <DropZone title="Drop files or a folder" hint="ARW only, as many as you like" active>
            <Button>Open files…</Button>
            <Button variant="secondary">Open folder…</Button>
          </DropZone>
        </div>
      </Demo>
      <Demo title="Large · the last drop had no ARW files" frame="bare">
        <div style={{ display: "flex", height: 380, padding: 12 }}>
          <DropZone title="Drop files or a folder" hint="ARW only, as many as you like" notice="No ARW files in what you dropped.">
            <Button>Open files…</Button>
            <Button variant="secondary">Open folder…</Button>
          </DropZone>
        </div>
      </Demo>
      <Demo title="Compact (wizard) · live: whole window and the zone itself" frame="bare">
        <div style={{ padding: 12 }}>
          <DropZone variant="compact" title="Or drop the zip here" hint="A zip or an unpacked folder both work." active={dragging} onDropPaths={(paths) => setLog(`zone drop: ${paths.join(", ")}`)}>
            <LinkButton>Choose file…</LinkButton>
          </DropZone>
          <div className="dev-output">
            {log}
            {"\n"}
            {windowLog}
          </div>
        </div>
      </Demo>
    </Section>
  );
}

// ---- StepRail ----------------------------------------------------------------------------------

const STEPS = [
  { title: "Choose", text: "Pick the ARRI packages you want." },
  { title: "Get the files", text: "Download them from ARRI's site." },
  { title: "Recognise", text: "We check what each package is." },
  { title: "Import", text: "Copy the looks into the app." },
  { title: "Finish", text: "Pick a default and clean up." },
];

function StepsSection() {
  const [step, setStep] = useState(2);
  return (
    <Section id="steps" title="StepRail" lead="The wizard's rail. Finished steps show a tick and can be revisited; later steps are locked.">
      <Demo title="Live · step 3 of 5 (click a finished step)" frame="free">
        <div style={{ width: 360, border: "3px solid var(--ink)" }}>
          <StepRail steps={STEPS} current={step} onSelect={setStep} />
        </div>
        <Button variant="secondary" onClick={() => setStep((s) => (s + 1) % 5)}>
          Next step
        </Button>
      </Demo>
      {[0, 4].map((s) => (
        <Demo key={s} title={`Step ${s + 1}`} frame="free">
          <div style={{ width: 360, border: "3px solid var(--ink)" }}>
            <StepRail steps={STEPS} current={s} label={`Steps, step ${s + 1}`} />
          </div>
        </Demo>
      ))}
    </Section>
  );
}
