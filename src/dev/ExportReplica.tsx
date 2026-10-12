/**
 * The Export screen of `design/mockups/export.html`, rebuilt from the components (see
 * `PhotosReplica`). `/components.html?replica=export&fmt=tiff&running=1.4`.
 */
import { useMemo, useState } from "react";
import {
  Button,
  Card,
  ExportQueueList,
  Field,
  Filmstrip,
  formats,
  KeyHint,
  LinkButton,
  MoonMark,
  PhotoCell,
  ProgressBar,
  Segmented,
  Slider,
  TitleBlock,
  UpdatePill,
} from "../components";
import type { ExportQueueItem } from "../components";
import { SCENES } from "../fixtures/scenes";
import { lisbonPhotos, sceneDataUrl } from "./demoData";

type Format = "jpeg" | "tiff" | "png" | "dng";
type Scope = "this" | "picked" | "all";

export interface ExportReplicaProps {
  initialFormat?: Format;
  /** Photos done so far; undefined shows the options, a number shows the queue (3 means finished). */
  initialProgress?: number;
  initialColorSpace?: "srgb" | "p3" | "rec2020";
  initialSize?: "full" | "edge";
}

const PICKED = [2, 3, 4];
const CURRENT = 3;
const EXT: Record<Format, string> = { jpeg: "jpg", tiff: "tif", png: "png", dng: "dng" };

/** The export panel and its queue, as the mockup draws them. */
export function ExportReplica({ initialFormat = "jpeg", initialProgress, initialColorSpace = "srgb", initialSize = "full" }: ExportReplicaProps) {
  const [scope, setScope] = useState<Scope>("picked");
  const [fmt, setFmt] = useState<Format>(initialFormat);
  const [quality, setQuality] = useState(92);
  const [bits, setBits] = useState<"8" | "16">("16");
  const [comp, setComp] = useState<"none" | "lzw" | "zip">("lzw");
  const [cs, setCs] = useState(initialColorSpace);
  const [size, setSize] = useState(initialSize);
  const [edge, setEdge] = useState(3000);
  const [exists, setExists] = useState<"skip" | "replace" | "both">("both");
  const [prog, setProg] = useState<number | undefined>(initialProgress);
  const [copied, setCopied] = useState(false);

  const included = (i: number): boolean => scope === "all" || (scope === "picked" ? PICKED.includes(i) : i === CURRENT);
  const idx = SCENES.map((_, i) => i).filter(included);
  const count = idx.length;
  const running = prog !== undefined;
  const finished = running && prog >= count;
  const noun = count === 1 ? "photo" : "photos";
  const ext = EXT[fmt];

  const scale = fmt === "dng" || size === "full" ? 1 : (edge / 8640) ** 2;
  const perPhoto =
    { jpeg: 6 + (quality - 50) * 0.45, tiff: (bits === "16" ? 298 : 149) * (comp === "none" ? 1 : 0.68), png: bits === "16" ? 210 : 105, dng: 300 }[fmt] * scale;
  const total = perPhoto * count;
  const estimate = `≈ ${total >= 1000 ? `${(total / 1000).toFixed(1)} GB` : `${Math.max(1, Math.round(total))} MB`}`;

  const files = idx.map((i) => `${SCENES[i]!.name}.ARW`);
  const flags = ["--look neutral", `--format ${fmt}`];
  if (fmt === "jpeg") flags.push(`--quality ${quality}`);
  if (fmt === "tiff" || fmt === "png") flags.push(`--bits ${bits}`);
  if (fmt !== "dng") flags.push(`--colour-space ${cs}`);
  if (fmt !== "dng" && size === "edge") flags.push(`--long-edge ${edge}`);
  flags.push('--out "~/Pictures/Lisbon export"');
  const cli = `yellowmoon render ${scope === "all" ? "~/Pictures/Lisbon" : files.join(" ")} \\\n  ${flags.join(" \\\n  ")}`;

  const doneCount = Math.min(count, Math.floor(prog ?? 0));
  const jobs: ExportQueueItem[] = idx.map((i, k) => {
    const file = `${SCENES[i]!.name}_neutral.${ext}`;
    if (k < doneCount) return { id: i, file, state: "done" };
    if (k === doneCount && (prog ?? 0) < count) return { id: i, file, state: "running", percent: Math.round(((prog ?? 0) - k) * 100) };
    return { id: i, file, state: "waiting" };
  });
  const pct = count ? Math.round((Math.min(prog ?? 0, count) / count) * 100) : 0;

  const photos = useMemo(() => lisbonPhotos({ picked: PICKED, included: idx }), [scope]);
  const scene = SCENES[CURRENT]!;

  const title = finished ? "Exported" : running ? "Exporting" : "Export";
  const blurb = finished
    ? `${count} ${noun} saved to Lisbon export.`
    : running
      ? `${doneCount} of ${count} done.`
      : `${count} ${noun} from Lisbon. Nothing is saved besides the exported files.`;

  const seg = <T extends string>(label: string, value: T, onChange: (v: T) => void, options: Array<[T, string]>, size: "sm" | "lg" = "lg") => (
    <Segmented label={label} value={value} onChange={onChange} size={size} options={options.map(([v, l]) => ({ value: v, label: l }))} />
  );

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
          <a className="rp-back" href="components.html?replica=photos">
            ← Back to photos
          </a>
          <span className="rp-status">Neural · Display P3</span>
        </nav>
        <div className="rp-header-right">
          <UpdatePill state={{ phase: "ready", version: "0.1.1" }} />
          <span className="rp-export" aria-current="page">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="square" aria-hidden="true">
              <path d="M12 3v12" />
              <path d="M6.5 9.5L12 15l5.5-5.5" />
              <path d="M4 20h16" />
            </svg>
            Export
            <KeyHint>⌘E</KeyHint>
          </span>
        </div>
      </header>

      <PhotoCell
        style={{ gridColumn: 1, gridRow: 2 }}
        imageSrc={sceneDataUrl(scene, 0.09)}
        imageLabel={`Photo ${scene.name}`}
        lookChip="Neutral · 100%"
        cameraLine={`${scene.name} · ISO 400 · 1/125 · f/2.8`}
        native={false}
      />
      <Filmstrip className="rp-filmstrip" mode="export" photos={photos} currentId={CURRENT + 1} />

      <aside aria-label="Export" className="rp-tools">
        <TitleBlock tone="ink" title={title} blurb={blurb} />
        <div className="rp-tools-scroll">
          {!running ? (
            <>
              <Card density="option" title="Photos">
                {seg("Photos", scope, setScope, [["this", "This photo"], ["picked", `Picked · ${PICKED.length}`], ["all", `All · ${SCENES.length}`]])}
              </Card>
              <Card density="option" title="Format">
                {seg("Format", fmt, setFmt, [["jpeg", "JPEG"], ["tiff", "TIFF"], ["png", "PNG"], ["dng", "DNG"]])}
                {fmt === "jpeg" ? (
                  <Slider className="rp-slider-tight" label="Quality" value={quality} min={50} max={100} step={1} resetColumn={false} onChange={setQuality} />
                ) : null}
                {fmt === "tiff" ? (
                  <>
                    <Field label="Bit depth" className="rp-field-sub">{seg("Bit depth", bits, setBits, [["8", "8-bit"], ["16", "16-bit"]], "sm")}</Field>
                    <Field label="Compression" className="rp-field-sub">{seg("Compression", comp, setComp, [["none", "None"], ["lzw", "LZW"], ["zip", "ZIP"]], "sm")}</Field>
                  </>
                ) : null}
                {fmt === "png" ? <Field label="Bit depth" className="rp-field-sub">{seg("Bit depth", bits, setBits, [["8", "8-bit"], ["16", "16-bit"]], "sm")}</Field> : null}
                {fmt === "dng" ? (
                  <p className="ym-row-note">
                    Linear DNG with your look baked in. Opens in Lightroom close to this render, with room to adjust exposure, highlights and white balance.
                  </p>
                ) : null}
              </Card>
              {fmt !== "dng" ? (
                <>
                  <Card density="option" title="Colour space">
                    {seg("Colour space", cs, setCs, [["srgb", "sRGB"], ["p3", "Display P3"], ["rec2020", "Rec.2020"]])}
                    {cs === "srgb" ? <p className="ym-row-note">Best for sharing and the web.</p> : null}
                  </Card>
                  <Card density="option" title="Size">
                    {seg("Size", size, setSize, [["full", "Full · 8640 px"], ["edge", "Long edge"]])}
                    {size === "edge" ? (
                      <Slider className="rp-slider-tight" label="Long edge" value={edge} min={1000} max={8640} step={20} format={formats.px0} resetColumn={false} onChange={setEdge} />
                    ) : null}
                  </Card>
                </>
              ) : null}
              <Card density="option" title="Save to">
                <div className="rp-save">
                  <span className="rp-folder">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" aria-hidden="true" style={{ flexShrink: 0 }}>
                      <path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
                    </svg>
                    ~/Pictures/Lisbon export
                  </span>
                  <LinkButton size="sm">Choose…</LinkButton>
                </div>
                <Field label="File name" className="rp-field-12">
                  <span className="rp-template">{"{name}_{look}"}</span>
                </Field>
                <div className="rp-example">
                  <span />
                  <span>
                    {SCENES[idx[0] ?? CURRENT]!.name}_neutral.{ext}
                  </span>
                </div>
                <Field label="If it exists" className="rp-field-12">{seg("If the file exists", exists, setExists, [["skip", "Skip"], ["replace", "Replace"], ["both", "Keep both"]], "sm")}</Field>
              </Card>
              <Card
                density="option"
                title="Same thing from the command line"
                last
                className="rp-cli"
                action={
                  <LinkButton size="sm" onClick={() => setCopied(true)}>
                    {copied ? "Copied" : "Copy"}
                  </LinkButton>
                }
              >
                <code className="rp-code">{cli}</code>
              </Card>
            </>
          ) : (
            <div className="rp-queue">
              <ProgressBar value={pct} label="Exporting" />
              <div style={{ marginTop: 20 }}>
                <ExportQueueList items={jobs} />
              </div>
              <p className="ym-row-note" style={{ marginTop: 16 }}>
                {finished
                  ? "Originals are never changed. Open the folder or go back to your photos."
                  : "Originals are never changed. You can keep working while this runs; an update waits until the export is finished."}
              </p>
            </div>
          )}
        </div>
        <div className="rp-footer">
          {!running ? (
            <Button block size="lg" aside={`${estimate} · ⌘↩`} onClick={() => setProg(0)}>
              {`Export ${count} ${noun}`}
            </Button>
          ) : finished ? (
            <div className="rp-footer-pair">
              <Button variant="secondary" size="lg">
                Show in Finder
              </Button>
              <Button size="lg" onClick={() => setProg(undefined)}>
                Done
              </Button>
            </div>
          ) : (
            <Button variant="secondary" size="lg" block onClick={() => setProg(undefined)}>
              Cancel
            </Button>
          )}
        </div>
      </aside>
    </div>
  );
}
