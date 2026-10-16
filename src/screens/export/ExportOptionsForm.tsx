import { useRef, useState } from "react";
import { Input } from "@/components/glass/input";
import type { ApplyScope, BitDepth, ColorSpace, ExportFormat, ExportOptions, ExportEstimate, IfExists, TiffCompression } from "../../api/types";
import { Card, Field, formats, LinkButton, Segmented, Slider } from "../../components";
import { longEdgeRange, snapLongEdge } from "../../export/format";
import type { ExportModel } from "../../export/model";
import type { AppStore } from "../../state/store";

/** The Adobe notice that goes with DNG (`docs/09` §Legal: "in the About window and the DNG export help text"). */
export const ADOBE_NOTICE = "This product includes DNG technology under license by Adobe.";

/**
 * Said under the DNG note when a look in the job has its own display rendering (`docs/09` §What a DNG carries): a
 * DNG still works, and is close to the picture here rather than the same. It is `ym_export::dng::OWN_DISPLAY_NOTE`,
 * which `ym render --help` also says; a test in `src-tauri/src/bindings.rs` fails if this copy drifts.
 */
export const OWN_DISPLAY_NOTE = "With a look that includes its own display rendering, the DNG is close to, not the same as, the picture you see.";

const DNG_NOTE =
  "Linear DNG with your look baked in. Opens in Lightroom close to this render, with room to adjust exposure, highlights and white balance.";

/**
 * Said under the DNG note: what other raw editors show (`docs/09` §Compatibility). The same sentence as `ym render --help`:
 * it is `ym_export::dng::OTHER_APPS_NOTE`, and a test in `src-tauri/src/bindings.rs` fails if this copy drifts.
 */
export const OTHER_APPS_NOTE = "Apps other than Adobe's show the stored linear image darker until their own exposure handling is applied.";

const DNG_LATER_EDITS = "White balance changes made in another app are approximate, because they act after the look.";

const COLOR_NOTES: Record<ColorSpace, string> = {
  srgb: "Best for sharing and the web.",
  display_p3: "Wider colours. Use it for recent Macs, iPhones and iPads.",
  rec2020: "Very wide colours, for video and HDR work.",
};

/** The long edge the slider starts at when Size changes to Long edge. */
const FIRST_LONG_EDGE = 3000;

/** Longest side offered in the defaults form, where no photo says how big "full" is. */
const DEFAULTS_MAX_EDGE = 10000;

export interface ExportOptionsFormProps {
  store: AppStore;
  options: ExportOptions;
  /**
   * `panel` is the Export panel: Photos first, then the options, the folder, and the command line. `defaults` is
   * Preferences › Export defaults: the same options without photos and without the command line.
   */
  mode: "panel" | "defaults";
  model: ExportModel;
  scope?: ApplyScope;
  /** The core's estimate for the job: DNG availability, the number of adjusted photos. */
  estimate: ExportEstimate | null;
  /** The command line for the job. */
  cli: string | null;
}

/** Copies text to the clipboard; false when the browser or webview refuses. Falls back to a hidden text box where the async API is missing. */
async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    // Fall through to the older way.
  }
  try {
    const box = document.createElement("textarea");
    box.value = text;
    box.setAttribute("readonly", "");
    box.style.position = "fixed";
    box.style.opacity = "0";
    document.body.appendChild(box);
    box.select();
    const ok = document.execCommand("copy");
    box.remove();
    return ok;
  } catch {
    return false;
  }
}

/**
 * The cards of the Export panel (`design/mockups/export.html`, `docs/09` §Options): Photos, Format with its own rows, Colour
 * space, Size, Save to with the file name and what to do when it exists, and the command line. Every change goes to the
 * store, which remembers it as the default.
 */
export function ExportOptionsForm({ store, options, mode, model, scope, estimate, cli }: ExportOptionsFormProps) {
  const panel = mode === "panel";
  const [copied, setCopied] = useState(false);
  const lastEdge = useRef(options.size.kind === "long_edge" ? options.size.px : FIRST_LONG_EDGE);
  if (options.size.kind === "long_edge") lastEdge.current = options.size.px;

  const set = (patch: Partial<ExportOptions>, final = true): void => store.setExportOptions(patch, final);
  const fullEdge = model.fullEdge;
  const range = longEdgeRange(panel ? (fullEdge ?? 0) : DEFAULTS_MAX_EDGE);
  const dngApproximate = panel && estimate?.dng_approximate === true;
  const bits = (value: string): BitDepth => (value === "8" ? 8 : 16);

  const chooseFolder = async (): Promise<void> => {
    try {
      const folder = await store.api.chooseFolder(options.destination ?? model.destination);
      if (folder) set({ destination: folder });
    } catch (error) {
      store.reportError(error);
    }
  };

  const copy = async (): Promise<void> => {
    if (!cli) return;
    if (await copyText(cli)) {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const adjusted = estimate?.adjusted_photos ?? 0;

  return (
    <>
      {panel && scope ? (
        <Card density="option" title="Photos">
          <Segmented
            label="Photos"
            size="lg"
            value={scope}
            onChange={(next) => store.setExportScope(next)}
            options={[
              { value: "this", label: "This photo" },
              { value: "picked", label: `Picked · ${model.nPicked}` },
              { value: "all", label: `All · ${model.nAll}` },
            ]}
          />
        </Card>
      ) : null}

      <Card density="option" title="Format">
        <Segmented
          label="Format"
          size="lg"
          value={options.format}
          onChange={(next: ExportFormat) => set({ format: next })}
          options={[
            { value: "jpeg", label: "JPEG" },
            { value: "tiff", label: "TIFF" },
            { value: "png", label: "PNG" },
            { value: "dng", label: "DNG" },
          ]}
        />
        {options.format === "jpeg" ? (
          <Slider
            className="ym-export__slider"
            label="Quality"
            value={options.jpeg_quality}
            min={50}
            max={100}
            step={1}
            resetColumn={false}
            onChange={(v, final) => set({ jpeg_quality: v }, final)}
          />
        ) : null}
        {options.format === "tiff" ? (
          <>
            <Field label="Bit depth" className="ym-export__sub">
              <Segmented
                label="Bit depth"
                size="sm"
                value={String(options.bit_depth)}
                onChange={(v) => set({ bit_depth: bits(v) })}
                options={[
                  { value: "8", label: "8-bit" },
                  { value: "16", label: "16-bit" },
                ]}
              />
            </Field>
            <Field label="Compression" className="ym-export__sub">
              <Segmented
                label="Compression"
                size="sm"
                value={options.tiff_compression}
                onChange={(v: TiffCompression) => set({ tiff_compression: v })}
                options={[
                  { value: "none", label: "None" },
                  { value: "lzw", label: "LZW" },
                  { value: "zip", label: "ZIP" },
                ]}
              />
            </Field>
          </>
        ) : null}
        {options.format === "png" ? (
          <Field label="Bit depth" className="ym-export__sub">
            <Segmented
              label="Bit depth"
              size="sm"
              value={String(options.bit_depth)}
              onChange={(v) => set({ bit_depth: bits(v) })}
              options={[
                { value: "8", label: "8-bit" },
                { value: "16", label: "16-bit" },
              ]}
            />
          </Field>
        ) : null}
        {options.format === "dng" ? (
          <>
            <p className="ym-row-note">{DNG_NOTE}</p>
            <p className="ym-row-note">{OTHER_APPS_NOTE}</p>
            {dngApproximate ? <p className="ym-row-note">{OWN_DISPLAY_NOTE}</p> : null}
            <p className="ym-row-note ym-export__legal">
              {DNG_LATER_EDITS} {ADOBE_NOTICE}
            </p>
          </>
        ) : null}
      </Card>

      {options.format !== "dng" ? (
        <>
          <Card density="option" title="Colour space">
            <Segmented
              label="Colour space"
              size="lg"
              value={options.color_space}
              onChange={(v: ColorSpace) => set({ color_space: v })}
              options={[
                { value: "srgb", label: "sRGB" },
                { value: "display_p3", label: "Display P3" },
                { value: "rec2020", label: "Rec.2020" },
              ]}
            />
            <p className="ym-row-note">{COLOR_NOTES[options.color_space]}</p>
          </Card>
          <Card density="option" title="Size">
            <Segmented
              label="Size"
              size="lg"
              value={options.size.kind}
              onChange={(kind) => {
                if (kind === "full") set({ size: { kind: "full" } });
                else set({ size: { kind: "long_edge", px: snapLongEdge(lastEdge.current, panel ? (fullEdge ?? 0) : DEFAULTS_MAX_EDGE) } });
              }}
              options={[
                { value: "full", label: panel && fullEdge ? `Full · ${fullEdge} px` : "Full" },
                { value: "long_edge", label: "Long edge", disabled: range === null },
              ]}
            />
            {options.size.kind === "long_edge" && range ? (
              <Slider
                className="ym-export__slider"
                label="Long edge"
                value={Math.min(range.max, Math.max(range.min, options.size.px))}
                min={range.min}
                max={range.max}
                step={20}
                format={formats.px0}
                resetColumn={false}
                onChange={(v, final) => set({ size: { kind: "long_edge", px: v } }, final)}
              />
            ) : null}
          </Card>
        </>
      ) : null}

      <Card
        density="option"
        title="Save to"
        action={
          options.destination !== null ? (
            <LinkButton size="sm" onClick={() => set({ destination: null })}>
              Use default
            </LinkButton>
          ) : undefined
        }
      >
        <div className="ym-export__save">
          <span className="ym-export__folder" title={panel ? model.destination : undefined}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" aria-hidden="true" style={{ flexShrink: 0 }}>
              <path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
            </svg>
            <span className="ym-export__folder-text">{panel || options.destination !== null ? model.destination : "Next to each source folder"}</span>
          </span>
          <LinkButton size="sm" onClick={() => void chooseFolder()}>
            Choose…
          </LinkButton>
        </div>
        <Field label="File name" className="ym-export__sub ym-export__sub--roomy">
          <Input
            type="text"
            size="sm"
            className="ym-export__template"
            aria-label="File name template"
            aria-invalid={model.problem !== null || undefined}
            aria-describedby="ym-export-template-note"
            spellCheck={false}
            autoComplete="off"
            value={options.name_template}
            onChange={(event) => set({ name_template: event.target.value }, false)}
            onBlur={() => set({ name_template: options.name_template }, true)}
          />
        </Field>
        <div className="ym-export__example" id="ym-export-template-note">
          <span />
          {model.problem ? (
            <span className="ym-export__problem" role="alert">
              {model.problem}
            </span>
          ) : (
            <span>
              {panel && model.example ? model.example : panel ? "" : "Tokens: {name} {look} {date} {seq} {camera}"}
              {model.warning ? <span className="ym-export__warning"> {model.warning}</span> : null}
            </span>
          )}
        </div>
        <Field label="If it exists" className="ym-export__sub ym-export__sub--roomy">
          <Segmented
            label="If the file exists"
            size="sm"
            value={options.if_exists}
            onChange={(v: IfExists) => set({ if_exists: v })}
            options={[
              { value: "skip", label: "Skip" },
              { value: "replace", label: "Replace" },
              { value: "keep_both", label: "Keep both" },
            ]}
          />
        </Field>
      </Card>

      {panel ? (
        <Card
          density="option"
          title="Same thing from the command line"
          last
          className="ym-export__cli"
          action={
            <LinkButton size="sm" disabled={!cli} onClick={() => void copy()}>
              {copied ? "Copied" : "Copy"}
            </LinkButton>
          }
        >
          <code className="ym-export__code" aria-label="Command line">
            {cli ?? "…"}
          </code>
          {adjusted > 0 ? (
            <p className="ym-row-note" role="note">
              {adjusted === 1 ? "1 photo has" : `${adjusted} photos have`} adjustments this command doesn&apos;t include.
            </p>
          ) : null}
        </Card>
      ) : null}
    </>
  );
}
