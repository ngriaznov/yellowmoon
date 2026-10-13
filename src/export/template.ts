/**
 * File name templates of the Export panel (`docs/09-export.md` §Options): `{name}`, `{look}`, `{date}`,
 * `{seq}` and `{camera}`. The panel previews the first file name with `previewFileName`, checks what the
 * person typed with `templateProblem`, and the mock API names its files with `expandNameTemplate` and
 * `sanitizeStem`. The core has the last word on the real names (it also adds " 2", " 3" when "If it exists" is
 * Keep both).
 */
import type { ExportFormat } from "../api/types";

/** The template the panel starts with. */
export const DEFAULT_TEMPLATE = "{name}_{look}";

/** The tokens a template may hold. */
export const TEMPLATE_TOKENS = ["name", "look", "date", "seq", "camera"] as const;
export type TemplateToken = (typeof TEMPLATE_TOKENS)[number];

/** File extension of each format. */
export const FILE_EXTENSION: Readonly<Record<ExportFormat, string>> = { jpeg: "jpg", tiff: "tif", png: "png", dng: "dng" };

/** What a template is filled with for one photo. */
export interface TemplateParts {
  /** File name without extension: "DSC04204". */
  name: string;
  /** The look's slug: "neutral", "film-warm". */
  look: string;
  /** Capture date: "2026-10-04". */
  date: string;
  /** Position in the job, from 1. */
  seq: number;
  /** The body as a slug: "sony-a1". */
  camera: string;
}

/** Expands `{name}`, `{look}`, `{date}`, `{seq}` and `{camera}`; unknown tokens stay as typed. `{seq}` has three digits. */
export function expandNameTemplate(template: string, parts: TemplateParts): string {
  return template.replace(/\{(name|look|date|seq|camera)\}/g, (_, key: TemplateToken) =>
    key === "seq" ? String(parts.seq).padStart(3, "0") : String(parts[key]),
  );
}

/** The longest stem the core writes, in UTF-8 bytes (`MAX_STEM_BYTES` in `ym-export`'s `naming.rs`). */
export const MAX_STEM_BYTES = 200;

/** What `char::is_whitespace` is in Rust (Unicode White_Space). JavaScript's `\s` differs: it counts U+FEFF and Rust does not. */
const WHITESPACE = "\\t-\\r \\u0085\\u00a0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
const EDGE_DOTS_AND_SPACES = new RegExp(`^[.${WHITESPACE}]+|[.${WHITESPACE}]+$`, "g");
const TRAILING_DOTS_AND_SPACES = new RegExp(`[.${WHITESPACE}]+$`);

/** UTF-8 length of one code point (a lone surrogate counts 3, as the replacement character the core would hold). */
const utf8Length = (codePoint: number): number => (codePoint < 0x80 ? 1 : codePoint < 0x800 ? 2 : codePoint < 0x10000 ? 3 : 4);

/**
 * Makes an expanded template safe as a file name stem, exactly as the core does (`sanitize_stem`, `docs/09` §Names and folders):
 * `/ \ : * ? " < > |` and control characters become `_`, leading and trailing dots and spaces are dropped, at most 200 bytes are
 * kept (never half a character), and a stem with nothing left is "export".
 */
export function sanitizeStem(text: string): string {
  // Control characters (Rust's `is_control`: categories Cc) go before the trim, so a tab is an underscore and not an edge space.
  const replaced = text.replace(/[/\\:*?"<>|\u0000-\u001f\u007f-\u009f]/g, "_");
  const trimmed = replaced.replace(EDGE_DOTS_AND_SPACES, "");
  let out = "";
  let bytes = 0;
  for (const ch of trimmed) {
    const size = utf8Length(ch.codePointAt(0) ?? 0);
    if (bytes + size > MAX_STEM_BYTES) break;
    out += ch;
    bytes += size;
  }
  out = out.replace(TRAILING_DOTS_AND_SPACES, "");
  return out === "" ? "export" : out;
}

/**
 * The file name a template gives one photo, with the extension of the format: the name the core writes (it sanitises the stem the
 * same way), though not the " 2" it adds when "If it exists" is Keep both.
 */
export function previewFileName(template: string, parts: TemplateParts, format: ExportFormat): string {
  return `${sanitizeStem(expandNameTemplate(template, parts))}.${FILE_EXTENSION[format]}`;
}

/** "Sony A1" gives "sony-a1": the `{camera}` token and the look slug are lower case words joined by hyphens. */
export function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

/** The longest file name the panel accepts, in characters (macOS allows 255 bytes; the extension and a " 2" suffix need room). */
export const MAX_TEMPLATE_LENGTH = 200;

/**
 * Why a template can't be used, in plain words, or null when it is fine. The panel shows the message under the
 * box and disables the Export button until it is gone.
 */
export function templateProblem(template: string): string | null {
  if (template.trim() === "") return "Type a file name, for example {name}_{look}.";
  if (template.length > MAX_TEMPLATE_LENGTH) return `Use at most ${MAX_TEMPLATE_LENGTH} characters.`;
  if (/[/\\:]/.test(template)) return "A file name can't contain / \\ or :.";
  if (/^\./.test(template.trim())) return "A file name can't start with a dot.";
  let open = false;
  for (const ch of template) {
    if (ch === "{") {
      if (open) return "Close every { with a } before the next {.";
      open = true;
    } else if (ch === "}") {
      if (!open) return "A } needs a { before it.";
      open = false;
    }
  }
  if (open) return "Close every { with a }.";
  for (const match of template.matchAll(/\{([^}]*)\}/g)) {
    const token = match[1] ?? "";
    if (!(TEMPLATE_TOKENS as readonly string[]).includes(token)) {
      return `{${token}} is not a token. Use {name}, {look}, {date}, {seq} or {camera}.`;
    }
  }
  return null;
}

/**
 * A remark that does not stop the export: with more than one photo, a template with neither `{name}` nor
 * `{seq}` gives every photo the same name.
 */
export function templateWarning(template: string, photoCount: number): string | null {
  if (photoCount > 1 && !template.includes("{name}") && !template.includes("{seq}")) {
    return "Every photo would get the same name. Add {name} or {seq}.";
  }
  return null;
}
