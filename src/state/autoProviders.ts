/**
 * The words of the Auto providers (`docs/22-auto-providers.md` sections 3.4 and 12): their names, the hint of each, the note under a
 * program that is listed but cannot be used, the Experimental note, the consent dialog's text, the instructions' label and the lines
 * of the Automatic card and the Start screen. Pure functions of the core's data, so Preferences, the card, the Start screen and the
 * tests use one copy of every sentence. The core has the same table for `ym` (`ym_core::auto_provider::unavailable_hint`); a sentence
 * changes in both places.
 */
import type { AppleAuto, AppleAutoReason, AutoProvider, ProviderState, ProviderStatus } from "../api/types";

/** The providers in the dropdown's order (`AutoProvider::ALL`). */
export const AUTO_PROVIDERS: readonly AutoProvider[] = ["yellowmoon", "apple", "claude", "agy"];

/** The programs that send a photo away, and so need the person's consent first (`AutoProvider::is_cloud`). */
export type CloudProvider = "claude" | "agy";

/** Whether a provider is one of the two programs. */
export function isCloud(provider: AutoProvider): provider is CloudProvider {
  return provider === "claude" || provider === "agy";
}

/** What the interface calls a provider (`AutoProvider::name`; `docs/22` section 1: "Antigravity (agy)" in the dropdown). */
export const PROVIDER_NAMES: Readonly<Record<AutoProvider, string>> = {
  yellowmoon: "Default Auto",
  apple: "Apple Intelligence",
  claude: "Claude Code",
  agy: "Antigravity",
};

/** The name a provider has in the dropdown and in the sentence that says what is run: Antigravity is "Antigravity (agy)" there. */
export const PROVIDER_LIST_NAMES: Readonly<Record<AutoProvider, string>> = { ...PROVIDER_NAMES, agy: "Antigravity (agy)" };

/** The title of the row (`docs/22` section 12.1), also the `Select`'s accessible name. */
export const AUTO_GRADING_TITLE = "Auto grading";

/** The hint under the row, by the provider that grades now (`docs/22` section 12.2, word for word). */
export const PROVIDER_HINTS: Readonly<Record<AutoProvider, string>> = {
  yellowmoon:
    "Default Auto sets the exposure and the white by a fixed rule and leaves every other value at 0. It is the same on every computer and sends nothing anywhere.",
  apple:
    "Apple's on-device model looks at each photo and grades it: exposure, highlights, shadows, black level, contrast, warmth, tint, saturation and the colours of the shadows and highlights. It runs on this Mac and sends nothing anywhere.",
  claude:
    "Claude Code looks at a small copy of each photo and grades it the same way. Yellowmoon runs the Claude Code installed on this computer and hands it the copy, 768 pixels on the long side, and your instructions. Claude Code sends them to Anthropic, or to the service it is set up to use, with your own sign-in. Yellowmoon itself makes no connection. Answers can differ from run to run.",
  agy: "Antigravity looks at a small copy of each photo and grades it the same way. Yellowmoon runs the Antigravity (agy) installed on this computer and hands it the copy, 768 pixels on the long side, and your instructions. Antigravity sends them to Google with your own sign-in. Google may keep and review them unless you opt out in Antigravity's own settings, and Google's Antigravity terms apply. Yellowmoon itself makes no connection. Answers can differ from run to run.",
};

/** The note under the hint while Antigravity is the provider (`docs/22` section 12.2). The badge beside it says Experimental too. */
export const EXPERIMENTAL_NOTE =
  "Experimental. Antigravity has no way to take a picture on its input, so its agent opens the copy as a file. That is slower and less predictable than Claude Code, and the agent may open other tools. Yellowmoon runs it in a sandbox, in an empty folder, and never skips its permission checks.";

/** The badge's text. */
export const EXPERIMENTAL_LABEL = "Experimental";

/** The muted tag after the name of an entry that is listed but cannot be chosen (`docs/22` section 12.1), by the provider's state. */
export function unavailableTag(state: ProviderState): string | null {
  switch (state) {
    case "signed_out":
      return "Not signed in";
    case "too_old":
      return "Update needed";
    case "unavailable":
      return "Not working";
    default:
      return null;
  }
}

/**
 * The sentence under the row for a program that is listed but cannot be used (`docs/22` section 3.4, the hint column), with the version
 * the program printed; null for what the table does not show (not installed, still checking, and Apple, whose row hides instead).
 * On Windows "Terminal" is "a terminal window".
 */
export function unavailableHint(
  provider: AutoProvider,
  reason: AppleAutoReason | null,
  toolVersion: string | null,
  os: string,
): string | null {
  if (!isCloud(provider) || reason === null) return null;
  const name = PROVIDER_NAMES[provider];
  const program = provider === "claude" ? "claude" : "agy";
  const terminal = os === "windows" ? "a terminal window" : "Terminal";
  switch (reason) {
    case "signed_out":
      return `${name} is installed but not signed in. Open ${terminal}, run ${program} and sign in.`;
    case "too_old":
      return `${name}${toolVersion ? ` ${toolVersion}` : ""} is too old. Update it: open ${terminal} and run ${program} update.`;
    case "probe_failed":
      return `${name} didn't answer its check. Open ${terminal}, run ${program} to see if it starts, then open Preferences again.`;
    case "unsafe_launcher":
      return provider === "claude"
        ? "Yellowmoon found Claude Code only as a script it can't run safely. Install the native version of Claude Code."
        : null;
    default:
      return null;
  }
}

/** The note for a listed provider that cannot be chosen: its hint, or null. */
export function providerNote(status: ProviderStatus, os: string): string | null {
  if (!status.listed || status.selectable) return null;
  return unavailableHint(status.provider, status.reason, status.tool_version, os);
}

/** The saved-choice note (`docs/22` section 12.2): the saved program is not the one that grades now. */
export function savedChoiceNote(saved: AutoProvider, effective: AutoProvider): string | null {
  if (!isCloud(saved) || effective !== "yellowmoon") return null;
  return `Your saved choice, ${PROVIDER_NAMES[saved]}, isn't available now, so Default Auto is used until it is.`;
}

/** The label and the hint of the instructions field, by the provider that grades now (`docs/22` section 12.4). */
export const INSTRUCTIONS_COPY: Readonly<Record<Exclude<AutoProvider, "yellowmoon">, { label: string; hint: string }>> = {
  apple: {
    label: "Instructions for Apple Intelligence",
    hint: "Apple's model reads these before each photo. The answer form stays the same.",
  },
  claude: {
    label: "Instructions for Claude Code",
    hint: "Claude reads these before each photo. The answer form stays the same.",
  },
  agy: {
    label: "Instructions for Antigravity",
    hint: "Antigravity reads these before each photo. The answer form stays the same.",
  },
};

/** What the consent dialog says for a program (`docs/22` section 12.3, word for word). */
export interface ConsentCopy {
  title: string;
  body: string;
  /** The primary button. */
  agree: string;
}

/** The consent dialog's text by program. */
export const CONSENT_COPY: Readonly<Record<CloudProvider, ConsentCopy>> = {
  claude: {
    title: "Send photos to Claude Code?",
    body: "To grade a photo, Yellowmoon runs Claude Code on this computer and hands it a copy of the photo, 768 pixels on the long side, and your instructions. Claude Code sends them to Anthropic, or to the service it is set up to use, with your own account. The original file, its name and its place on disk stay here. Whatever Anthropic keeps depends on your account's privacy settings. Yellowmoon itself makes no connection.",
    agree: "Use Claude Code",
  },
  agy: {
    title: "Send photos to Antigravity?",
    body: "To grade a photo, Yellowmoon runs Antigravity (agy) on this computer and hands it a copy of the photo, 768 pixels on the long side, and your instructions. Antigravity sends them to Google with your own account. Google may keep and review them unless you opt out in Antigravity's own settings, and Google's Antigravity terms apply. The original file, its name and its place on disk stay here. Antigravity is an agent that can open other tools, so Yellowmoon runs it in a sandbox, in an empty folder, and never skips its permission checks. This is experimental. Yellowmoon itself makes no connection.",
    agree: "Use Antigravity",
  },
};

/** The sentence a Windows build adds to the end of the body (`docs/22` D11). */
export const WINDOWS_CONSENT_SENTENCE = "The temporary folder's path includes your Windows user name, and the program may send it along.";

/** The consent body for a program on an operating system (Rust's `std::env::consts::OS`). */
export function consentBody(provider: CloudProvider, os: string): string {
  const { body } = CONSENT_COPY[provider];
  return os === "windows" ? `${body} ${WINDOWS_CONSENT_SENTENCE}` : body;
}

/** The consent dialog's Cancel button. */
export const CONSENT_CANCEL = "Cancel";

/**
 * The provider a photo's Auto state belongs to: the answer's or failure's own (`info.provider`), else the provider that grades (the
 * saved choice, which the app always follows) while it is pending or unavailable.
 */
export function stateProvider(apple: AppleAuto, chosen: AutoProvider): AutoProvider {
  return apple.state === "answered" || apple.state === "failed" ? apple.info.provider : chosen;
}

/** The reasons a program the person chose is not available that the Automatic card says so for (`docs/22` section 12.5). */
const CARD_UNAVAILABLE_REASONS: readonly AppleAutoReason[] = ["not_installed", "too_old", "signed_out", "probe_failed", "unsafe_launcher"];

/**
 * The Automatic card's line for a photo's state, or null while the card is the one without a provider (`docs/22` section 12.5):
 * answered, pending and failed say so with the provider's name; a program that was chosen and is not available says that; nothing for
 * Apple, for `off`, for `checking` and for `not_consented`.
 */
export function autoCardLine(apple: AppleAuto, chosen: AutoProvider): string | null {
  const name = PROVIDER_NAMES[stateProvider(apple, chosen)];
  switch (apple.state) {
    case "answered":
      return `Set with ${name}.`;
    case "pending":
      return `${name} is looking at this photo…`;
    case "failed":
      return `${name} couldn't look at this photo, so Auto uses Yellowmoon's own settings.`;
    case "unavailable":
      return isCloud(chosen) && CARD_UNAVAILABLE_REASONS.includes(apple.reason)
        ? `${name} isn't available right now, so Auto uses Yellowmoon's own settings.`
        : null;
    default:
      return null;
  }
}

/** The Start screen's note on a developed photo that waits for its answer (`docs/22` section 12.5). */
export function gradedByNote(provider: AutoProvider): string {
  return `Graded by ${PROVIDER_NAMES[provider]}…`;
}

/** The Start screen's hint while every photo is developed and some still wait. */
export function gradingHint(provider: AutoProvider): string {
  return `${PROVIDER_NAMES[provider]} is looking at each photo. You can start looking while it finishes.`;
}

/** The Start screen's title while some photos wait. */
export function gradingTitle(count: number, provider: AutoProvider): string {
  return `Grading ${count} ${count === 1 ? "photo" : "photos"} with ${PROVIDER_NAMES[provider]}`;
}
