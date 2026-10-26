/**
 * The mock's Auto providers (`docs/22` sections 3, 10 and 12.8): which of Apple Intelligence, Claude Code and Antigravity a computer
 * has and in what state, the status the core would make of that (`auto_status`), who grades (`Activity::decide`) and where a photo
 * stands as a result. Pure functions of plain data, so `src/api.mock.ts` plays them and the tests read them without a browser.
 */
import { MOCK_DEFAULT_INSTRUCTIONS, MOCK_DEFAULT_INSTRUCTIONS_SHA256, MOCK_INSTRUCTIONS_MAX_CHARS } from "./appleAutoMock";
import type {
  AppleAuto,
  AppleAutoAnswer,
  AppleAutoInfo,
  AppleAutoReason,
  AutoConsent,
  AutoProvider,
  AutoStatus,
  ProviderStatus,
} from "./types";

/** The three providers a computer can have (Default Auto is always there). */
export type MockProviderName = "apple" | "claude" | "agy";

/** What a mock provider is doing: available, or one of the ways a program is listed but cannot be used, or still being checked. */
export type MockProviderState = "available" | "signed_out" | "too_old" | "unavailable" | "checking";

/** The providers a mock computer has; one that is not named is not installed. */
export type MockProviders = Partial<Record<MockProviderName, MockProviderState>>;

const PROVIDER_STATES: readonly MockProviderState[] = ["available", "signed_out", "too_old", "unavailable", "checking"];

/**
 * Reads `?mockProviders=` (`docs/22` section 12.8): a comma list of `name[:state]`, `name` one of `apple`, `claude`, `agy` and `state`
 * one of `available` (the default), `signed_out`, `too_old`, `unavailable` and `checking`; `none` (or an empty list) is a computer with
 * no provider. An entry that is neither is left out. Undefined for text that is not there.
 */
export function parseMockProviders(text: string | null): MockProviders | undefined {
  if (text === null) return undefined;
  const out: MockProviders = {};
  for (const entry of text.split(",").map((e) => e.trim()).filter(Boolean)) {
    const [name, state = "available"] = entry.split(":");
    if (name !== "apple" && name !== "claude" && name !== "agy") continue;
    if (!(PROVIDER_STATES as readonly string[]).includes(state)) continue;
    out[name] = state as MockProviderState;
  }
  return out;
}

/** Reads `?mockConsent=claude,agy`: the programs the person has already agreed to. */
export function parseMockConsent(text: string | null): AutoConsent | undefined {
  if (text === null) return undefined;
  const names = text.split(",").map((e) => e.trim());
  return { claude: names.includes("claude"), agy: names.includes("agy") };
}

/** The versions the mock's programs print, and the too-old ones (the doubles of `docs/22` section 15.2 print `2.1.100` and `1.0.9`). */
export const MOCK_TOOL_VERSIONS = {
  claude: { current: "2.1.310", old: "2.1.100" },
  agy: { current: "1.4.7", old: "1.0.9" },
} as const;

/**
 * Which instructions the mock's model reads: the default, or the person's own with a stand-in digest (12 hex digits of an FNV-1a
 * hash: the browser has no synchronous SHA-256, and the mock only needs two texts to read apart).
 */
export function mockInstructionsId(own: string): Pick<AppleAutoInfo, "instructions" | "instructions_sha256"> {
  if (own.trim() === "" || own === MOCK_DEFAULT_INSTRUCTIONS) {
    return { instructions: "default", instructions_sha256: MOCK_DEFAULT_INSTRUCTIONS_SHA256 };
  }
  let h = 0xcbf29ce484222325n;
  for (const byte of new TextEncoder().encode(own)) h = BigInt.asUintN(64, (h ^ BigInt(byte)) * 0x100000001b3n);
  return { instructions: "edited", instructions_sha256: h.toString(16).padStart(16, "0").slice(0, 12) };
}

/** The model the mock's Apple Auto says it is (the example of `docs/19` section 7), with the default instructions. */
export const MOCK_APPLE_INFO: AppleAutoInfo = {
  provider: "apple",
  model: "AFM 3 Core",
  os: "27.0.1 (27A404)",
  tool_version: null,
  api: null,
  prompt: 3,
  ...mockInstructionsId(""),
};

/** Claude Code as the mock says it answers (`docs/22` section 12.8): `claude-sonnet-5-5`, version 2.1.310, Anthropic's own service. */
export const MOCK_CLAUDE_INFO: AppleAutoInfo = {
  provider: "claude",
  model: "claude-sonnet-5-5",
  os: "",
  tool_version: MOCK_TOOL_VERSIONS.claude.current,
  api: "firstParty",
  prompt: 1,
  ...mockInstructionsId(""),
};

/** Antigravity as the mock says it answers: its default model, version 1.4.7. */
export const MOCK_AGY_INFO: AppleAutoInfo = {
  provider: "agy",
  model: "default",
  os: "",
  tool_version: MOCK_TOOL_VERSIONS.agy.current,
  api: null,
  prompt: 1,
  ...mockInstructionsId(""),
};

const INFOS: Readonly<Record<MockProviderName, AppleAutoInfo>> = { apple: MOCK_APPLE_INFO, claude: MOCK_CLAUDE_INFO, agy: MOCK_AGY_INFO };

/** The info of an answer by a provider, with the instructions it read. */
export function mockInfo(provider: MockProviderName, own = ""): AppleAutoInfo {
  return { ...INFOS[provider], ...mockInstructionsId(own) };
}

/** What one mock provider reads as, in the core's terms (`ProviderStatus`, `docs/22` section 3.1). */
export function mockProviderStatus(
  provider: AutoProvider,
  mock: MockProviderState | undefined,
  consent: AutoConsent,
  own: string,
): ProviderStatus {
  const base = {
    provider,
    info: null,
    tool_version: null,
    consent: provider === "claude" ? consent.claude : provider === "agy" ? consent.agy : true,
    experimental: provider === "agy",
  };
  if (provider === "yellowmoon") {
    return { ...base, state: "available", reason: null, listed: true, selectable: true };
  }
  const name = provider;
  if (mock === undefined) {
    return { ...base, state: "not_installed", reason: "not_installed", listed: false, selectable: false };
  }
  if (provider === "apple") {
    // Apple's own answers: available, or one of its nine reasons, and then it is not listed (the row hides, `docs/19` section 2).
    if (mock === "available") return { ...base, state: "available", reason: null, info: mockInfo("apple", own), listed: true, selectable: true };
    if (mock === "checking") return { ...base, state: "checking", reason: "checking", listed: false, selectable: false };
    return { ...base, state: "unavailable", reason: "apple_intelligence_off", listed: false, selectable: false };
  }
  const versions = MOCK_TOOL_VERSIONS[name as "claude" | "agy"];
  switch (mock) {
    case "available":
      return { ...base, state: "available", reason: null, info: mockInfo(name as "claude" | "agy", own), tool_version: versions.current, listed: true, selectable: true };
    case "signed_out":
      return { ...base, state: "signed_out", reason: "signed_out", tool_version: versions.current, listed: true, selectable: false };
    case "too_old":
      return { ...base, state: "too_old", reason: "too_old", tool_version: versions.old, listed: true, selectable: false };
    case "unavailable":
      return { ...base, state: "unavailable", reason: "probe_failed", listed: true, selectable: false };
    default:
      return { ...base, state: "checking", reason: "checking", listed: false, selectable: false };
  }
}

/** Who acts for the session, and why not (`Activity::decide`, `docs/22` section 10.4). */
export type MockActivity =
  | { kind: "off" }
  | { kind: "unavailable"; reason: AppleAutoReason }
  | { kind: "active"; provider: Exclude<AutoProvider, "yellowmoon"> };

/**
 * `Activity::decide` for the mock: no provider on the computer is the browser build (no advisors: off); the chosen provider is Default
 * Auto (off); a provider that is not there, not available or, for a program, not agreed to, is unavailable with its reason; the rest
 * acts.
 */
export function mockActivity(providers: MockProviders, chosen: AutoProvider, consent: AutoConsent): MockActivity {
  if (Object.keys(providers).length === 0 || chosen === "yellowmoon") return { kind: "off" };
  const status = mockProviderStatus(chosen, providers[chosen], consent, "");
  if (providers[chosen] === undefined) return { kind: "unavailable", reason: "not_installed" };
  if (status.state !== "available") return { kind: "unavailable", reason: status.reason ?? "unavailable" };
  if ((chosen === "claude" || chosen === "agy") && !status.consent) return { kind: "unavailable", reason: "not_consented" };
  return { kind: "active", provider: chosen };
}

/** The `auto_status()` the core would answer for a mock computer (`docs/22` sections 3.1 and 12.6). */
export function mockAutoStatus(args: {
  providers: MockProviders;
  preference: AutoProvider;
  consent: AutoConsent;
  own: string;
}): AutoStatus {
  const { providers, preference, consent, own } = args;
  const statuses = (["yellowmoon", "apple", "claude", "agy"] as const).map((p) =>
    mockProviderStatus(p, p === "yellowmoon" ? undefined : providers[p], consent, own),
  );
  const activity = mockActivity(providers, preference, consent);
  const effective: AutoProvider = activity.kind === "active" ? activity.provider : "yellowmoon";
  const apple = statuses.find((s) => s.provider === "apple");
  const visible = apple?.state === "available" || statuses.some((s) => (s.provider === "claude" || s.provider === "agy") && s.listed);
  return {
    providers: statuses,
    preference,
    chosen: preference,
    effective,
    active: effective !== "yellowmoon",
    visible,
    consent,
    table: 2,
    ...mockInstructionsId(own),
    own_instructions: own,
    default_instructions: MOCK_DEFAULT_INSTRUCTIONS,
    instructions_max_chars: MOCK_INSTRUCTIONS_MAX_CHARS,
  };
}

/** The states of a photo that the mock plays while a provider acts (`?mockAppleAuto=`, `docs/22` section 12.8). */
export type MockPhotoMode = "none" | "unavailable" | "pending" | "answered" | "failed";

/**
 * Where a photo stands (the core's `Activity::photo_state`): off, unavailable with the reason, or, while a provider acts, pending, with
 * the answer or failed (no answer in time). `mode` names which of the three the mock plays; with none given the provider answers.
 */
export function mockPhotoState(activity: MockActivity, mode: MockPhotoMode, answer: AppleAutoAnswer, own: string): AppleAuto {
  if (activity.kind === "off") return { state: "off" };
  if (activity.kind === "unavailable") return { state: "unavailable", reason: activity.reason };
  const info = mockInfo(activity.provider, own);
  if (mode === "pending") return { state: "pending" };
  if (mode === "failed") return { state: "failed", kind: "timeout", message: "", info };
  return { state: "answered", answer, info };
}

/** The refusal of a patch that chooses a program without the person's yes (`PreferencesPatch::check`, `docs/22` section 10.2). */
export function consentRefusal(provider: "claude" | "agy"): string {
  return `Say yes to sending photos to ${provider === "claude" ? "Claude Code" : "Antigravity"} first.`;
}
