import { describe, expect, it } from "vitest";
import { MOCK_APPLE_INFO, MOCK_CLAUDE_INFO } from "../api.mock";
import { mockAutoStatus, mockProviderStatus } from "../api/providersMock";
import type { MockProviders } from "../api/providersMock";
import type { AppleAuto, AutoConsent, AutoProvider } from "../api/types";
import {
  autoCardLine,
  AUTO_PROVIDERS,
  autoRowShown,
  checkingNote,
  consentBody,
  CONSENT_COPY,
  dropdownEntries,
  dropdownValue,
  gradedByNote,
  gradingHint,
  gradingTitle,
  isCloud,
  PROVIDER_LIST_NAMES,
  PROVIDER_NAMES,
  providerNote,
  savedChoiceNote,
  savedProgramChecking,
  stateProvider,
  unavailableHint,
  unavailableTag,
  USE_DEFAULT_AUTO,
} from "./autoProviders";

describe("the words of the Auto providers (docs/22 sections 3.4 and 12)", () => {
  it("names the providers as AutoProvider::name does, and Antigravity (agy) in the dropdown", () => {
    expect(AUTO_PROVIDERS).toEqual(["yellowmoon", "apple", "claude", "agy"]);
    expect(AUTO_PROVIDERS.map((p) => PROVIDER_NAMES[p])).toEqual(["Default Auto", "Apple Intelligence", "Claude Code", "Antigravity"]);
    expect(AUTO_PROVIDERS.map((p) => PROVIDER_LIST_NAMES[p])).toEqual(["Default Auto", "Apple Intelligence", "Claude Code", "Antigravity (agy)"]);
    expect(AUTO_PROVIDERS.filter(isCloud)).toEqual(["claude", "agy"]);
  });

  it("gives the tag of an entry that cannot be chosen: a program that was found and cannot be used, or the saved one still being checked", () => {
    expect(unavailableTag("signed_out")).toBe("Not signed in");
    expect(unavailableTag("too_old")).toBe("Update needed");
    expect(unavailableTag("unavailable")).toBe("Not working");
    expect(unavailableTag("checking")).toBe("Checking…");
    for (const state of ["available", "not_installed"] as const) expect(unavailableTag(state)).toBeNull();
  });

  it("has the hint of every state of Claude Code, with the version and the operating system", () => {
    const hint = (reason: Parameters<typeof unavailableHint>[1], version: string | null = null, os = "macos") => unavailableHint("claude", reason, version, os);
    expect(hint("signed_out")).toBe("Claude Code is installed but not signed in. Open Terminal, run claude and sign in.");
    expect(hint("too_old", "2.1.100")).toBe("Claude Code 2.1.100 is too old. Update it: open Terminal and run claude update.");
    expect(hint("probe_failed")).toBe("Claude Code didn't answer its check. Open Terminal, run claude to see if it starts, then open Preferences again.");
    expect(hint("unsafe_launcher")).toBe("Yellowmoon found Claude Code only as a script it can't run safely. Install the native version of Claude Code.");
    expect(hint("not_installed")).toBeNull();
    expect(hint("checking")).toBeNull();
    expect(hint(null)).toBeNull();
    expect(hint("signed_out", null, "windows")).toBe("Claude Code is installed but not signed in. Open a terminal window, run claude and sign in.");
    expect(hint("too_old", "2.1.100", "windows")).toBe("Claude Code 2.1.100 is too old. Update it: open a terminal window and run claude update.");
    expect(hint("too_old", null)).toBe("Claude Code is too old. Update it: open Terminal and run claude update.");
    // The launcher sentence names no terminal.
    expect(hint("unsafe_launcher", null, "windows")).toBe(hint("unsafe_launcher"));
  });

  it("has the hint of every state of Antigravity", () => {
    const hint = (reason: Parameters<typeof unavailableHint>[1], version: string | null = null, os = "linux") => unavailableHint("agy", reason, version, os);
    expect(hint("signed_out")).toBe("Antigravity is installed but not signed in. Open Terminal, run agy and sign in.");
    expect(hint("too_old", "1.0.9")).toBe("Antigravity 1.0.9 is too old. Update it: open Terminal and run agy update.");
    expect(hint("probe_failed")).toBe("Antigravity didn't answer its check. Open Terminal, run agy to see if it starts, then open Preferences again.");
    expect(hint("signed_out", null, "windows")).toBe("Antigravity is installed but not signed in. Open a terminal window, run agy and sign in.");
    // The script launcher is a Claude Code case only.
    expect(hint("unsafe_launcher")).toBeNull();
    expect(hint("not_installed")).toBeNull();
  });

  it("has no hint for Apple Intelligence or Default Auto: their rows hide instead", () => {
    expect(unavailableHint("apple", "apple_intelligence_off", null, "macos")).toBeNull();
    expect(unavailableHint("yellowmoon", null, null, "macos")).toBeNull();
  });

  it("makes the note of a listed provider that cannot be chosen, and none for the others", () => {
    const consent: AutoConsent = { claude: true, agy: true };
    const note = (mock: Parameters<typeof mockProviderStatus>[1], provider: "claude" | "agy" | "apple" = "claude") =>
      providerNote(mockProviderStatus(provider, mock, consent, ""), "macos");
    expect(note("signed_out")).toBe("Claude Code is installed but not signed in. Open Terminal, run claude and sign in.");
    expect(note("too_old", "agy")).toBe("Antigravity 1.0.9 is too old. Update it: open Terminal and run agy update.");
    expect(note("available")).toBeNull();
    expect(note(undefined)).toBeNull();
    expect(note("checking")).toBeNull();
    expect(note("unavailable", "apple")).toBeNull();
  });

  it("says the saved choice is not available only for a program that is not the one that grades", () => {
    expect(savedChoiceNote("claude", "yellowmoon")).toBe("Your saved choice, Claude Code, isn't available now, so Default Auto is used until it is.");
    expect(savedChoiceNote("agy", "yellowmoon")).toBe("Your saved choice, Antigravity, isn't available now, so Default Auto is used until it is.");
    expect(savedChoiceNote("claude", "claude")).toBeNull();
    expect(savedChoiceNote("claude", "apple")).toBeNull();
    expect(savedChoiceNote("apple", "yellowmoon")).toBeNull();
    expect(savedChoiceNote("yellowmoon", "yellowmoon")).toBeNull();
  });

  it("shows the row while a provider is there or the saved choice is a program, and not for a saved Apple Intelligence that is gone", () => {
    const status = (providers: MockProviders, preference: AutoProvider, consent: AutoConsent = { claude: true, agy: true }) =>
      mockAutoStatus({ providers, preference, consent, own: "" });
    // A provider is there.
    expect(autoRowShown(status({ apple: "available" }, "apple"))).toBe(true);
    expect(autoRowShown(status({ claude: "signed_out" }, "yellowmoon"))).toBe(true);
    // Nothing is there: the default (a saved Apple Intelligence) and Default Auto hide the row, as the browser build does.
    expect(autoRowShown(status({}, "apple"))).toBe(false);
    expect(autoRowShown(status({ apple: "unavailable" }, "apple"))).toBe(false);
    expect(autoRowShown(status({ apple: "unavailable" }, "yellowmoon"))).toBe(false);
    // A saved program keeps it, so the choice can always be seen and withdrawn.
    expect(autoRowShown(status({ apple: "unavailable" }, "claude"))).toBe(true);
    expect(autoRowShown(status({ apple: "unavailable" }, "agy"))).toBe(true);
  });

  it("shows a saved program in the dropdown only while it is listed and cannot be chosen, so that Default Auto is a change", () => {
    const value = (providers: MockProviders, preference: AutoProvider, consent: AutoConsent = { claude: true, agy: true }) =>
      dropdownValue(mockAutoStatus({ providers, preference, consent, own: "" }));
    // Listed but not usable: the program itself (a disabled entry with its tag).
    expect(value({ claude: "signed_out" }, "claude")).toBe("claude");
    expect(value({ claude: "too_old" }, "claude")).toBe("claude");
    expect(value({ agy: "unavailable" }, "agy")).toBe("agy");
    // Usable: who grades. Not listed (gone): Default Auto grades, and the note's button withdraws it.
    expect(value({ claude: "available" }, "claude")).toBe("claude");
    expect(value({ apple: "available" }, "apple")).toBe("apple");
    expect(value({ apple: "unavailable" }, "claude")).toBe("yellowmoon");
    expect(value({ apple: "unavailable" }, "agy")).toBe("yellowmoon");
    // Still being checked for the first time (the app starts no program at launch): the saved program itself, as the disabled entry with
    // "Checking…", not a Default Auto it has not been told to use.
    expect(value({ claude: "checking" }, "claude")).toBe("claude");
    expect(value({ agy: "checking" }, "agy")).toBe("agy");
    expect(value({ apple: "available", claude: "checking" }, "apple")).toBe("apple");
    expect(value({ claude: "checking" }, "yellowmoon")).toBe("yellowmoon");
    // Available but never agreed to: Default Auto grades and is shown.
    expect(value({ claude: "available" }, "claude", { claude: false, agy: false })).toBe("yellowmoon");
    // A saved choice that is not a program shows who grades.
    expect(value({ claude: "signed_out" }, "yellowmoon")).toBe("yellowmoon");
    expect(value({ claude: "signed_out", apple: "available" }, "apple")).toBe("apple");
    expect(USE_DEFAULT_AUTO).toBe("Use Default Auto");
  });

  it("says a program is being checked, and shows the saved one, while its first check has not ended (the app starts no program at launch)", () => {
    const status = (providers: MockProviders, preference: AutoProvider, consent: AutoConsent = { claude: true, agy: true }) =>
      mockAutoStatus({ providers, preference, consent, own: "" });
    // The line under the row names the programs that are being checked, in the dropdown's order.
    expect(checkingNote(status({ claude: "checking" }, "apple"))).toBe("Checking Claude Code…");
    expect(checkingNote(status({ agy: "checking" }, "apple"))).toBe("Checking Antigravity…");
    expect(checkingNote(status({ claude: "checking", agy: "checking" }, "apple"))).toBe("Checking Claude Code and Antigravity…");
    expect(checkingNote(status({ apple: "available", claude: "available", agy: "checking" }, "apple"))).toBe("Checking Antigravity…");
    // Apple Intelligence is an operating-system call: it is never "being checked".
    expect(checkingNote(status({ apple: "checking" }, "apple"))).toBeNull();
    for (const providers of [{}, { claude: "available" }, { claude: "signed_out", agy: "too_old" }, { agy: "unavailable" }] as const) {
      expect(checkingNote(status(providers, "apple"))).toBeNull();
    }
    // Only a saved program counts: it is the one whose row must not say "isn't available now" yet.
    expect(savedProgramChecking(status({ claude: "checking" }, "claude"))).toBe(true);
    expect(savedProgramChecking(status({ agy: "checking" }, "agy"))).toBe(true);
    expect(savedProgramChecking(status({ agy: "checking" }, "claude"))).toBe(false);
    expect(savedProgramChecking(status({ claude: "checking" }, "apple"))).toBe(false);
    expect(savedProgramChecking(status({ claude: "signed_out" }, "claude"))).toBe(false);
    expect(savedProgramChecking(status({ claude: "available" }, "claude"))).toBe(false);
    // The dropdown lists the saved program while it is being checked, and no other program that is.
    const entries = (providers: MockProviders, preference: AutoProvider) => dropdownEntries(status(providers, preference)).map((p) => p.provider);
    expect(entries({ claude: "checking" }, "claude")).toEqual(["yellowmoon", "claude"]);
    expect(entries({ claude: "checking", agy: "checking" }, "agy")).toEqual(["yellowmoon", "agy"]);
    expect(entries({ apple: "available", claude: "checking" }, "apple")).toEqual(["yellowmoon", "apple"]);
    expect(entries({ apple: "available", claude: "available", agy: "signed_out" }, "agy")).toEqual(["yellowmoon", "apple", "claude", "agy"]);
    // The row is not shown just because a program is being checked: a computer with none would see it come and go.
    expect(autoRowShown(status({ claude: "checking" }, "apple"))).toBe(false);
    expect(autoRowShown(status({ claude: "checking" }, "claude"))).toBe(true);
  });

  it("has the consent dialog's text for each program, with the Windows sentence only on Windows", () => {
    expect(CONSENT_COPY.claude.title).toBe("Send photos to Claude Code?");
    expect(CONSENT_COPY.agy.title).toBe("Send photos to Antigravity?");
    expect([CONSENT_COPY.claude.agree, CONSENT_COPY.agy.agree]).toEqual(["Use Claude Code", "Use Antigravity"]);
    for (const os of ["macos", "linux"]) expect(consentBody("claude", os)).toBe(CONSENT_COPY.claude.body);
    expect(consentBody("agy", "windows")).toBe(`${CONSENT_COPY.agy.body} The temporary folder's path includes your Windows user name, and the program may send it along.`);
    // The dialog says what is sent where: the picture's size, the instructions, the service, and that Yellowmoon makes no connection.
    for (const copy of Object.values(CONSENT_COPY)) {
      expect(copy.body).toContain("768 pixels on the long side");
      expect(copy.body).toContain("your instructions");
      expect(copy.body).toContain("Yellowmoon itself makes no connection.");
      expect(copy.body).toContain("The original file, its name and its place on disk stay here.");
    }
  });
});

describe("the Automatic card's line (docs/22 section 12.5)", () => {
  const answered = (info: typeof MOCK_APPLE_INFO): AppleAuto => ({
    state: "answered",
    info,
    answer: { scene: "", plan: "", steps: { exposure: 0, highlights: null, shadows: null, blacks: null, contrast: null, white: null, tint: null, saturation: null, shadow_color: null, shadow_color_strength: null, highlight_color: null, highlight_color_strength: null } },
  });

  it("names the provider the answer belongs to, whatever is chosen now", () => {
    expect(autoCardLine(answered(MOCK_CLAUDE_INFO), "apple")).toBe("Set with Claude Code.");
    expect(autoCardLine(answered(MOCK_APPLE_INFO), "claude")).toBe("Set with Apple Intelligence.");
    expect(autoCardLine({ state: "failed", kind: "cli_failed", message: "x", info: MOCK_CLAUDE_INFO }, "claude")).toBe(
      "Claude Code couldn't look at this photo, so Auto uses Yellowmoon's own settings.",
    );
    expect(stateProvider(answered(MOCK_CLAUDE_INFO), "apple")).toBe("claude");
  });

  it("names the chosen provider while the photo waits", () => {
    expect(autoCardLine({ state: "pending" }, "agy")).toBe("Antigravity is looking at this photo…");
    expect(autoCardLine({ state: "pending" }, "apple")).toBe("Apple Intelligence is looking at this photo…");
    expect(stateProvider({ state: "pending" }, "agy")).toBe("agy");
  });

  it("says a program that was chosen is not available, for the reasons of a program, and nothing for the rest", () => {
    for (const reason of ["not_installed", "too_old", "signed_out", "probe_failed", "unsafe_launcher"] as const) {
      expect(autoCardLine({ state: "unavailable", reason }, "claude")).toBe("Claude Code isn't available right now, so Auto uses Yellowmoon's own settings.");
    }
    expect(autoCardLine({ state: "unavailable", reason: "signed_out" }, "agy")).toBe("Antigravity isn't available right now, so Auto uses Yellowmoon's own settings.");
    for (const reason of ["checking", "not_consented", "not_a_mac", "apple_intelligence_off"] as const) {
      expect(autoCardLine({ state: "unavailable", reason }, "claude")).toBeNull();
    }
    // Nothing for Apple Intelligence, as before, whatever the reason.
    expect(autoCardLine({ state: "unavailable", reason: "not_installed" }, "apple")).toBeNull();
    expect(autoCardLine({ state: "off" }, "claude")).toBeNull();
  });
});

describe("the Start screen's lines (docs/22 section 12.5)", () => {
  it("name the provider", () => {
    expect(gradedByNote("claude")).toBe("Graded by Claude Code…");
    expect(gradingHint("agy")).toBe("Antigravity is looking at each photo. You can start looking while it finishes.");
    expect(gradingTitle(12, "apple")).toBe("Grading 12 photos with Apple Intelligence");
    expect(gradingTitle(1, "claude")).toBe("Grading 1 photo with Claude Code");
  });
});
