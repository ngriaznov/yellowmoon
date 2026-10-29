import { expect, test } from "@playwright/test";
import type { Locator, Page } from "@playwright/test";
import { openApp, openPhotos, watchErrors } from "./helpers/app";

// The Auto providers in the app (docs/22 section 12) in a real browser against the mocked API. The mock plays a computer with Apple
// Intelligence, Claude Code and Antigravity in some state (`mockProviders`), the saved choice (`mockAutoProvider`) and the programs the
// person agreed to (`mockConsent`): the Preferences row "Auto grading" with its dropdown, hints and notes, the consent dialog, the
// instructions' label, and the Automatic card's line.

const MOCKUP = { mockCurrent: 3, mockPicked: "2,3,4" } as const;
const TITLE = "Auto grading";
const CLAUDE_BODY =
  "To grade a photo, Yellowmoon runs Claude Code on this computer and hands it a copy of the photo, 768 pixels on the long side, and your instructions. Claude Code sends them to Anthropic, or to the service it is set up to use, with your own account. The original file, its name and its place on disk stay here. Whatever Anthropic keeps depends on your account's privacy settings. Yellowmoon itself makes no connection.";
const WINDOWS_SENTENCE = "The temporary folder's path includes your Windows user name, and the program may send it along.";

/** Preferences over Start, on a mock computer. */
async function preferences(page: Page, options: Record<string, string | number>) {
  await openApp(page, options);
  await page.keyboard.press("Meta+,");
  const dialog = page.getByRole("dialog", { name: "Preferences" });
  await expect(dialog.getByRole("group", { name: "Display rendering" })).toBeVisible();
  return dialog;
}

/** Whether everything in the dropdown's trigger shows: the text is not cut off and nothing reaches the chevron. */
const triggerFits = (trigger: Locator) =>
  trigger.evaluate((el) => {
    const value = el.querySelector<HTMLElement>('[data-slot="select-value"]')!;
    const inside = el.getBoundingClientRect();
    const shown = [...value.children].every((child) => child.getBoundingClientRect().right <= inside.right - 8);
    return value.scrollWidth - value.clientWidth <= 1 && shown;
  });

const saved = (page: Page) => page.evaluate(async () => (await window.__yellowmoon!.mockApi.preferencesGet()).auto_provider);

test.describe("the row", () => {
  test("is hidden where no provider is there: the browser build, and a computer that has none", async ({ page }) => {
    for (const options of [{}, { mockProviders: "none" }, { mockProviders: "claude:checking" }] as Array<Record<string, string>>) {
      const dialog = await preferences(page, options);
      await expect.poll(() => page.evaluate(() => window.__yellowmoon!.mockApi.autoStatus().then((s) => s.visible))).toBe(false);
      await expect(dialog.getByRole("group", { name: TITLE })).toHaveCount(0);
      await expect(dialog.getByRole("combobox", { name: TITLE })).toHaveCount(0);
      await expect(dialog.getByRole("textbox")).toHaveCount(0);
    }
  });

  test("is shown with a program that is only installed, even signed out", async ({ page }) => {
    const dialog = await preferences(page, { mockProviders: "claude:signed_out" });
    await expect(dialog.getByRole("group", { name: TITLE })).toBeVisible();
    const select = dialog.getByRole("combobox", { name: TITLE });
    await expect(select).toHaveText("Default Auto");
    await expect(dialog.getByRole("note")).toHaveText("Claude Code is installed but not signed in. Open Terminal, run claude and sign in.");
  });

  test("lists the providers in order, a program that cannot be used disabled with its tag, and Antigravity Experimental", async ({ page }) => {
    const problems = watchErrors(page);
    // Both programs are agreed to, so an entry that could be chosen would be saved at once: the check below can fail.
    const dialog = await preferences(page, {
      mockProviders: "apple,claude:too_old,agy:signed_out",
      mockAutoProvider: "yellowmoon",
      mockConsent: "claude,agy",
    });
    // While the list is open Radix hides the rest of the page from assistive technology, the trigger included.
    await expect(dialog.getByRole("combobox", { name: TITLE })).toHaveText("Default Auto");
    const trigger = page.getByRole("combobox", { name: TITLE, includeHidden: true });
    await trigger.click();
    const options = page.getByRole("option");
    await expect(options).toHaveCount(4);
    await expect(options.nth(0)).toContainText("Default Auto");
    await expect(options.nth(1)).toContainText("Apple Intelligence");
    await expect(options.nth(2)).toContainText("Claude Code");
    await expect(options.nth(2)).toContainText("Update needed");
    await expect(options.nth(3)).toContainText("Antigravity (agy)");
    await expect(options.nth(3)).toContainText("Experimental");
    await expect(options.nth(3)).toContainText("Not signed in");
    await expect(options.nth(2)).toHaveAttribute("aria-disabled", "true");
    await expect(options.nth(3)).toHaveAttribute("aria-disabled", "true");
    await expect(options.nth(1)).not.toHaveAttribute("aria-disabled", "true");
    // The badges are glasscn's, and a disabled entry is not chosen by a click.
    await expect(options.nth(2).locator('[data-slot="badge"]')).toHaveCount(1);
    await expect(options.nth(3).locator('[data-slot="badge"]')).toHaveCount(2);
    for (const index of [2, 3]) {
      await options.nth(index).click({ force: true });
      // Nothing was chosen: the choice is not saved, no question opened, and the dropdown still shows what it showed.
      expect(await saved(page)).toBe("yellowmoon");
      await expect(page.getByRole("dialog", { name: /^Send photos/ })).toHaveCount(0);
      await expect(trigger).toHaveText("Default Auto");
      await expect(options).toHaveCount(4);
    }
    await page.keyboard.press("Escape");
    await expect(trigger).toHaveText("Default Auto");
    expect(problems).toEqual([]);
  });

  test("says in a muted note under the row what to do for each program that cannot be used, with a terminal window on Windows", async ({ page }) => {
    let dialog = await preferences(page, { mockProviders: "apple,claude:too_old,agy:unavailable" });
    await expect(dialog.getByRole("note")).toHaveText([
      "Claude Code 2.1.100 is too old. Update it: open Terminal and run claude update.",
      "Antigravity didn't answer its check. Open Terminal, run agy to see if it starts, then open Preferences again.",
    ]);
    // Under the row, muted.
    const [rowBox, noteBox] = await Promise.all([dialog.getByRole("group", { name: TITLE }).boundingBox(), dialog.getByRole("note").first().boundingBox()]);
    expect(noteBox!.y).toBeGreaterThanOrEqual(rowBox!.y + rowBox!.height - 1);
    const [noteColour, titleColour] = await Promise.all([
      dialog.getByRole("note").first().evaluate((el) => getComputedStyle(el).color),
      dialog.getByRole("group", { name: TITLE }).locator(".ym-pref__title").evaluate((el) => getComputedStyle(el).color),
    ]);
    expect(noteColour).not.toBe(titleColour);
    dialog = await preferences(page, { mockProviders: "claude:signed_out", mockOs: "windows" });
    await expect(dialog.getByRole("note")).toHaveText("Claude Code is installed but not signed in. Open a terminal window, run claude and sign in.");
  });

  test("shows the hint of the provider that grades, whole, and Antigravity's Experimental note with its badge", async ({ page }) => {
    const dialog = await preferences(page, { mockProviders: "apple,claude,agy", mockAutoProvider: "agy", mockConsent: "claude,agy" });
    const row = dialog.getByRole("group", { name: TITLE });
    await expect(row).toContainText("Antigravity looks at a small copy of each photo and grades it the same way.");
    await expect(row).toContainText("Google may keep and review them unless you opt out in Antigravity's own settings, and Google's Antigravity terms apply.");
    const hint = row.locator(".ym-pref__hint");
    expect(await hint.evaluate((el) => el.scrollHeight <= el.clientHeight + 1)).toBe(true);
    const note = row.getByText(/^Experimental\. Antigravity has no way to take a picture on its input/);
    await expect(note).toBeVisible();
    await expect(row.locator(".ym-pref__experimental [data-slot=badge]")).toHaveText("Experimental");
    // The note is under the hint.
    const [hintBox, noteBox] = await Promise.all([hint.boundingBox(), note.boundingBox()]);
    expect(noteBox!.y).toBeGreaterThanOrEqual(hintBox!.y + hintBox!.height - 1);
    // The trigger shows the choice with its badge, all of it: nothing is cut off.
    const trigger = dialog.getByRole("combobox", { name: TITLE });
    await expect(trigger).toContainText("Antigravity");
    await expect(trigger).toContainText("Experimental");
    const clipped = await trigger.evaluate((el) => {
      const value = el.querySelector<HTMLElement>('[data-slot="select-value"]')!;
      const shown = [...value.children].every((child) => {
        const box = child.getBoundingClientRect();
        const inside = el.getBoundingClientRect();
        return box.right <= inside.right - 8;
      });
      return { scroll: value.scrollWidth - value.clientWidth, shown };
    });
    expect(clipped.scroll).toBeLessThanOrEqual(1);
    expect(clipped.shown).toBe(true);
    // The instructions are labelled for it.
    await expect(dialog.getByRole("textbox", { name: "Instructions for Antigravity" })).toBeVisible();
  });

  test("says the saved program is not available now, and follows it when it signs in", async ({ page }) => {
    const dialog = await preferences(page, { mockProviders: "apple,claude:signed_out", mockAutoProvider: "claude", mockConsent: "claude" });
    const note = dialog.getByText("Your saved choice, Claude Code, isn't available now, so Default Auto is used until it is.");
    await expect(note).toBeVisible();
    // The dropdown shows the saved program as the disabled entry it is, with its tag; Default Auto grades.
    await expect(dialog.getByRole("combobox", { name: TITLE })).toHaveText("Claude Code Not signed in");
    await expect(dialog.getByRole("textbox")).toHaveCount(0);
    await page.evaluate(() => window.__yellowmoon!.mockApi.setProviders({ apple: "available", claude: "available" }));
    await expect(note).toHaveCount(0);
    await expect(dialog.getByRole("combobox", { name: TITLE })).toHaveText("Claude Code");
    await expect(dialog.getByRole("textbox", { name: "Instructions for Claude Code" })).toBeVisible();
  });

  test("lets Default Auto be chosen while the saved program is signed out (the dropdown shows the program, so choosing Default Auto is a change)", async ({ page }) => {
    const problems = watchErrors(page);
    // Claude Code is saved, agreed to, installed and signed out; nothing else can be chosen: an Intel Mac, or one before macOS 27.
    const dialog = await preferences(page, { mockProviders: "apple:unavailable,claude:signed_out", mockAutoProvider: "claude", mockConsent: "claude" });
    const trigger = dialog.getByRole("combobox", { name: TITLE });
    await expect(trigger).toHaveText("Claude Code Not signed in");
    // The tag fits the 220 px of the trigger whole.
    expect(await triggerFits(trigger)).toBe(true);
    await trigger.click();
    const options = page.getByRole("option");
    await expect(options).toHaveCount(2);
    await expect(options.nth(1)).toHaveAttribute("aria-disabled", "true");
    await options.nth(0).click();
    await expect.poll(() => saved(page)).toBe("yellowmoon");
    await expect(page.getByRole("dialog", { name: /^Send photos/ })).toHaveCount(0);
    await expect(trigger).toHaveText("Default Auto");
    await expect(dialog.getByText(/^Your saved choice/)).toHaveCount(0);
    await expect(dialog.getByRole("button", { name: "Use Default Auto" })).toHaveCount(0);
    expect(problems).toEqual([]);
  });

  test("fits Antigravity's name and tag in the trigger when it is the saved choice and signed out", async ({ page }) => {
    const dialog = await preferences(page, { mockProviders: "agy:signed_out", mockAutoProvider: "agy", mockConsent: "agy" });
    const trigger = dialog.getByRole("combobox", { name: TITLE });
    await expect(trigger).toHaveText("Antigravity Not signed in");
    expect(await triggerFits(trigger)).toBe(true);
  });

  test("shows the saved program with Checking… while its first check has not ended, and settles when it ends", async ({ page }) => {
    // Nothing runs when the app starts: Preferences opens before the first check of a program has answered (a second or two).
    const problems = watchErrors(page);
    const dialog = await preferences(page, { mockProviders: "claude:checking", mockAutoProvider: "claude", mockConsent: "claude" });
    const trigger = dialog.getByRole("combobox", { name: TITLE });
    // The row is there with the saved program, not a Default Auto nobody chose; nothing says it is not available.
    await expect(trigger).toHaveText("Claude Code Checking…");
    expect(await triggerFits(trigger)).toBe(true);
    await expect(dialog.getByRole("note")).toHaveText("Checking Claude Code…");
    await expect(dialog.getByText(/isn't available now/)).toHaveCount(0);
    await expect(dialog.getByRole("button", { name: "Use Default Auto" })).toHaveCount(0);
    // The check ends: signed in, and the row is the ordinary one.
    await page.evaluate(() => window.__yellowmoon!.mockApi.setProviders({ claude: "available" }));
    await expect(trigger).toHaveText("Claude Code");
    await expect(dialog.getByRole("note")).toHaveCount(0);
    await expect(dialog.getByRole("textbox", { name: "Instructions for Claude Code" })).toBeVisible();
    expect(problems).toEqual([]);
  });

  test("adds a line, and the entries as they are found, while the other programs are being checked", async ({ page }) => {
    const dialog = await preferences(page, { mockProviders: "apple,claude:checking,agy:checking" });
    await expect(dialog.getByRole("combobox", { name: TITLE })).toHaveText("Apple Intelligence");
    await expect(dialog.getByRole("note")).toHaveText("Checking Claude Code and Antigravity…");
    await page.evaluate(() => window.__yellowmoon!.mockApi.setProviders({ apple: "available", claude: "available", agy: "signed_out" }));
    await expect(dialog.getByRole("note")).toHaveText("Antigravity is installed but not signed in. Open Terminal, run agy and sign in.");
    await dialog.getByRole("combobox", { name: TITLE }).click();
    await expect(page.getByRole("option")).toHaveCount(4);
  });

  test("keeps the row when the saved program is gone and nothing else is listed, and Use Default Auto withdraws the choice", async ({ page }) => {
    // Claude Code was saved and agreed to, then uninstalled; Apple Intelligence is not available: no provider is listed, but the choice is a program.
    const dialog = await preferences(page, { mockProviders: "apple:unavailable", mockAutoProvider: "claude", mockConsent: "claude" });
    await expect.poll(() => page.evaluate(() => window.__yellowmoon!.mockApi.autoStatus().then((s) => s.visible))).toBe(false);
    const row = dialog.getByRole("group", { name: TITLE });
    await expect(row).toBeVisible();
    await expect(dialog.getByRole("combobox", { name: TITLE })).toHaveText("Default Auto");
    const note = dialog.getByText("Your saved choice, Claude Code, isn't available now, so Default Auto is used until it is.");
    await expect(note).toBeVisible();
    const button = dialog.getByRole("button", { name: "Use Default Auto" });
    await expect(button).toBeVisible();
    // Beside the note: the same line.
    const [noteBox, buttonBox] = await Promise.all([note.boundingBox(), button.boundingBox()]);
    expect(buttonBox!.x).toBeGreaterThanOrEqual(noteBox!.x + noteBox!.width - 1);
    expect(Math.abs(buttonBox!.y + buttonBox!.height / 2 - (noteBox!.y + noteBox!.height / 2))).toBeLessThan(buttonBox!.height);
    await button.click();
    await expect.poll(() => saved(page)).toBe("yellowmoon");
    await expect(page.getByRole("dialog", { name: /^Send photos/ })).toHaveCount(0);
    // Withdrawn: nothing is left to show, so the row goes with its note.
    await expect(row).toHaveCount(0);
    await expect(note).toHaveCount(0);
    expect(await page.evaluate(async () => (await window.__yellowmoon!.mockApi.preferencesGet()).auto_consent)).toEqual({ claude: true, agy: false });
  });

  test("keeps the Use Default Auto button for a saved program that is signed out, beside the dropdown that shows it", async ({ page }) => {
    const dialog = await preferences(page, { mockProviders: "apple,claude:signed_out", mockAutoProvider: "claude", mockConsent: "claude" });
    await dialog.getByRole("button", { name: "Use Default Auto" }).click();
    await expect.poll(() => saved(page)).toBe("yellowmoon");
    await expect(dialog.getByRole("combobox", { name: TITLE })).toHaveText("Default Auto");
    await expect(dialog.getByRole("button", { name: "Use Default Auto" })).toHaveCount(0);
    // The row stays: Apple Intelligence and the unusable program are still listed.
    await expect(dialog.getByRole("group", { name: TITLE })).toBeVisible();
  });

  test("goes when the last provider leaves while Preferences is open", async ({ page }) => {
    const dialog = await preferences(page, { mockProviders: "claude" });
    await expect(dialog.getByRole("group", { name: TITLE })).toBeVisible();
    await page.evaluate(() => window.__yellowmoon!.mockApi.setProviders({}));
    await expect(dialog.getByRole("group", { name: TITLE })).toHaveCount(0);
  });
});

test.describe("the consent dialog", () => {
  const options = { mockProviders: "apple,claude,agy", mockAppleAuto: "answered" } as const;

  test("opens over Preferences when Claude Code is chosen for the first time, 480 px wide, focus on Cancel", async ({ page }) => {
    const problems = watchErrors(page);
    const dialog = await preferences(page, options);
    await dialog.getByRole("combobox", { name: TITLE }).click();
    await page.getByRole("option", { name: "Claude Code" }).click();
    const consent = page.getByRole("dialog", { name: "Send photos to Claude Code?" });
    await expect(consent).toBeVisible();
    await expect(consent).toContainText(CLAUDE_BODY);
    await expect(consent.getByRole("button", { name: "Cancel" })).toBeFocused();
    await expect(consent.getByRole("button", { name: "Use Claude Code" })).toHaveAttribute("data-slot", "button");
    // The layout width, not the box on screen: the dialog is still scaling up while it opens.
    expect(await consent.evaluate((el) => (el as HTMLElement).offsetWidth)).toBe(480);
    // Nothing has changed.
    expect(await saved(page)).toBe("apple");
    expect(problems).toEqual([]);
  });

  test("Use Claude Code saves the choice and the consent in one patch; the Automatic card then names it", async ({ page }) => {
    const dialog = await preferences(page, options);
    await dialog.getByRole("combobox", { name: TITLE }).click();
    await page.getByRole("option", { name: "Claude Code" }).click();
    await page.getByRole("button", { name: "Use Claude Code" }).click();
    await expect(page.getByRole("dialog", { name: "Send photos to Claude Code?" })).toHaveCount(0);
    await expect.poll(() => page.evaluate(async () => (await window.__yellowmoon!.mockApi.preferencesGet()).auto_consent)).toEqual({ claude: true, agy: false });
    expect(await saved(page)).toBe("claude");
    await expect(dialog.getByRole("combobox", { name: TITLE })).toHaveText("Claude Code");
    await expect(dialog.getByRole("textbox", { name: "Instructions for Claude Code" })).toBeVisible();
    // Asked once: another choice and back needs no second question.
    await dialog.getByRole("combobox", { name: TITLE }).click();
    await page.getByRole("option", { name: "Apple Intelligence" }).click();
    await dialog.getByRole("combobox", { name: TITLE }).click();
    await page.getByRole("option", { name: "Claude Code" }).click();
    await expect(dialog.getByRole("combobox", { name: TITLE })).toHaveText("Claude Code");
    await expect(page.getByRole("dialog", { name: /^Send photos/ })).toHaveCount(0);
  });

  test("Cancel keeps the previous choice, and so do Esc and a click outside, which close only the dialog", async ({ page }) => {
    const dialog = await preferences(page, options);
    const select = dialog.getByRole("combobox", { name: TITLE });
    const consent = page.getByRole("dialog", { name: "Send photos to Claude Code?" });
    const ask = async () => {
      await select.click();
      await page.getByRole("option", { name: "Claude Code" }).click();
      await expect(consent).toBeVisible();
    };
    await ask();
    await consent.getByRole("button", { name: "Cancel" }).click();
    await expect(consent).toHaveCount(0);
    await expect(select).toHaveText("Apple Intelligence");
    await ask();
    await page.keyboard.press("Escape");
    await expect(consent).toHaveCount(0);
    await expect(dialog).toBeVisible();
    await expect(select).toHaveText("Apple Intelligence");
    await ask();
    await expect(consent.getByRole("button", { name: "Cancel" })).toBeFocused();
    // Radix starts to listen for a click outside a tick after the dialog opens, which a busy machine can stretch: click until it is heard.
    await expect(async () => {
      await page.mouse.click(5, 5);
      await expect(consent).toHaveCount(0, { timeout: 1500 });
    }).toPass({ timeout: 15_000 });
    await expect(dialog).toBeVisible();
    expect(await saved(page)).toBe("apple");
    expect(await page.evaluate(async () => (await window.__yellowmoon!.mockApi.preferencesGet()).auto_consent)).toEqual({ claude: false, agy: false });
  });

  test("does not come back unasked when About from the native menu took its place", async ({ page }) => {
    const dialog = await preferences(page, options);
    await dialog.getByRole("combobox", { name: TITLE }).click();
    await page.getByRole("option", { name: "Claude Code" }).click();
    const consent = page.getByRole("dialog", { name: "Send photos to Claude Code?" });
    await expect(consent).toBeVisible();
    // The menu's About Yellowmoon: another dialog replaces Preferences, and the question goes with it.
    await page.evaluate(() => window.__yellowmoon!.mockApi.emitEvent("menu_action", { action: "about" } as never));
    const about = page.getByRole("dialog", { name: "About Yellowmoon" });
    await expect(about).toBeVisible();
    await expect(consent).toHaveCount(0);
    await page.keyboard.press("Escape");
    await expect(about).toHaveCount(0);
    // The next ⌘, opens Preferences alone, and nothing was saved.
    await page.keyboard.press("Meta+,");
    await expect(page.getByRole("dialog", { name: "Preferences" })).toBeVisible();
    await expect(page.getByRole("dialog", { name: /^Send photos/ })).toHaveCount(0);
    expect(await saved(page)).toBe("apple");
  });

  test("Antigravity's dialog says it is experimental, and a Windows build adds its sentence", async ({ page }) => {
    const dialog = await preferences(page, { ...options, mockOs: "windows" });
    await dialog.getByRole("combobox", { name: TITLE }).click();
    await page.getByRole("option", { name: /^Antigravity \(agy\)/ }).click();
    const consent = page.getByRole("dialog", { name: "Send photos to Antigravity?" });
    await expect(consent).toContainText("This is experimental. Yellowmoon itself makes no connection.");
    await expect(consent).toContainText(`Yellowmoon itself makes no connection. ${WINDOWS_SENTENCE}`);
    await consent.getByRole("button", { name: "Use Antigravity" }).click();
    await expect(dialog.getByRole("combobox", { name: TITLE })).toContainText("Antigravity");
    expect(await saved(page)).toBe("agy");
  });

  test("the sentence is not there off Windows", async ({ page }) => {
    const dialog = await preferences(page, options);
    await dialog.getByRole("combobox", { name: TITLE }).click();
    await page.getByRole("option", { name: "Claude Code" }).click();
    await expect(page.getByRole("dialog", { name: "Send photos to Claude Code?" })).not.toContainText(WINDOWS_SENTENCE);
  });
});

test.describe("the Automatic card and the Start screen", () => {
  test("name the program that grades: Claude Code, in the card and while the photos wait", async ({ page }) => {
    await openApp(page, { ...MOCKUP, mockProviders: "claude", mockAutoProvider: "claude", mockConsent: "claude", mockAppleAuto: "pending" });
    await page.getByRole("button", { name: "Open files…" }).click();
    await expect(page.getByRole("heading", { level: 2, name: "Grading 12 photos with Claude Code" })).toBeVisible();
    await expect(page.getByText("Claude Code is looking at each photo. You can start looking while it finishes.")).toBeVisible();
    await expect(page.getByText("Graded by Claude Code…").first()).toBeVisible();
    await page.getByRole("button", { name: /Open photos/ }).click();
    const card = page.getByRole("region", { name: "Automatic" });
    await expect(card.getByText("Claude Code is looking at this photo…")).toBeVisible();
    await page.evaluate(() => window.__yellowmoon!.mockApi.setAppleAuto("answered"));
    await expect(card.getByText("Set with Claude Code.")).toBeVisible();
    await page.evaluate(() => window.__yellowmoon!.mockApi.setAppleAuto("failed"));
    await expect(card.getByText("Claude Code couldn't look at this photo, so Default Auto is used.")).toBeVisible();
  });

  test("say a program that was chosen is not available right now", async ({ page }) => {
    await openApp(page, { ...MOCKUP, mockProviders: "claude:signed_out", mockAutoProvider: "claude", mockConsent: "claude" });
    await openPhotos(page);
    const card = page.getByRole("region", { name: "Automatic" });
    await expect(card.getByText("Claude Code isn't available right now, so Default Auto is used.")).toBeVisible();
    expect(await card.locator(".ym-auto-row__label").allTextContents()).toEqual(["Exposure", "White", "White balance", "Noise", "Camera"]);
    await page.evaluate(() => window.__yellowmoon!.mockApi.setProviders({ claude: "available" }));
    await expect(card.getByText("Set with Claude Code.")).toBeVisible();
  });
});
