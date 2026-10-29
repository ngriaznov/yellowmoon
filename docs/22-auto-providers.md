# 22 · Auto providers

Auto can be graded by one of four providers: **Default Auto** (the rule, `docs/02`), **Apple Intelligence** (the on-device model, `docs/19`), **Claude Code** and **Antigravity (agy)**, the last two being command-line programs the person installed and signed in to themselves. Yellowmoon starts them as child processes; they talk to their own services. Yellowmoon opens no network connection (hard rule 7, read as the owner reads it, section 2).

This file covers: what each provider is (section 1), what leaves the computer and the terms (2), when a provider is available (3), how Yellowmoon finds a program (4), how it runs one (5), the picture and the answer's form (6), the request to each program (7, 8), failures, the breaker, concurrency and cancelling (9), the preference, the consent and which provider acts (10), the core's API (11), the app (12), `ym` (13), reports and metadata (14), the new crate and its test doubles (15), the tests (16), what is provisional (17), the plan for three parallel lanes (18) and what a signed-in machine must still check (19).

**What stays in `docs/19`.** Everything about the picture the model sees (section 3), the steps and the table that turns them into values (sections 4 and 5), the worker's order, the viewer not waiting, exports and `ym` waiting, bursts, removing photos and the instructions' rules (section 6) is provider-neutral and holds for all three graders. `docs/19` section 2's switch, section 8's `--apple-auto`, section 9's `ym:AppleAuto` and section 10's Preferences row are replaced by sections 10, 13, 14 and 12 of this file. `PIPELINE_VERSION` and `TABLE_VERSION` (2) do not change: with Default Auto the pixels are the same as before.

**A word on names.** In the code and in the wire types the grading is still called "apple" (`ym_core::apple_auto`, `AppleAuto`, `AutoEstimates::apple`, `apple_steps`, `PhotoStage::Apple`, `apple_pending`, `estimates.apple.state`): in those names "apple" means "the chosen provider's grading", and `AdvisorInfo::provider` says whose. No mass rename (decision D3, below). Everything new, and everything a person or a script sees, uses the neutral names: `auto_provider`, `AutoProvider`, `ym:AutoAdvisor`, `auto_advisor`, `--auto-provider`.

## 0. Decisions

### 0.1 The owner's decisions (2026-10-26), binding

| # | Decision |
|---|---|
| O1 | "Additionally to Apple Intelligence I want to support Google AGY CLI and Claude Code (Claude CLI). We should see a new dropdown in settings to choose from; if nothing available - dropdown and prompt is hidden." "Make sure that accessing CLIs is cross-platform." |
| O2 | Rule 7: "Ym won't access network, we will invoke agy and Claude - they will access network." Yellowmoon (the app and `ym`) opens no connection; it starts the person's own program, which does. The settings text still says plainly what the program sends (the 768 px picture and the instructions) and to whom. No HTTP client may enter Yellowmoon's dependency tree; a test checks the new crate's. |
| O3 | A cloud provider is never chosen by default: the default is Apple Intelligence where it is available, otherwise Default Auto. Choosing Claude Code or Antigravity for the first time asks for consent (a dialog saying what is sent where; Cancel keeps the previous choice). |
| O4 | A program that is installed but not signed in is listed **disabled** with a hint ("Claude Code is installed but not signed in. Open Terminal, run claude and sign in."). The dropdown and the instructions are hidden only when none of the three providers is installed or available. |
| O5 | `agy` gets the picture as an `@photo.jpg` mention in its prompt (the picture written to an empty private temp folder that is the working directory), runs sandboxed, never with permissions skipped, and the dropdown marks it **Experimental** ("add experimental note in UI"), with a note that Google's Antigravity terms apply and its agent may open other tools. Claude Code gets the picture as a base64 image block on stdin, with `--tools ""`, `--json-schema`, `--safe-mode`, `--no-session-persistence`, telemetry off, an empty temp folder as working directory. Neither route touches the person's credentials. |
| O6 | The UI may say "Claude Code" and "Antigravity (agy)" in plain text as what Yellowmoon runs. No logos. Never "ARRI" or "ALEXA" as a name. |
| O7 (2026-10-29) | "The fix should be only instantiate CLIs when we need grading and immediately close them after finish." Asked when Yellowmoon may start Claude Code or `agy` apart from grading itself, the owner chose **"Preferences + before grading"**: nothing runs at app launch; opening Preferences checks the installed programs (version and sign-in) so the dropdown can show "not signed in" with its hint; the chosen program is checked once right before its first grading; grading itself is one process per photo, closed when it answers. **This replaces what D7 and section 3.3 said about the app's launch probe** (`StartProbe::Now`): the app builds its advisors with `StartProbe::Never`, as `ym` does, and the read points are the table of section 3.3. Because nothing is checked at launch, Preferences can open before a program has answered; section 12.1 says what the row shows meanwhile. |

### 0.2 Decisions this spec makes where the brief left a choice

| # | Decision | Reason |
|---|---|---|
| D1 | The instructions keep their stored key `apple_auto_instructions` (one shared text for all providers, the same default text, the same 2000-character limit). | A rename would need a migration, a wire change and test churn and buys nothing the person sees. |
| D2 | Reports migrate: the XMP property `ym:AppleAuto` becomes `ym:AutoAdvisor` at once (no overlap); `ym:Auto` widens to `rule`, `apple`, `claude`, `agy`; in `ym --json` the new key `auto_advisor` is written and the old `apple_auto` stays for one version as an Apple-only view (section 14). | Nothing of ours reads `ym:AppleAuto` back, so a second copy in every file would only bloat it; scripts read `--json`, so that gets one version of grace. |
| D3 | `apple_auto.rs` keeps its name and the "apple" names stay in the core, the wire and the session; the new neutral concepts live in a new module, `ym_core::auto_provider`. | The smallest diff that stays clear; one sentence above says what "apple" means in those names. |
| D4 | A struct, `ProviderAdvisors` (three optional slots: apple, claude, agy), is the registry; the session holds it and the chosen provider is session state. | Three fixed providers need no map; the struct is easy to build in tests. |
| D5 | No retries for the two programs. Instead each advisor has a breaker: three failures in a row (`rate_limited`, `timeout`, `cli_failed`) make it answer `unavailable` at once for 60 s, then one trial request goes through. | The programs retry transient API errors themselves, a plan's limit outlasts any pause we could add, and 500 photos at 90 s each would otherwise keep a batch busy for hours. |
| D6 | The advisor says how many requests it takes at once (`AutoAdvisor::concurrency`, default 1: Apple 1, Claude Code 2, agy 1), and the session runs that many workers. | Process start-up is about a second of each request; two at once halve a batch. |
| D7 | Availability is cached; probes run on a background thread with timeouts and tell the core when the answer changes (`on_change`). `refresh()` never blocks; `settle(timeout)` waits (for `ym`). The sign-in check of `agy`, the only probe known to make its program contact a network, runs only when a full check is asked for (Preferences opening, or the first grading of a chosen `agy`), never at launch. **Which read points start a probe is O7 and section 3.3: none at launch.** | The main thread must never wait for a process, and a program must not talk to the network at every launch of Yellowmoon. |
| D8 | Discovery order: the known install folders, then `PATH`, then the login shell (macOS and Linux); the first candidate whose version is high enough is used. | The research's order; a stale copy that is too old does not hide a newer one. |
| D9 | Claude Code's answer without `structured_output` is `bad_answer`, and `refused` only when its `stop_reason` is `refusal`. | The envelope has no other sign of a refusal. |
| D10 | A `not_signed_in` failure is not stored as the photo's failure when the program is found signed out on the re-read: the request is abandoned and the photo waits, as before a call. | One sign-out must not fail every photo of a batch and then hide behind the failures. |
| D11 | The temporary folders are made in the operating system's temp folder (private, mode 0700, per user). On Windows the path carries the user's name, which the program may send with its environment; this is accepted and said in the copy of the consent dialog only for Windows. | `%ProgramData%` would be readable by other users, which is worse for a photo. |
| D12 | `agy`: no `--model` unless a developer sets one, `--effort low`, `--print-timeout 110s`, `--sandbox`, `--disable-slash-commands`; the prompt names `@photo.jpg`. Our own hard stop is 120 s. | The brief: agy's default model; the CLI's own timeout must fire first so that its partial output and exit code are readable. |
| D13 | The picture is a baseline JPEG, 4:4:4, quality 92, Huffman tables not optimised, no metadata. | `jpeg-encoder` then writes one interleaved scan, which every decoder reads; the same bytes every run. |
| D14 | The answer is validated strictly against the schema, after `null` values and unknown keys are removed. | A model that writes `"exposure": null` or an extra note must not fail a photo. |
| D15 | `apple_auto` (the boolean) is read once for the migration and never written again; consent is saved with the choice in one patch; the core refuses a patch that chooses a cloud provider without consent. | No mirror to keep in step; one atomic save; a frontend bug cannot skip the consent. |
| D16 | `ym` follows the saved choice only when its consent is saved; naming a provider with `--auto-provider` is the consent for that run and is never saved. | "ym never uses a cloud provider unasked." |
| D17 | Two test doubles, `ym-fake-claude` and `ym-fake-agy`, small Rust binaries of the new crate, driven by environment variables. | They run the same on Linux, macOS and Windows. |
| D18 | Model and effort have developer knobs in the environment (`YELLOWMOON_CLAUDE_MODEL`, `YELLOWMOON_CLAUDE_EFFORT`, `YELLOWMOON_AGY_MODEL`, `YELLOWMOON_AGY_EFFORT`), never a preference. | The blind A/B needs them (section 17); nothing else may persist (hard rule 4). |
| D19 | Prompt versions are per provider (Apple 3, Claude Code 1, agy 1), the schema has its own version (1), and all are constants in `ym_core::auto_provider`. | Reports need them without a provider being available. |

## 1. The providers

| Id (wire) | Name in the UI | What runs | How the picture gets there | At once | Hard stop | Answers repeat? | Marked |
|---|---|---|---|---|---|---|---|
| `yellowmoon` | Default Auto | the rule (`docs/02`) | none | none | none | yes, everywhere | none |
| `apple` | Apple Intelligence | the on-device model, through `ym-apple` (`docs/19`) | a `CGImage` in the Swift bridge | 1 | 30 s | on one Mac and macOS build (greedy sampling) | none |
| `claude` | Claude Code | the person's `claude` program | a base64 image block on stdin (section 7) | 2 | 90 s | no | none |
| `agy` | Antigravity (agy) | the person's `agy` program | `photo.jpg` in the working folder, named as `@photo.jpg` in the prompt (section 8) | 1 | 120 s | no | **Experimental** |

`ym_core::auto_provider::AutoProvider` is the enum (`Yellowmoon`, `Apple`, `Claude`, `Agy`; JSON `"yellowmoon"`, `"apple"`, `"claude"`, `"agy"`). It has `name()` ("Default Auto", "Apple Intelligence", "Claude Code", "Antigravity"), `is_cloud()` (Claude and Agy: the picture leaves through a program), `timeout_s()` (30, 90, 120), `concurrency()` (the default the advisors report: 1, 1, 2, 1), `prompt_version()` (0, 3, 1, 1) and `slot()` (0 for apple, 1 for claude, 2 for agy; the session's per-provider answers use it).

The default model and effort are provisional until a blind A/B per provider (section 17): Claude Code `sonnet` at `--effort low`; agy's own default model at `--effort low`.

## 2. Rule 7, privacy and terms

### 2.1 What Yellowmoon does and never does

- **It starts a program and talks to it through a pipe or a file.** It opens no socket, uses no HTTP client, and holds no URL except the documentation links the interface opens in the system browser. `crates/ym-agents/tests/no_network.rs` runs `cargo tree -p ym-agents -e normal --prefix none` and fails when the closure holds any of `reqwest`, `hyper`, `h2`, `ureq`, `curl`, `isahc`, `attohttpc`, `surf`, `awc`, `native-tls`, `openssl`, `rustls`, `tungstenite`; the same test checks `ym-cli`'s tree. `jsonschema` is used with `default-features = false`, which keeps its HTTP and file reference resolvers out. The updater stays the only network code of the app (hard rule 7); the app's tree holds its HTTP stack, `ym`'s does not.
- **It never touches the person's credentials.** It does not read `~/.claude`, `~/.gemini`, `%APPDATA%\Claude`, the keychain, a credential file or the value of `ANTHROPIC_API_KEY`, `GEMINI_API_KEY` or any `CLAUDE_CODE_*`/`AGY_*` variable. It does not sign anyone in (no `claude auth login`, no `claude setup-token`, no `agy` login flow), does not modify the program, does not use `--bare` (which would ignore the person's subscription sign-in) and never passes `--dangerously-skip-permissions`. The environment is inherited so that the person's own provider settings (a proxy, Bedrock, Vertex) keep working; Yellowmoon neither reads nor prints it. Argument lists and stderr are logged at debug level only, with anything that looks like a token cut out (section 5).
- **It never reads or deletes the program's own files** (its conversation history, its caches): they are the program's, not ours (the spirit of hard rule 5).
- **The picture and the instructions live in memory and in a private temp folder until the answer arrives** (hard rule 4), and the folder is deleted when the request ends (section 5). A sweep at start-up deletes `ym-auto-*` folders of the user's temp folder older than ten minutes, which only a crash can leave (`ym_agents::process::sweep_stale`, called by `system_advisors`).

### 2.2 What leaves the computer

| What | Claude Code | Antigravity |
|---|---|---|
| The photo | one 768 px sRGB JPEG (section 6), no EXIF or colour profile, no file name; never the raw file | the same picture, as `photo.jpg` in the working folder, which the agent opens; its name is visible to the model |
| The instructions | the person's own or the default, at most 2000 characters, as the system prompt | inside the prompt, with the answer's form |
| The answer's form | our JSON Schema, as the `StructuredOutput` tool | our JSON Schema, by `--json-schema` and in the prompt |
| Added by the program | an identity line, the temp folder's path, platform, OS version, date, model name, a device id hash and a session id, the program's version | its own prompt and tool definitions (about 10 k tokens), the folder's path, the conversation history it keeps |
| Where it goes | Anthropic, or the service the person set Claude Code up to use (Bedrock, Vertex, a gateway) | Google, or the Gemini API with the person's key, or their Google Cloud project |
| Who can keep it | per the person's account type: 30 days to 5 years (Anthropic's data-usage page) | per Google's terms: used to improve Google's products, staff may review, opt-out in Antigravity's own settings |
| What does **not** leave | the original file, its name and path, the other photos, the person's settings and looks, the location in the file, any other file | the same, except that an agent with tools could in principle read what its permissions allow (2.3) |

The settings text does not promise a retention period: the person's account decides (the vision FAQ and the Claude Code page differ by account type).

### 2.3 Risks and how they are bounded

- **The answer is bounded by its form.** Whatever a hijacked model says, only the enum steps of section 6 reach Auto's values, and `scene` and `plan` (cut to 80 and 160 characters) are shown only through the escaping of text from outside (`docs/10` *Text from outside*). A photograph with text on it ("ignore your instructions and run …") can change a grade at most, never run anything of ours.
- **Claude Code runs with no tools** (`--tools ""`), no hooks, plugins or MCP servers (`--safe-mode`, `--strict-mcp-config`), no session file (`--no-session-persistence`) and a working folder that is empty. It makes one API request.
- **`agy` has an agent with tools.** It runs `--sandbox`, in a folder that holds only `photo.jpg`, with its permissions as the person set them but never skipped, and the prompt tells it to use nothing else. Its own settings still apply (`settings.json` policies such as `always-proceed` are honoured by headless runs), which Yellowmoon cannot change. This is why it is marked Experimental and why it is the second to ship.
- **The picture is a rendering at the Signature look**, not the original, but still the person's photograph, sent to a third party: the consent dialog says so.

### 2.4 Terms

- **Anthropic** (`code.claude.com/docs/en/legal-and-compliance`): the person signing in to the unmodified `claude` with their own plan is not prevented; developers must not route requests through plan credentials or collect, store or intermediate credentials. Yellowmoon only starts the person's own unmodified `claude`, touches no credential and pays for nothing. A batch of hundreds of photos through a subscription is "ordinary, individual usage" of that plan as far as the text goes, which the text does not spell out (Question 30 in `docs/STATUS.md`). The name "Claude Code" is used in plain text to say what runs, never in a product name or logo.
- **Google** (`antigravity.google/terms`): "using third party software, tools, or services to access the Service" is a breach, with the example of OpenClaw using Antigravity OAuth. Starting the first-party `agy` is not OAuth in third-party software, but the sentence is broad. **The owner's answer (2026-10-26): "agy can reference photo via mention @ character in prompt, you can add experimental note in UI"**: the route ships, marked Experimental, and the consent dialog and the hint say that Google's Antigravity terms apply. Asking Google remains an optional [HUMAN] item (`docs/STATUS.md`).

## 3. Availability

### 3.1 States

Each provider is in one **state** with a **reason** when it is not available (`ProviderState` and `UnavailableReason`, JSON snake_case):

| State | Reason (wire) | Meaning | Listed in the dropdown |
|---|---|---|---|
| `available` | none | the provider can answer now | selectable (Apple: only here) |
| `not_installed` | `not_installed` | no usable program was found (section 4) | not listed |
| `too_old` | `too_old` | the program runs but its version is below the minimum | listed, disabled |
| `signed_out` | `signed_out` | it runs and is not signed in | listed, disabled |
| `unavailable` | `probe_failed` | the program was found but its check failed, timed out or printed what we cannot read | listed, disabled |
| `unavailable` | `unsafe_launcher` | Windows: only a `.cmd`/`.bat` shim was found and no real `claude.exe` next to it (section 4) | listed, disabled |
| `unavailable` | Apple's nine reasons (`docs/19` section 1: `not_a_mac`, `intel_mac`, `built_without_support`, `macos_too_old`, `device_not_eligible`, `apple_intelligence_off`, `model_not_ready`, `unavailable`, `no_image_input`) | Apple Intelligence cannot answer | not listed (as `docs/19` section 2: the row is hidden) |
| `checking` | `checking` | the program was never checked (nothing runs when the app starts, 3.3) or its first check has not finished | not listed yet (the saved program is shown, disabled, with the tag "Checking…": 12.1) |

`not_consented` is one more reason, which only `Activity::decide` gives (section 10.4): a cloud provider that is available but whose consent is not saved acts as Default Auto. It is never a provider's own state.

`ProviderState::of(&Availability)` maps a reading to a state: available; `not_installed`, `too_old`, `signed_out` and `checking` to themselves; every other reason to `unavailable`. Default Auto is always `available`.

**The dropdown is visible** (`AutoStatus::visible`) when Apple Intelligence is available or any of Claude Code and Antigravity is listed (that is, in the state `available`, `too_old`, `signed_out` or `unavailable`). Otherwise it and the instructions are hidden, as the Apple row is today on a Mac without Apple Intelligence, and Default Auto is the only Auto.

### 3.2 Conditions

Checked in this order; the first that fails gives the state.

| Provider | Condition | State when it fails |
|---|---|---|
| Apple | `docs/19` section 1, unchanged | `unavailable` with Apple's reason |
| Claude Code | 1. a program is found (section 4) | `not_installed` (`unsafe_launcher` when only a `.cmd` shim exists) |
| | 2. `claude --version` prints a version at least `2.1.169` (the first with `--safe-mode`) | `too_old`; the version is kept for the hint; `probe_failed` when it prints none or times out |
| | 3. `claude auth status` (JSON on stdout) exits 0 with `"loggedIn": true` | `signed_out` when it prints `"loggedIn": false` (exit 1); `probe_failed` otherwise |
| agy | 1. a program is found | `not_installed` |
| | 2. `agy --version` prints a version at least `1.2.14` (the schema rules and exit code 3) | `too_old`; `probe_failed` |
| | 3. `agy -p /usage --output-format json` in a new empty temp folder exits 0. This is the one check known to make its program contact its service (it asks for the quota), so it runs only in a full check (3.3: Preferences opening, or the first grading of a chosen `agy`, which contacts its service anyway), never at launch. **Unverified** (section 19) | `signed_out` when the output or stderr says `authentication required`; `probe_failed` on any other failure or on the timeout; until a check has run the state is `checking` |

`claude auth status` also gives `authMethod` and `apiProvider`; they are kept (`ToolInfo::auth`, `ToolInfo::api`) so that the hint can say where the picture goes ("Anthropic" for `firstParty`, otherwise "the service your Claude Code is set up to use") and `ym doctor` can show them.

**Probe time limits:** the shell lookup 3 s per attempt (section 4), a `--version` 5 s, `claude auth status` 10 s, the `agy` usage check 15 s. A probe that runs out of time kills its process tree (section 5) and gives `probe_failed`. All probes run in the new empty temp folder, with the environment of section 5.

### 3.3 When it is read

As `docs/19` section 1 for Apple Intelligence, whose availability is an operating-system call. **For the two programs, nothing runs when the app starts** (owner, O7, 2026-10-29: "only instantiate CLIs when we need grading and immediately close them after finish"): a program is started when someone needs it, and a program that was not chosen is started only when Preferences opens. The read points:

| When | What is started | Where |
|---|---|---|
| The app starts, the saved choice and consent are read (`Core::open`), the choice or the consent changes (`Core::preferences_set`, `ym`'s `--auto-provider`) | **nothing.** The core only reads what each advisor last said; a program that was never checked says `checking`. A change of the choice or the consent is made in Preferences, which has just checked every program, and `ym` checks the program it names before its core opens (13.1). | `Core::open`, `Session::change_choice` |
| Preferences opens (`Core::auto_status`, the command `auto_status`) | **every program**, the person's own check (`AutoAdvisor::refresh_asked`: version, sign-in; for `agy` the check that contacts its service). This is how the dropdown can show "Not signed in" with its hint. | `Session::auto_status` |
| Photos are opened (`Session::open`: the app's `open_paths`, `ym render`, `ym info`) | **the chosen program**, when it could act (below) and what it last said is not `available`: it was never checked (it reads `checking`), or the last check found it signed out, missing, too old or failing (one check per opening is enough). Once it said `available` a later opening starts nothing. | `Session::refresh_apple_auto` |
| An export starts (`Core::export_start`, `ym render`) | **the chosen program, only when it was never checked** (it still reads `checking`; a check that is running serves every caller). | `Session::refresh_auto_for_export` |
| A request (one photo) | **one process**, the request itself, closed when it answers; its success renews the cached `available`, `not_signed_in` says signed out (below). | 5, 9 |
| An advisor says its availability changed | nothing: the session reads what the advisors say. | `Session::read_auto_availability` |
| `ym doctor` | every program (13.3) | |

The difference for the two programs is that **a read never waits for a process**:

- `AutoAdvisor::availability()` returns the last probe's result at once (property reads, a lock). An advisor that no probe has run for reads `checking`.
- `AutoAdvisor::refresh()` starts a probe on a background thread and returns. A call within 5 s of the last probe's end, or while a probe runs, does nothing (one probe at a time; the answer of the running probe serves every caller). `ym_agents::system_advisors(StartProbe::Never)` is what **both** the app and `ym` pass: no probe starts when the advisors are built, so a run that does not use a program never starts one. (`StartProbe::Now`, which starts the first probe when it builds the advisors, is kept for tests and tools; no front end passes it, because it would start the programs at every launch whatever the person chose.)
- The check of a provider that is chosen, for photos that are about to be graded, starts when the chosen provider reads anything but `available` (the first reading is `checking`, and a probe that was cancelled half-way leaves the reading as it was), and the Auto workers wait until it ends: a provider that reads `checking` is not asked, and a photo is served Default Auto's values only until it answers, as for any provider that is not available yet (the viewer never waits, `docs/19` section 6). The check takes a second or two (the login shell's look-up when the program is in none of the known folders, `--version`, `claude auth status`).
- When a probe changes the cached result, the advisor calls the listener the core registered (`AutoAdvisor::on_change`). The core then re-reads what the advisors say (`Session::read_auto_availability`: no process, and **no probe**: it reads `AutoAdvisor::availability()` and does not call `refresh()`, so a request-time sign-out and a probe cannot feed each other) and the app tells the webview (`auto_status_changed`, section 12.6), which reads `auto_status_cached()` (no program is asked to probe).
- `AutoAdvisor::settle(timeout)` waits until no probe runs or the time is up. Only `ym` and the export use it, and only on the **chosen** provider's advisor (`ProviderAdvisors::settle_provider`, `Core::settle_auto`): `ym` refreshes the chosen provider and settles it before the core opens, so a run never starts with `checking`, and an export started right after launch with `agy` chosen does not spend its 5 s on Claude Code's check. `ProviderAdvisors::settle` (every advisor, one shared deadline) is for `ym doctor`, which refreshes every one.
- **Which program.** Preferences opening (`Core::auto_status`) refreshes **all** advisors, which is when `agy`'s sign-in check runs for a program nobody chose. Every other read point starts the check of the **chosen** provider's advisor alone, **and only when that provider could act**: a cloud program that the run did not name and whose consent is not saved is not probed at all (`AppleState::chosen_may_act`, `Session::refresh_apple_auto`; 10.2, 13.1). Its availability stays what it last said (`checking` before a first probe) and the photos read `not_consented` (10.4), the same on every run: a saved cloud choice nobody agreed to starts no program, whatever the command.
- After the run is halted (9.4) nothing is probed either: the registry's `refresh` and the session's reads do nothing.
- A successful request renews the cached `available`. A request that fails with `not_signed_in` sets the cached state to `signed_out` and calls the listener at once (section 9). **That sign-out is held against the probes:** a probe only reads the local sign-in state (`claude auth status` says `loggedIn: true` from stored credentials while the service answers 401; `agy`'s `/usage` exits 0 while requests answer 403), so its `available` does not undo it, and a photo is not asked again and again with the picture sent each time. The hold ends when a request succeeds, when 10 minutes have passed since the last sign-out a request found (`Tuning::signed_out_hold`), or when the person asks for a new check. The person's own checks are Preferences opening and `ym doctor`, which refresh every advisor at once (`ProviderAdvisors::refresh(None)` calls `AutoAdvisor::refresh_asked`); the automatic ones (photos opening, an export starting: `refresh`) leave the hold standing. A probe that `cancel()` stopped half-way leaves the cached reading as it was.
- The worker reads the availability again before each request (as `docs/19` section 6): a photo is not failed when the provider stopped being available before its call; it waits for the provider to be back.
- A state of `not_installed` is searched again by the next check (Preferences opening, or photos opening), so a person who installs Claude Code and opens Preferences sees it within a few seconds, with no polling.
- **An export in the first seconds.** `Core::export_start` runs on a blocking thread in the shell and `ym render` on the main thread of a process with nothing else to do; when the chosen provider still reads `checking` (and could act: see above) they ask it to check and call `settle(5 s)` on it first, so an export started at once does not skip the provider by accident. A program that was checked and found signed out, missing or failing is not checked again for an export: the export starts with what is known.
- **A program is closed when it has answered.** One request is one process (`claude -p`, `agy -p`), never a resident one: when it exits, the run kills what it left in its process group, reaps it and removes its temporary folder (section 5), and nothing of a request outlives its answer (`tests/process.rs`: `a_program_that_exits_leaves_nothing_of_its_tree_behind`). The programs of a check (`--version`, `claude auth status`, `agy -p /usage`) end the same way.

### 3.4 What the person is told

The wording is fixed here (`ym_core::auto_provider::unavailable_hint`, `unavailable_words`); the app and `ym` use the same functions.

| State and reason | The hint (a sentence with its action; Preferences shows it under the row, `ym` prints it as the warning) | The clause (`ym doctor`, `ym info`: "Not available (<clause>)") |
|---|---|---|
| Claude Code `signed_out` | Claude Code is installed but not signed in. Open Terminal, run claude and sign in. | Claude Code isn't signed in |
| Claude Code `too_old` | Claude Code {version} is too old. Update it: open Terminal and run claude update. | Claude Code {version} is too old |
| Claude Code `probe_failed` | Claude Code didn't answer its check. Open Terminal, run claude to see if it starts, then open Preferences again. | Claude Code didn't answer its check |
| Claude Code `unsafe_launcher` | Yellowmoon found Claude Code only as a script it can't run safely. Install the native version of Claude Code. | Claude Code was found only as a script that Yellowmoon can't run safely |
| Claude Code `not_installed` | (not shown) | Claude Code isn't installed |
| Antigravity `signed_out` | Antigravity is installed but not signed in. Open Terminal, run agy and sign in. | Antigravity isn't signed in |
| Antigravity `too_old` | Antigravity {version} is too old. Update it: open Terminal and run agy update. | Antigravity {version} is too old |
| Antigravity `probe_failed` | Antigravity didn't answer its check. Open Terminal, run agy to see if it starts, then open Preferences again. | Antigravity didn't answer its check |
| Antigravity `not_installed` | (not shown) | Antigravity isn't installed |
| either, `checking` | (not shown) | Yellowmoon is still checking |
| either, not consented | (the consent dialog asks) | you haven't agreed to send photos to {name} yet |
| Apple | as `docs/19` section 8 (*Reasons in words*) | as there |

`{version}` is the version the program printed. On `windows` the word "Terminal" in these hints is "a terminal window"; the function takes `std::env::consts::OS` into account, tested for all three.

## 4. Discovery

`ym_agents::discovery` finds the executable of a tool (`Tool::Claude`, `Tool::Agy`). The result is an absolute path to a real executable; a script is never returned.

**Candidates, in this order, without duplicates (compared by canonical path):**

1. **Known folders.**

| Tool | macOS and Linux | Windows |
|---|---|---|
| `claude` | `~/.local/bin/claude` (the native installer's link), `~/.claude/local/claude` (an older local install; not named by the research, harmless when absent), `/opt/homebrew/bin/claude`, `/usr/local/bin/claude`, `/usr/bin/claude`, `~/.npm-global/bin/claude`, `~/.bun/bin/claude` | `%USERPROFILE%\.local\bin\claude.exe`, `%LOCALAPPDATA%\Microsoft\WinGet\Links\claude.exe`, `%APPDATA%\npm\claude.cmd` (resolved as below) |
| `agy` | `~/.local/bin/agy` | `%LOCALAPPDATA%\agy\bin\agy.exe` |

(Sources: both installers put the program in `~/.local/bin`, which a Finder- or Dock-started macOS app does not have on its `PATH`; `agy`'s installer writes `%LOCALAPPDATA%\agy\bin\agy.exe` on Windows and takes `--dir` for another place, which the `PATH` and shell steps find. The Homebrew, npm and bun folders are the usual global bin folders; the research infers them and the docs name no path.)

2. **`PATH`.** `which::which_in_global` over the `PATH` value: the first match that is an **absolute** path. A relative entry (`.`, or an empty one on Unix) is skipped, and so is its match, never the current folder: on Windows `.\claude.exe` of the folder `ym` was started in must not be run. On Windows the extension comes from `PATHEXT`.
3. **The login shell, macOS and Linux only**, and only when steps 1 and 2 gave no candidate that passes the version check. `"$SHELL" -l -c 'command -v <tool>'`, stdin `/dev/null`, stdout read to 64 KB, stderr discarded, a **3 s** time limit, the whole process group killed on expiry. The shell is started in a **new session** (`setsid`; `RunSpec::new_session`), so that it has no controlling terminal: an interactive zsh or bash that finds itself in a background process group of the terminal stops itself (`SIGTTIN`) and would wait out the limit. When it prints nothing usable, a second attempt `"$SHELL" -i -l -c '…'` (zsh and bash only: interactive start-up reads `.zshrc`, where nvm and many `PATH` edits live) with the same limit. `$SHELL` must be one of `sh`, `bash`, `zsh`, `dash`, `ksh`, `fish` by file name (other shells, such as `nu`, are skipped). The answer is the **last** output line that is an absolute path to an existing executable file (banners and warnings come first); nothing is evaluated. The person's start-up files run arbitrary code (their own) and may be slow: that is why this is last and bounded.

**Windows scripts.** A `.cmd`, `.bat` or `.ps1` is never started and nothing goes through `cmd /C` (Rust escapes arguments for batch files unsoundly, CVE-2024-24576, and a JSON schema and long instructions would be refused or mangled). When the candidate is `claude.cmd`, the real program is looked for next to it at `node_modules\@anthropic-ai\claude-code\bin\claude.exe` and used when it exists; otherwise the candidate is dropped, and when nothing else is found the state is `unavailable` / `unsafe_launcher`. `agy` ships as `agy.exe` and needs no such step.

**Validation.** Each candidate in order runs `--version` (5 s); its first `\d+\.\d+\.\d+` is the version. The first candidate whose version is at least the minimum is used. If none passes: `too_old` when at least one ran and printed a version (the highest is reported), `probe_failed` when candidates exist but none printed one, `not_installed` when there are none. The chosen path and version are cached with the probe and shown by `ym doctor` (the path is never written to a file or to a report; section 14).

**Tests:** a fake `claude` in a temp folder reached by each step (the known-folder step and `PATH` on all systems; the shell step on Unix with a script standing in for `$SHELL`, one that prints a banner first, one that hangs past its limit); a too-old copy first and a good copy second; on Windows a `claude.cmd` alone (`unsafe_launcher`) and with the npm layout (resolved).

## 5. Running a program

`ym_agents::process::run(RunSpec) -> RunOutput` is the one place that starts a child. A `RunSpec` has the program, the arguments, the environment changes, the working folder, the bytes for stdin, the time limit, the caps and a cancel token.

- **Start.** `process_wrap::std::CommandWrap` over `std::process::Command` with stdin, stdout and stderr piped, wrapped in `ProcessGroup::leader()` on Unix and `JobObject` plus `CreationFlags(CREATE_NO_WINDOW)` on Windows, and a guard of our own that kills the tree when it is dropped (*As built*, 15.7: the standard-library API of `process-wrap` 10 has no `KillOnDrop` and its `JobObject` does not set `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`; `CreationFlags` and `JobObject` may be added in either order, the crate merges their flags). A program that cannot be started is `SpawnFailed(error)`. The probes and the shell lookup go through the same function.
- **Arguments.** The picture and the instructions never travel in arguments (Windows allows about 32 K characters), except that `agy`'s prompt (at most about 6 KB) does, as its flag requires. Claude Code's schema (under 4 KB) is one argument. Arguments are passed as separate `OsString`s, never through a shell.
- **Environment.** Inherited, then for **every** child (probes included): `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1`, `DISABLE_TELEMETRY=1`, `DISABLE_ERROR_REPORTING=1`, `DISABLE_AUTOUPDATER=1` (the last three for versions that predate the first), and `NO_COLOR=1`; and removed: `CLAUDECODE`, `CLAUDE_CODE_ENTRYPOINT`, `CLAUDE_CODE_SESSION_ID`, `CLAUDE_CODE_CHILD_SESSION` and every variable starting `CLAUDE_CODE_REMOTE`, which describe a parent Claude Code session when `ym` runs inside one (the list is unverified, section 19). For `agy` only the `NO_COLOR` change applies: no telemetry switch is documented for it. Nothing else is added, and nothing is printed.
- **Working folder.** `tempfile::TempDir::with_prefix("ym-auto-")` in `std::env::temp_dir()` (mode 0700 on Unix, the user's own on Windows, D11), made per request. It holds `cwd/` (the working folder; empty for Claude Code, only `photo.jpg` for `agy`) and `in/` (`instructions.txt` for Claude Code, `schema.json` for `agy`), so that the working folder holds nothing the program does not need. Probes use an empty `cwd/` of their own. The folder is deleted when the request ends, on every path including a panic (the `TempDir` drops).
- **stdin** is written from its own thread, then closed; a broken pipe is not an error (a program that fails early does not read it). **stdout and stderr** are read by one thread each, so a full pipe never stops the child while the writer waits. stdout is capped at **4 MB** and one line at 1 MB; stderr keeps only its last 4 KB. Beyond the cap the child is killed and the outcome is `OutputTooLarge` (a failure of kind `cli_failed`, "produced too much output").
- **Time and cancelling.** The main loop polls the child every 20 ms against the deadline and the cancel token. On expiry or cancel: on Unix `SIGTERM` to the group followed by `SIGCONT` (a stopped group does not act on a pending `SIGTERM`), wait up to 2 s, then `SIGKILL` to the group; on Windows `TerminateJobObject`; a cancel by `CancelSource::cancel` has already sent the first of these before the loop sees it (9.4); then `wait()` always, then join the threads (waiting at most 1 s for a pipe a grandchild still holds). The outcomes are `Exited`, `TimedOut`, `Cancelled`, `OutputTooLarge`, `SpawnFailed`.
- **Redaction.** Anything that follows a token pattern in stderr or a message (`sk-…`, `sk-ant-…`, `AIza…`, `ya29.…`, `Bearer …`, a JWT (`eyJ…` with two dots), a run of 40 or more base64 characters (letters, digits, `-`, `_`, `+`, `=` and `/`; a run with a `/` in it must also have upper case, lower case and a digit, so that an ordinary path survives)) is replaced by `…` before it goes to a log or a message; messages are cut to 300 characters.
- **Panics.** The `RunOutput` is returned through the guard of the `ProcessWrap`: a panic in the advisor's own code after the spawn drops the child wrapper, which kills the tree.

**Tests** (section 15's fakes, on Linux and macOS; Windows is not run for now, the owner 2026-10-29): a slow program is killed together with a grandchild it started (the pid file shows both gone), `Cancelled` ends a running request within 3 s, a program that writes 2 MB before reading its stdin does not deadlock, output beyond the cap kills it, a program that exits without reading stdin is not an error, the environment of the child has the switches and lacks the parent-session variables, `CREATE_NO_WINDOW` is set on Windows (read back through the fake), and no `ym-auto-*` folder is left after success, failure, timeout and a panic; the sweep removes a `ym-auto-*` folder older than ten minutes and leaves a newer one and any other folder alone.

## 6. The picture and the answer's form

### 6.1 The picture

The worker gives the advisor the picture of `docs/19` section 3 (`AdvisorPicture`: 8-bit sRGB, at most 768 px on the long edge, rendered once on the CPU reference path, the same bytes for every provider and in the app and `ym`). The two programs take it as a JPEG made by `ym_agents::jpeg::encode(&AdvisorPicture) -> Vec<u8>`:

- `jpeg_encoder::Encoder::new(&mut out, 92)` (quality 92), `ColorType::Rgb`, `set_progressive(false)`, `set_optimized_huffman_tables(false)`, `set_sampling_factor(SamplingFactor::R_4_4_4)` (so the encoder writes one interleaved baseline scan, which every decoder reads), no density, no EXIF, no ICC profile, no comment: the picture carries no metadata and no file name. The same picture gives the same bytes.
- `IMAGE_ENCODING = "jpeg444q92@768"` names this in reports (section 14). Changing any of it changes `SCHEMA_VERSION` or the prompt version, because the encoding can change an answer.
- A 768 × 512 picture is 532 visual tokens for Claude (`⌈w/28⌉ × ⌈h/28⌉`); 768 px is never resized by either program. The JPEG of a photograph is roughly 150 to 300 KB (an estimate; the check of section 17 measures it).
- A test decodes the output with `zune-jpeg` (as `ym-export`'s tests do): the size, the first marker `FFD8`, a single scan, and every pixel within 6 of the source on a smooth gradient picture.

### 6.2 The answer's form

The form is `docs/19` section 4's: the same fields, the same cases, the same guides. The programs get it as a JSON Schema with every numeric step and colour as a **string enum with Apple's case names** (the model reads names, not numbers). Every field is optional but `scene` and `plan`. `ym_agents::schema` builds it:

| JSON field | Step in `AutoSteps` | Values, in order of their code (code) | `description` (the guide of `docs/19` section 4, verbatim) |
|---|---|---|---|
| `scene` | `AdvisorAnswer::scene` | a string | What the photo shows and how it is lit, in at most eight words. |
| `plan` | `AdvisorAnswer::plan` | a string | The look you will give this photo and how you will get it, in at most fifteen words. |
| `exposure` | `exposure` | `muchDarker` (−3), `darker` (−2), `slightlyDarker` (−1), `keep` (0), `slightlyBrighter` (1), `brighter` (2), `muchBrighter` (3) | Brightness of the main subject. |
| `highlights` | `highlights` | `muchDarker` (−2), `darker` (−1), `keep` (0), `brighter` (1), `muchBrighter` (2) | The brightest tones: darker to bring back sky and bright detail, brighter for an airy, luminous look. |
| `shadows` | `shadows` | `muchDarker`, `darker`, `keep`, `brighter`, `muchBrighter` (−2 … 2) | The dark tones: brighter to open up detail in the shadows, darker for depth and mood. |
| `blacks` | `blacks` | `muchDeeper`, `deeper`, `keep`, `lifted`, `muchLifted` (−2 … 2) | The black level: deeper for rich, solid blacks, lifted for a soft, faded, filmic black. |
| `contrast` | `contrast` | `muchLess`, `less`, `keep`, `more`, `muchMore` (−2 … 2) | Overall contrast: more for punch and depth, less for a soft, gentle look. |
| `warmth` | `white` | `muchCooler`, `cooler`, `keep`, `warmer`, `muchWarmer` (−2 … 2) | Colour of the light: warmer for golden, cosy or sunny light, cooler for blue hour, night or a cold mood; also to remove an unwanted cast. |
| `tint` | `tint` | `muchGreener`, `greener`, `keep`, `moreMagenta`, `muchMoreMagenta` (−2 … 2) | Green against magenta: more magenta to remove a green cast from fluorescent light or foliage, greener to remove a magenta cast. |
| `saturation` | `saturation` | `muchLess`, `less`, `keep`, `more`, `muchMore` (−2 … 2) | Strength of the colours: more for vivid, rich colour, less for a muted, restrained or moody look; keep skin natural. |
| `shadowColour` | `shadow_color` | `none` (0), `teal` (1), `blue` (2), `green` (3), `amber` (4), `magenta` (5) | A colour for the shadows, as colourists do: teal or blue for a cinematic or cool mood, amber for warmth, none for a neutral grade. |
| `shadowColourStrength` | `shadow_color_strength` | `subtle` (1), `medium` (2), `strong` (3) | How strongly the shadows are coloured. |
| `highlightColour` | `highlight_color` | `none` (0), `warm` (1), `gold` (2), `cool` (3), `pink` (4) | A colour for the highlights: warm or gold for sunlit, golden or nostalgic light (with teal shadows this gives the classic cinematic look), cool for a cold, clean feel, none for neutral. |
| `highlightColourStrength` | `highlight_color_strength` | `subtle` (1), `medium` (2), `strong` (3) | How strongly the highlights are coloured. |

The field names keep the British spelling of `docs/19` (the model reads them). **The schema text** is built as one compact JSON string, properties in the table's order (the order the model should think in, and the reason it is written by hand rather than from a `serde_json` map, whose keys come out sorted in this workspace, `tools/ci/check_serde_order.sh`):

```text
{"type":"object","title":"Answer","additionalProperties":false,"required":["scene","plan"],"properties":{"scene":{"type":"string","description":"What the photo shows …"},"plan":{"type":"string","description":"The look you will give …"},"exposure":{"type":"string","enum":["muchDarker","darker","slightlyDarker","keep","slightlyBrighter","brighter","muchBrighter"],"description":"Brightness of the main subject."}, … }}
```

It is ASCII only, has no double quote or backslash inside a description, and is about 3 KB. `SCHEMA_VERSION = 1`. A test (`schema.rs`) compares every enum's case names and every description with the Swift source `crates/ym-apple/swift/AppleAuto.swift` (read as text: the `case` lines of each enum and the `@Guide(description:)` strings) and with the table above, so the three cannot drift.

### 6.3 From the program's answer to `AdvisorAnswer`

1. Take the `structured_output` object (a JSON string is parsed once).
2. **Drop** every key whose value is `null` and every key that is not in the schema. (An absent field sets nothing, `docs/19` section 5; an extra note is harmless.)
3. **Validate** the rest against the schema with `jsonschema` (draft 2020-12, `default-features = false`; the validator is built once per advisor). An unknown enum name, a number where a name belongs, a missing `scene` or `plan`, or a wrong type fails the photo with `bad_answer` and the validator's first message (cut to 300 characters).
4. **Map** each name to its step by its position in the table's list (the number in brackets) and each colour by its code; `warmth` becomes the step `white`.
5. `AdvisorAnswer { steps, scene, plan, served_model }` goes through `AdvisorAnswer::checked` (ranges, strengths only with a colour, scene and plan cut to 80 and 160 characters, `bad_answer` when no value is set). `served_model` is the model id the program says answered (section 7.3).

## 7. Claude Code

### 7.1 The probes

`claude --version` prints `2.1.310 (Claude Code)`; `claude auth status` prints JSON by default (`--text` is for people), exit 0 signed in, exit 1 signed out:

```json
{"loggedIn":true,"authMethod":"oauth_token","apiProvider":"firstParty", …}
```

Only `loggedIn`, `authMethod` and `apiProvider` are read; the other fields (they include the person's folders) are ignored and never stored. Unknown fields are ignored everywhere.

### 7.2 The request

**Files** (in the request's temp folder, section 5): `in/instructions.txt`, the instructions in effect (`effective_instructions`), UTF-8 without a byte-order mark, line feeds as they are; `cwd/` empty.

**Command line** (the working folder is `cwd/`; arguments one by one; `<…>` is filled in):

```text
<claude> -p
  --input-format stream-json --output-format stream-json --verbose
  --safe-mode --tools "" --no-session-persistence --max-turns 3
  --model <model> --effort <effort>
  --system-prompt-file <tmp>/in/instructions.txt
  --json-schema <the schema of 6.2, one argument>
  --disable-slash-commands --strict-mcp-config
```

`<model>` is `sonnet` and `<effort>` is `low` unless `YELLOWMOON_CLAUDE_MODEL` or `YELLOWMOON_CLAUDE_EFFORT` is set (D18). A model is always passed: the program's own default is its largest model. `--system-prompt-file` replaces Claude Code's own system prompt (the program still adds its identity line first). Why each flag: `-p` no interface and no trust dialog; `--input-format stream-json` is the only way to send an image in print mode (and requires the `stream-json` output); `--safe-mode` switches every customisation off (hooks, plugins, skills, CLAUDE.md, MCP servers) while keeping the person's sign-in, model and provider settings; `--tools ""` leaves only the `StructuredOutput` tool; `--no-session-persistence` writes no transcript; `--max-turns 3` leaves room for the program's one reminder when the model answers in words; `--strict-mcp-config` and `--disable-slash-commands` are belt and braces.

**Standard input**, one line, then end of input:

```json
{"type":"user","message":{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":"<base64 of the JPEG>"}},{"type":"text","text":"Grade this photo."}]}}
```

(The image first, then the request line, as Apple's prompt does. The request line is fixed whatever the instructions say.) A malformed image block would not fail the program: it sends the model a text note instead, so the Rust side checks the picture before it is encoded (section 6.1) and never relies on an error to notice a bad one.

### 7.3 Reading the answer

Standard output is newline-delimited JSON events. Read all lines (blank and unparsable lines are skipped); keep the **last** object with `"type":"result"`; keep every `"type":"system","subtype":"api_retry"` event's `error`. Then:

| What was seen | Outcome |
|---|---|
| the process could not start | `cli_failed`: "Claude Code could not be started: <the system's message>" |
| the time limit (90 s) | `timeout` |
| cancelled (the session closed, `ym` was interrupted, new instructions) | the request is abandoned: no result is stored |
| no `result` line | `cli_failed` with the exit code and the last 300 characters of stderr (redacted); a run that we killed is the time-limit or the cancel row above, not this one |
| `is_error: true` and its `result` text starts with `Not logged in`, or an `api_retry` with `error: "authentication_failed"`, or `api_error_status` 401 or 403 | `not_signed_in` (the advisor's cached state becomes `signed_out` and the listener is called) |
| `is_error: true` and `api_error_status` 429, or an `api_retry` with `error: "rate_limit"` | `rate_limited` |
| `is_error: true`, anything else | `cli_failed` with the `result` text (cut to 300 characters) |
| `is_error: false`, `structured_output` present | validate and map (6.3); `bad_answer` when that fails |
| `is_error: false`, no `structured_output`, `stop_reason` is `refusal` | `refused` |
| `is_error: false`, no `structured_output` | `bad_answer`: "Claude Code answered in words instead of the answer form" |

The exit code alone decides nothing: a signed-out run has `subtype: "success"` and `is_error: true`. **The model that answered** is the key of `modelUsage` with the most `outputTokens` (the object has one key in practice); it is `served_model`, and the session writes it into the photo's `info.model`. Until a request has answered, `info.model` is the alias (`sonnet`). The program's `total_cost_usd` and token counts are not used or stored.

## 8. Antigravity (`agy`)

Everything here is from Google's documentation, the changelog and `agy --help`; **no request to `agy` has been run** (section 19 lists what a signed-in machine must check). The route is the second to ship and is dropped rather than worked around if its checks fail.

### 8.1 The probes

`agy --version` prints `1.4.7`. The sign-in check is `agy -p /usage --output-format json` in an empty new `cwd/` (the 1.1.11 changelog says the `/usage` command answers without starting an agent turn or spending quota). Exit 0 with output is signed in; a non-zero exit whose output or stderr contains `authentication required` is signed out; any other failure, or 15 s without an answer, is `probe_failed`. It runs only when asked for (3.3).

### 8.2 The request

**Files:** `cwd/photo.jpg` (the JPEG of 6.1) and `in/schema.json` (the schema of 6.2, the same text). The working folder holds only the photograph.

**Command line** (the working folder is `cwd/`):

```text
<agy> -p <the prompt below, one argument>
  --output-format json --json-schema <tmp>/in/schema.json
  --effort <effort> --print-timeout 110s
  --disable-slash-commands --sandbox
  [--model <slug>]
```

`<effort>` is `low` (or `YELLOWMOON_AGY_EFFORT`); `--model` is passed only when `YELLOWMOON_AGY_MODEL` is set (D18), so the person's default model answers. `--print-timeout 110s` is the program's own limit, set below our hard stop of 120 s so that it ends the turn itself, returns what it has and exits 0 (a timeout is read from the envelope, not the exit code). Never `--dangerously-skip-permissions`, never `--continue` (headless runs are stateless).

**The prompt** (one argument; the first line is fixed so that the person's own text never begins with `-` or `/`):

```text
You will grade one photograph. Follow these instructions.

<the instructions in effect>

The photograph is @photo.jpg in the current folder. Look only at that file. Do not open, read, write or run anything else, and use no tool but the one that shows you the photograph. Grade this photo. Answer only with the JSON object of the answer form below, as your final structured answer.

Answer form (JSON Schema):
<the schema of 6.2>
```

That the agent really receives the pixels through an `@photo.jpg` mention in `-p` mode is **unverified** (the owner states it works; section 19).

### 8.3 Reading the answer

Standard output is read whole and its **last** line that parses as a JSON object is the envelope (banner lines before it are skipped, and so is whatever follows it, such as "Done" or an update notice; an object printed over several lines is read from its `{` line on):

```json
{"conversation_id":"…","status":"SUCCESS","response":"…","num_turns":1,"structured_output":{…},"usage":{…}}
```

| What was seen | Outcome |
|---|---|
| the process could not start | `cli_failed` |
| our hard stop (120 s) | `timeout` |
| cancelled | abandoned |
| exit 2 | `cli_failed`: "Antigravity didn't accept Yellowmoon's options (version <v>). Update Yellowmoon." |
| exit 1 and stderr contains `authentication required` | `not_signed_in` (cached state `signed_out`, listener called) |
| exit 1, anything else | `cli_failed` with stderr's last 300 characters (redacted) |
| exit 3 with an `AGY_ERROR:` line on stderr (a JSON object after the colon) | `rate_limited` when the line names `RESOURCE_EXHAUSTED`, a rate limit (`RATE_LIMIT_EXCEEDED`, `rateLimitExceeded`, `ratelimit`), `Too Many Requests`, `quota` or the word `429`; `not_signed_in` when it names `UNAUTHENTICATED`, the words `401` or `403`, `authentication required`, `authentication failed`, `not signed in`, `not logged in` or `login required`; otherwise `cli_failed` (so "authentication service unavailable" is one). The line is matched as upper case whole words, with `_`, `-`, punctuation and camel-case boundaries as the spaces between them. Its field names are **unverified**, so it is searched as text, not by key |
| exit 0, no envelope | `cli_failed` |
| exit 0, `status` is `SUCCESS`, `structured_output` present | validate and map (6.3); `bad_answer` when that fails |
| exit 0, `status` is `SUCCESS`, no `structured_output` | `bad_answer`: "Antigravity answered in words instead of the answer form" |
| exit 0, `status` is `ERROR`, `CANCELED`, `INTERRUPTED`, `INVALID`, `WAITING` or `RUNNING` | `timeout` when our clock is past 105 s (the program's own limit fired), else `cli_failed` with the envelope's `error` text |

`denied_actions` (actions the agent wanted and was refused) are logged at debug level and otherwise ignored: the answer's form bounds what they could change (2.3). The envelope names no model, so `info.model` is the slug passed with `--model`, or `default`. The conversation the program keeps for each run is its own and is neither read nor deleted (where it lives, and whether a script may remove it, is unknown, section 19).

## 9. Failures, the breaker, concurrency and cancelling

### 9.1 Kinds and words

`FailureKind` (`docs/19` section 8) gets two kinds, `not_signed_in` and `cli_failed`; the others keep their wire names. The words (`ym_core::auto_provider::failure_words(provider, kind)`; `ym info`'s `Failed (<words>)`, the app's logs) differ by provider:

| Kind (wire) | Apple (as `docs/19`) | Claude Code | Antigravity |
|---|---|---|---|
| `unavailable` | Apple Intelligence stopped being available | Claude Code stopped being available (also: the breaker is open) | Antigravity stopped being available (also: the breaker is open) |
| `rate_limited` | macOS asked Yellowmoon to wait | Claude Code reported a rate limit | Antigravity reported a rate limit |
| `timeout` | no answer within 30 seconds | no answer within 90 seconds | no answer within 120 seconds |
| `refused` | the model refused | Claude refused | Antigravity refused |
| `bad_answer` | the answer could not be read | the answer could not be read | the answer could not be read |
| `not_signed_in` | (not used) | Claude Code isn't signed in | Antigravity isn't signed in |
| `cli_failed` | (not used) | Claude Code reported an error | Antigravity reported an error |
| `other` | an error | an error | an error |
| the other Apple kinds | `docs/19` | (not used) | (not used) |

`cli_failed` and `other` add `: <message>` as `AdvisorError`'s display does. A failure of any kind uses the rule for that photo and says why (`ym info`, the Automatic card, `ym:AutoAdvisor`); Re-run asks again.

### 9.2 Retries and the breaker

- **No retries** for Claude Code and Antigravity. `FailureKind::retried` takes the provider (`retried(self, provider)`): it is true only for Apple and only for `rate_limited` and `concurrent_requests`, as `docs/19` section 6 says (twice, 2 s apart); for the two programs it is false for every kind. Each program already retries transient API errors itself (Claude Code reports them as `api_retry` events), and a plan's limit lasts longer than any pause.
- **The breaker.** Each advisor counts consecutive failures of the kinds `rate_limited`, `timeout` and `cli_failed` (a success, or a `bad_answer`, `refused` or `not_signed_in`, resets the count; `not_signed_in` has its own rule below). At **3** the breaker opens: for **60 s** `ask` returns `unavailable` ("stopped after three failures in a row: <the last failure in words>") at once, without starting a process. After 60 s one request goes through as a trial: success closes the breaker, failure opens it for another 60 s. The photos it refused are failed with kind `unavailable` and use the rule; answers already stored stay. The breaker is the advisor's own state, in a mutex (two requests run at once), and `settle` and `refresh` do not clear it.
- **`ym` says so once**: when an advisor's breaker opens during a run it prints "Claude Code failed three times in a row (<the last failure in words>). Photos it doesn't answer use Default Auto." on stderr (a warning; `--quiet` hides it). The advisor gives the core the sentence through `AutoAdvisor::details().breaker` (`Some` while the breaker is open), which `ym` reads after the run's waits.
- **`not_signed_in`.** The advisor sets its cached state to `signed_out` and calls the listener. In the core, a request that ends in `not_signed_in` reads the availability again; when the provider is not available now, the request is abandoned (the photo stays pending, as before a call) instead of stored as a failure, and the whole session reads as unavailable until the provider is back. The next photos' requests then find it unavailable and wait. The cached `signed_out` is **held** (3.3): an automatic `refresh()` does not find the program "signed in again" while the service still refuses it; a request that succeeds, 10 minutes, or the person's own check (`refresh_asked`: Preferences opening, `ym doctor`) ends the hold.

### 9.3 Concurrency

`AutoAdvisor::concurrency()` (Apple 1, Claude Code 2, Antigravity 1). The session starts as many workers as the largest of its advisors (`ProviderAdvisors::max_concurrency`) and lets at most the **chosen** advisor's number run at once. Each worker takes the next photo in the order of `docs/19` section 6 that no worker has in flight. Two requests of one advisor share its cache and breaker (mutexes). `ym render` therefore asks two photos at a time with Claude Code.

### 9.4 Cancelling

`AutoAdvisor::cancel()` kills the advisor's running children (section 5) and makes the requests that were running return without a result. The core calls it when the session closes or drops, when the instructions change (their answers would be dropped anyway), and from `Core::cancel_auto()`, which the app calls on quit and `ym` calls from its Ctrl-C handler before it exits (a child is not stopped by the parent's death on Unix, and not on Windows either: *As built*, 15.7, says why the job object does not do it). It is not called when the provider changes: an answer to the old provider's request is kept for that provider (section 10.5). A call that returns after the session closed has its answer dropped (`docs/19` section 6).

**Quitting and Ctrl-C wait for the programs (the core and `ym`).** `cancel()` only tells the programs to stop; the request's own thread then kills what ignores the polite stop (after 2 s, `TERM_GRACE` in `ym-agents`), reaps it and removes its temporary folder. Both waits below (`AUTO_CANCEL_WAIT` in `ym-core`, `INTERRUPT_WAIT` in `ym`) are **4 s**: that grace plus 2 s, so a program that ignores `SIGTERM` (gone at about 2.05 s) still has 2 s of margin on a loaded machine. `ym-core` does not depend on `ym-agents`, so each crate writes the number with a comment naming the grace. A program that stops on `SIGTERM` is not waited for any longer than it takes to be reaped (`wait_idle` returns as soon as nothing runs). A process that ends right after `cancel()` would leave all of that undone, so the end of the run goes through three calls on the registry (`ProviderAdvisors`):

- `halt()`: sets a flag that every clone of the registry shares, then cancels every advisor. From then on the session asks nothing more (a worker takes no photo, a request about to start does not), starts no probe, and a request that ends because of it **stores nothing**, not even "the request was cancelled" as the photo's failure (`docs/01`): the photos it was asked about stay unasked, and a halted session does not count them as work (`wait_idle` returns). The flag cannot be cleared: the registry belongs to the run that is ending. A cancel for new instructions or a closed session (which are not the end of the run) is dropped on arrival as before.
- `wait_idle(timeout)`: waits, with one deadline shared by the advisors, until each advisor has no request or probe running, that is every program was reaped and every temporary folder removed (`AutoAdvisor::wait_idle`).
- `Session::cancel_auto()` (and so `Core::cancel_auto()` and `Core::shutdown()`, which the app calls on quit) halts the registry and then waits for it, **up to 4 s** (`AUTO_CANCEL_WAIT`), and logs a warning when the bound passes; the process may end as soon as it returns. `ym`'s Ctrl-C handler does the same on its own thread (it has no session), **up to 4 s** (`INTERRUPT_WAIT`), before it sets the flag the command polls and before **every** `exit(130)` (the first signal of a command that does not poll, and a second signal). It is registered (`register_advisors`) before the start-up probe, and `ym doctor` registers before its probes, so a Ctrl-C while they run stops them too. The session's drop cancels and waits up to 4 s as well (the person is not waiting there: `ym` ends the process right after its core is dropped); `Session::close()` and a change of the instructions cancel and do not wait (the person is at the keyboard).

**What `cancel()` does in `ym-agents`** (the advisors' own side of the above). Each advisor keeps one cancel source, which lists the children of every request and probe that took a token from it, from their start to their reaping. `cancel()` moves the source's epoch and then, **before it returns**, sends each listed child `SIGTERM` and `SIGCONT` to its process group (Unix: the `SIGCONT` so that a stopped group acts on the `SIGTERM`) or terminates its job (Windows), whatever the threads of the runs are doing: a caller that exits at once has told every program to go. A run that has not started its child when the epoch moves starts none. The rest is the runs' own (section 5): within 20 ms each sees the cancel, sends `SIGKILL` to a group that is still there 2 s after the `SIGTERM`, reaps it, and drops its temporary folder. A run that ends because its child died of the `SIGTERM` is a cancelled run, not a failed one. **Probes are cancelled too** (their `--version`, `claude auth status`, `agy -p /usage` and login-shell runs take their token from the same source), so a hanging check does not outlive the app. `AutoAdvisor::wait_idle(timeout)` returns `true` once no request and no probe of the advisor is in flight (every child reaped, every folder removed), or `false` at the timeout; after `cancel()` it needs up to 2 s more than the polite stop when a program ignores `SIGTERM`.

A request is panic-safe as a whole (`docs/19` section 6); the child is killed by its wrapper when the advisor's code panics.

## 10. The preference, the consent and which provider acts

### 10.1 What `preferences.json` holds

| Key | Type | Default | Notes |
|---|---|---|---|
| `auto_provider` | `"yellowmoon"`, `"apple"`, `"claude"` or `"agy"` | `"apple"` | The saved choice. A provider that is not available (or not consented) at run time acts as `yellowmoon`, and the choice stays saved. |
| `auto_consent` | `{ "claude": bool, "agy": bool }` | both `false` | Whether the person agreed to send photos through that program (10.2). A missing key is `false`; keys of other names are dropped; a non-boolean is `false` with a warning. |
| `apple_auto_instructions` | string, at most 2000 characters | `""` | **Unchanged** (D1): the person's own instructions, `""` for the default; one text for every provider. |
| `apple_auto` | boolean | (read only) | Read once for the migration below; **never written again** (D15). |

`docs/01` §Persistence lists these. The file now has ten fields: `default_look`, `display_rendering`, `export_defaults`, `remove_location_from_exports`, `check_for_updates_automatically`, `watch_downloads_during_import`, `last_used_folders`, `auto_provider`, `auto_consent` and `apple_auto_instructions`. A version before this one reads a file without `apple_auto` as on, so a downgrade turns Apple Auto on for a person who had chosen otherwise; that is accepted (no mirror is written, D15).

**Migration** (when the file is read, `ym_core::preferences`):

| The file has | `auto_provider` is |
|---|---|
| a valid `auto_provider` | that value (`apple_auto` is ignored) |
| an `auto_provider` that is not one of the four names | read as absent, with the warning "Preference \"auto_provider\" can't be used, so the default is used." |
| no valid `auto_provider`, `apple_auto: false` | `yellowmoon` |
| no valid `auto_provider`, `apple_auto: true` or absent | `apple` |

The next save writes `auto_provider` and `auto_consent` and drops `apple_auto`. The `apple_auto_instructions` rules (empty or only white space is `""`; the default text is saved as `""`; longer than 2000 characters gives the default with a warning) are unchanged, and the refusal message loses its product name: "The instructions are N characters long. At most 2000 fit, because the model must also see the photo." (`check_instructions`).

### 10.2 Consent

- **When it is asked.** In the app, when the person chooses Claude Code or Antigravity in the dropdown and its `auto_consent` entry is not `true` (12.3). It is asked once per provider: it stays saved until the file is edited; this version has no control that withdraws it (the person can choose another provider at any time).
- **How it is saved.** The webview sends one patch with both fields, `{ "auto_provider": "claude", "auto_consent": { "claude": true, "agy": false } }`, so there is never a saved cloud choice without its consent. The patch replaces the whole `auto_consent` it names (as every field of a patch does). The core **refuses** a patch that makes `auto_provider` Claude or Agy when `auto_consent` for it is not `true` after the patch is applied: `PreferencesError::Invalid("Say yes to sending photos to Claude Code first.")` (or "…Antigravity…"). Choosing `yellowmoon` or `apple` is always allowed, whether or not it is available.
- **`ym`.** Naming a provider with `--auto-provider claude|agy` is the consent for that run and is never saved (D16). Without the option `ym` follows the saved choice, and a saved cloud choice acts only when its consent is saved; otherwise it acts as `yellowmoon`, silently as the app does (`ym doctor` says why).
- **A file** with `auto_provider` Claude or Agy and no consent is kept as it is: it acts as `yellowmoon` (reason `not_consented`) with no warning, and the dropdown asks the question when the person chooses it.

### 10.3 The choice for a run

`AutoChoice` is `FromPreference` (the app, and `ym` without an option) or `Provider(p)` (`--auto-provider p`, `--apple-auto on` = `Provider(Apple)`, `--apple-auto off` = `Provider(Yellowmoon)`). It replaces `AppleAutoMode` (`On`, `Off`) everywhere. The provider **chosen** for the run is the preference for `FromPreference` and `p` for `Provider(p)`; an explicit `Provider(p)` is consent for a cloud `p`.

### 10.4 Which provider acts (`Activity::decide`)

In this order; the first rule that applies decides (`docs/19` section 6 had four; this replaces them):

1. The core was given no advisors (the golden tool, most tests, `doctor --bench`): **Off**.
2. The chosen provider is `yellowmoon`: **Off**.
3. No advisor is registered for the chosen provider, or its availability has not been read: **Unavailable** with reason `not_installed`.
4. Its availability is `Unavailable(reason)`: **Unavailable** with that reason (also `checking`). The choice is kept and ignored, as `docs/19` section 2 says for Apple. **One exception, which is rule 5 applied early:** a cloud provider that the run did not name and whose consent is not saved is never probed (3.3, 13.1), so it still reads `checking` and no answer is coming; that reads `not_consented`, and a photo says so every time instead of `checking` or `not_consented` by the luck of the timing. Every other reason keeps its words.
5. The provider is cloud, the choice is `FromPreference` and its `auto_consent` is not `true`: **Unavailable** with reason `not_consented`.
6. Otherwise **Active** with the advisor's `AdvisorInfo`.

`Off`, `Unavailable { reason }`, `Pending`, `Answered` and `Failed` are the states of a photo (`AppleAuto`) as before; `Unavailable` and `Off` both mean the rule's Auto values, `Unavailable` is what the Automatic card and `ym info` explain. **The effective provider** (`AutoStatus::effective`) is the chosen one when the activity is `Active` and `yellowmoon` otherwise.

### 10.5 What the session keeps

- **Answers are kept per photo and per provider**, in memory, for the session (`Photo::apple_results: [Option<AppleAuto>; 3]`, indexed by `AutoProvider::slot`). The state a photo shows is the **chosen** provider's result, or `Pending` while the provider acts and has none. Switching provider asks the photos that have no answer from the new provider (the photo on screen first) and leaves the other providers' answers in place; **switching back reuses them**, and nothing is asked again.
- **New instructions drop every provider's answers** (they are shared; `docs/19` section 6), advisors' running requests are cancelled (9.4), and each request carries the instructions' generation as before.
- **Re-run** drops the chosen provider's result for that photo and asks again. Removing photos, Undo, bursts and the order of requests are as `docs/19` section 6, for the chosen provider's results.
- **A request in flight when the provider changes** finishes, and its answer is stored under the provider that was asked (it is valid for that provider), not shown unless the person switches back. The picture of a photo is dropped after its answer as before; a provider asked later has it rendered again (`docs/19` section 3, the worker's own render).
- **Nothing is written to disk** (hard rule 4). The key of an answer is `(photo, provider)` plus the session's instructions generation; the model id and the program's version are recorded with it but a change of either mid-session (an update of the program) is not noticed: Re-run asks again.

## 11. The core

New module `crates/ym-core/src/auto_provider.rs` (decision D3), exported as `ym_core::auto_provider`. Changes to `crates/ym-core/src/apple_auto.rs` are listed after it. (`pub` items carry doc comments; JSON is snake_case.)

```text
// ---- ym_core::auto_provider (new) ----
enum AutoProvider { Yellowmoon, Apple, Claude, Agy }       // wire "yellowmoon" | "apple" | "claude" | "agy"
  fn name(self) -> &'static str                            // "Default Auto", "Apple Intelligence", "Claude Code", "Antigravity"
  fn is_cloud(self) -> bool                                // Claude, Agy
  fn timeout_s(self) -> u32                                // 0, 30, 90, 120
  fn concurrency(self) -> usize                            // 1, 1, 2, 1
  fn prompt_version(self) -> u32                           // 0, 3 (PROMPT_VERSION), 1, 1
  fn slot(self) -> Option<usize>                           // None, Some(0), Some(1), Some(2)
  const ALL: [AutoProvider; 4]                             // the order of the dropdown
enum AutoChoice { FromPreference, Provider(AutoProvider) }  // replaces AppleAutoMode
struct AutoConsent { claude: bool, agy: bool }             // serde(default); fn get(&self, AutoProvider) -> bool (true for yellowmoon, apple)
struct ToolInfo { version: String, api: Option<String>, auth: Option<String> }   // CLI providers
enum ProviderState { Available, NotInstalled, TooOld, SignedOut, Unavailable, Checking }
  fn of(&Availability) -> ProviderState
struct AdvisorDetails { path: Option<PathBuf>, tool_version: Option<String>, api: Option<String>,
                        auth: Option<String>, breaker: Option<String> }              // Default = all None
struct ProviderStatus {
    provider: AutoProvider, state: ProviderState, reason: Option<UnavailableReason>,
    info: Option<AdvisorInfo>,           // when available, with the instructions in use
    tool_version: Option<String>,        // even when too old or signed out
    path: Option<PathBuf>,               // not on the app's wire (`ym doctor` only)
    consent: bool, experimental: bool,   // consent: AutoConsent::get; experimental: Agy
    listed: bool, selectable: bool,      // 3.1
}
struct AutoStatus {
    providers: Vec<ProviderStatus>,      // AutoProvider::ALL order
    preference: AutoProvider, choice: AutoChoice, chosen: AutoProvider, effective: AutoProvider,
    active: bool, visible: bool,         // effective != yellowmoon; the dropdown and the instructions show
    consent: AutoConsent, table: u32,
    #[flatten] instructions: InstructionsId, own_instructions: String,
    default_instructions: String, instructions_max_chars: usize,
}
struct ProviderAdvisors { .. }           // Clone + Default; three optional slots
  fn with(self, AutoProvider, Arc<dyn AutoAdvisor>) -> Self     // Yellowmoon is refused
  fn get(&self, AutoProvider) -> Option<&Arc<dyn AutoAdvisor>>
  fn is_empty(&self) -> bool
  fn max_concurrency(&self) -> usize     // 1 when empty
  fn refresh(&self, only: Option<AutoProvider>)                 // non-blocking
  fn settle(&self, timeout: Duration)                           // every advisor, one deadline: `ym doctor`
  fn settle_provider(&self, AutoProvider, timeout: Duration)    // the chosen one alone: `ym`, the export
  fn cancel_all(&self)                                          // tells every program to stop, returns at once
  fn wait_idle(&self, timeout: Duration) -> bool                // every program reaped, every folder removed
  fn halt(&self)                                                // the run is ending: cancel_all + ask and probe nothing more (9.4)
  fn is_halted(&self) -> bool
  fn on_change(&self, listener: ChangeListener)                 // registers on every advisor
type ChangeListener = Arc<dyn Fn() + Send + Sync>;
consts: CLAUDE_PROMPT_VERSION = 1, AGY_PROMPT_VERSION = 1, SCHEMA_VERSION = 1, IMAGE_ENCODING = "jpeg444q92@768",
        CLAUDE_MIN_VERSION = "2.1.169", AGY_MIN_VERSION = "1.2.14", CLAUDE_DEFAULT_MODEL = "sonnet",
        CLI_EFFORT = "low", CLAUDE_TIMEOUT_S = 90, AGY_TIMEOUT_S = 120, BREAKER_FAILURES = 3, BREAKER_COOLDOWN_S = 60,
        PROBE_MIN_INTERVAL_S = 5
fn unavailable_hint(AutoProvider, UnavailableReason, tool_version: Option<&str>, os: &str) -> String   // 3.4, the hint column
fn unavailable_words(AutoProvider, UnavailableReason, tool_version: Option<&str>) -> String            // 3.4, the clause column
fn failure_words(AutoProvider, FailureKind) -> &'static str                                           // 9.1
fn cloud_notice(AutoProvider, api: Option<&str>) -> Option<String>                                    // 13.2

// ---- ym_core::apple_auto (changed) ----
UnavailableReason += NotInstalled, TooOld, SignedOut, ProbeFailed, UnsafeLauncher, Checking, NotConsented
                                  // wire "not_installed", "too_old", "signed_out", "probe_failed", "unsafe_launcher", "checking", "not_consented"
FailureKind       += NotSignedIn, CliFailed                                // "not_signed_in", "cli_failed"
  fn retried(self, provider: AutoProvider) -> bool                         // 9.2
AdvisorInfo { provider: AutoProvider,                                      // new; AdvisorInfo::new(model, os) is Apple's
              model: String, os: String /* "" for the programs */, tool: Option<ToolInfo>,   // tool: new
              prompt: u32,                                                  // new (Apple: PROMPT_VERSION)
              #[flatten] instructions: InstructionsId }
  fn cli(provider, model, tool: ToolInfo) -> AdvisorInfo                    // new constructor; prompt = provider.prompt_version()
  fn version_string(&self) -> String                                       // 14.3
AdvisorAnswer { steps, scene, plan, served_model: Option<String> }         // served_model new (serde default, skipped when None)
trait AutoAdvisor: Send + Sync {
    fn availability(&self) -> Availability;                                // never blocks on a process
    fn ask(&self, picture: &AdvisorPicture, instructions: &str) -> Result<AdvisorAnswer, AdvisorError>;
    // new, all with defaults, so ym-apple and FakeAdvisor are unchanged:
    fn concurrency(&self) -> usize { 1 }
    fn refresh(&self) {}                                                   // 3.3
    fn settle(&self, _timeout: Duration) {}
    fn cancel(&self) {}                                                    // 9.4: tells the programs to stop, returns at once
    fn wait_idle(&self, _timeout: Duration) -> bool { true }               // 9.4: every program reaped, every folder removed
    fn on_change(&self, _listener: ChangeListener) {}                      // 3.3
    fn details(&self) -> AdvisorDetails { AdvisorDetails::default() }
}
Activity::decide(has_advisors: bool, chosen: AutoProvider, explicit: bool, consent: &AutoConsent,
                 availability: Option<&Availability>) -> Activity          // 10.4
AppleAuto::advisor_json(&self) -> Option<String>                           // replaces xmp_json; 14.2
removed: AppleAutoMode (use AutoChoice), AppleAutoStatus (use AutoStatus)
unchanged: AdvisorPicture, AutoSteps, TABLE, DEFAULT_INSTRUCTIONS, INSTRUCTIONS_MAX_CHARS, PICTURE_LONG_EDGE_PX,
           REQUEST_TIMEOUT_MS (Apple's), RETRIES, RETRY_PAUSE_MS, share_burst_steps, FakeAdvisor
```

`FakeAdvisor` stays Apple-shaped (model `Test advisor`, os `test`). `ym_core::auto_provider::FakeProvider` (new, for tests of every crate) wraps it: `FakeProvider::new(AutoProvider, Availability)`; `ask` gives `FakeAdvisor`'s steps; its info is `AdvisorInfo::cli(provider, "Test <name>", ToolInfo { version: "0.0.0-test", api: None, auth: None })` for Claude and Agy; it can be told a concurrency, a delay and a failure to script the session's tests.

### 11.1 Preferences

`Preferences { …, auto_provider: AutoProvider, auto_consent: AutoConsent, apple_auto_instructions: String }` (`apple_auto: bool` is gone). `PreferencesPatch { auto_provider: Option<AutoProvider>, auto_consent: Option<AutoConsent>, apple_auto_instructions: Option<String>, … }` (`deny_unknown_fields` stays, so a patch with `apple_auto` is an error: the webview and `ym` never send it). `PreferencesPatch::check` adds the consent rule of 10.2.

### 11.2 The session (`crates/ym-core/src/session.rs`)

- `AppleAutoEnv { advisors: ProviderAdvisors, pictures, preference: AutoProvider, consent: AutoConsent, choice: AutoChoice, instructions: String, retry_pause }` (`AppleAutoEnv::new(advisors, pictures)`: preference `Apple`, no consent, `FromPreference`).
- `AppleState` holds the chosen provider (from `choice` and `preference`), the preference, the consent, the last availability **of the chosen advisor**, `in_flight: Vec<PhotoId>` (was one) and the instructions and their generation. `activity()` is `Activity::decide(…)`.
- `Photo::apple_result` becomes `apple_results: [Option<AppleAuto>; 3]` with `Photo::apple_result(provider)`; every use goes through the chosen provider (10.5).
- **Workers.** `Session::new` starts `advisors.max_concurrency()` threads (`ym-auto-0`, …). `next_apple_task` returns `None` while `in_flight.len() >= chosen advisor's concurrency()` and never returns a photo that is in flight. `apple_asking` becomes a counter; `Drop` calls `cancel_all()`, then joins the workers that are not inside a request, as before.
- **Requests.** `apple_request` takes the chosen advisor and provider from the task (the task remembers them, so a switch mid-request stores under the right provider); the retry loop applies `FailureKind::retried(provider)`; an `Answered` takes `info.model` from `answer.served_model` when it is `Some`; a `not_signed_in` failure reads the availability again and abandons the request when it is unavailable now (9.2).
- **Methods** (old name → new): `set_apple_auto_preference(bool)` → `set_auto_preference(AutoProvider)`; `set_apple_auto_mode(AppleAutoMode)` → `set_auto_choice(AutoChoice)`; new `set_auto_consent(AutoConsent)`; `apple_auto_status()` → `auto_status() -> AutoStatus`; `refresh_apple_auto()` is the read of a session opening (3.3): it reads the chosen advisor's availability, asks it to check only when that is not `available` (and the provider could act), and returns it; new `refresh_auto_for_export()`, the same read for an export, which asks for a check only when the chosen advisor still reads `checking`; the setters (`set_auto_preference`, `set_auto_choice`, `set_auto_consent`) read the new chosen advisor's availability and start **no check**; new `cancel_auto()`; `apple_auto_active()`, `apple_auto_asks()`, `set_apple_auto_instructions`, `apple_auto_instructions`, `prioritise_apple_auto` keep their names.
- **`auto_status()`** refreshes **all** advisors (3.3), then builds the `AutoStatus` from the cached availabilities: no process runs on the caller's thread.
- **The listener.** `Core::open` registers one with `ProviderAdvisors::on_change`: it calls `Session::read_auto_availability()` (no probe, 3.3) and then the callbacks given to `Core::on_auto_changed` (the app's emits `auto_status_changed`). A listener runs on the advisor's probe thread, so it must not block. `Core::open` registers the listener **first**, then builds the session, then reads what the advisors say (`Session::read_auto_availability()`, no probe: nothing runs at launch, 3.3), so a probe that finished in between is not missed.

### 11.3 `Core` (`crates/ym-core/src/app.rs`)

`CoreOptions { advisors: ProviderAdvisors (default empty), auto_choice: AutoChoice (default FromPreference), apple_auto_instructions: Option<String> (kept), … }` replaces `advisor` and `apple_auto`. `Core::auto_status()`, `Core::set_auto_choice(choice)`, `Core::on_auto_changed(callback)`, `Core::cancel_auto()`, `Core::settle_auto(timeout)` (`ym` only: `ProviderAdvisors::settle`). `Core::preferences_set` brings the session in step with the three preference fields: `set_auto_preference`, `set_auto_consent`, and (unless the run overrides them) `set_apple_auto_instructions`. `Core::export_start` and `ym render` read the chosen provider's availability again before queuing (`Session::refresh_auto_for_export`: the program is asked to check only when it was never checked, 3.3).

### 11.4 As built (lane B: the core, `ym`, the reports and the A/B kit, 2026-10-26)

What was built follows sections 10, 11, 13, 14 and 17. Where the text was silent, or the code needed something it does not name, this is what was done:

- **Step 0.** `ym_core::auto_provider` and the changes of section 11 to `apple_auto.rs` and `preferences.rs` came first, with the old names removed (`AppleAutoMode`, `AppleAutoStatus`, `CoreOptions::advisor`, `Preferences::apple_auto`; an `AdvisorAnswer` literal needs `served_model`). The edits outside the lane, each the smallest possible: `crates/ym-apple/src/abi.rs` builds its `AdvisorAnswer` with `served_model: None`, and the breaker's sentence in `crates/ym-agents/src/advisor.rs` and `tests/breaker.rs` says "Default Auto." (the rename below).
- **The consent rule is `PreferencesPatch::check_against(&Preferences)`.** Section 11.1 says `PreferencesPatch::check` adds the rule; the rule needs the consent as it is **after** the patch, so `check` keeps its field checks and `check_against(current)` adds the consent rule; `PreferencesStore::update` calls it inside its `modify`, against the file's current content (a patch that names only `auto_provider` is judged against the saved consent). The message is "Say yes to sending photos to Claude Code first." (the provider's `name()`). `auto_consent` in a file is read forgivingly (10.1): a missing key is `false`, other names are dropped, a non-boolean is `false` with a warning.
- **Requests in flight are counted per provider** (`in_flight: Vec<(PhotoId, AutoProvider)>`, not `Vec<PhotoId>`): with one list, a request still running for the provider the person had just left (Claude Code takes up to 90 s) held back every worker of the new one. `next_apple_task` compares the chosen provider's own count with its `concurrency()` and never returns a photo that is in flight for the chosen provider; a photo can be in flight for two providers at once.
- **`Session::auto_status_cached()`** builds the `AutoStatus` from the cached availabilities without asking any advisor to refresh; `auto_status()` refreshes every advisor and then calls it. `ym` uses the cached form once its run is open (its notices, the info row, the JSON), so a run never starts a program it did not choose.
- **The words.** `unavailable_hint(Apple, …)` is `UnavailableReason::words()` (the Apple row has no separate hint), and `unavailable_hint` is empty for `not_installed`, `checking` and `not_consented`, as 3.4 says. `cloud_notice` says "Anthropic" only when `api` is exactly `firstParty`; an unknown or missing `api` reads "the service your Claude Code is set up to use". `FailureKind::retried(provider)` is true only for Apple's `rate_limited` and `concurrent_requests`.
- **Default Auto** (owner, 2026-10-26: "Instead of 'Yellowmoon's own Auto' just say 'Default Auto'"). `AutoProvider::name()` is "Default Auto"; `ym doctor`, `ym info`, the stderr sentences, the report text, `ym render --help`, the README and the docs say Default Auto. The JSON and the wire value stay `yellowmoon`. The sentences of `ym` (Apple's of `docs/19` section 8 included) and the card's lines (12.5) end "Default Auto is used." (owner, 2026-10-29: "yea"; they said "Auto uses Yellowmoon's own settings." until then).
- **The legacy `apple_auto` key** is written from the photo's state under the **chosen** provider: the old Apple-only shape when that provider is Apple Intelligence, `{ "state": "off" }` for every other (so with Claude Code chosen it says `off` even if Apple Intelligence answered the photo earlier in the session; the session keeps that answer for a switch back). `info --json` has it for every developed photo; `render --json` writes it only for a photo **Apple Intelligence** acted on (answered, failed or pending), so for a photo Claude Code graded the key is absent there and `auto_advisor` carries the answer (`auto_advisor` is the chosen provider's, written for every photo of `info` and for every photo a provider acted on in `render`).
- **`ym doctor`.** The Apple row is `Apple Intelligence`; its old suffix " · turned off in Preferences" is gone, because the preference is no longer a boolean (the **Auto grading** row says what is saved and whether the consent is). `auto.providers.yellowmoon` is `{ "state": "available", "reason": null }`. For a program, `model` is the alias or slug asked for (`YELLOWMOON_CLAUDE_MODEL`, `YELLOWMOON_AGY_MODEL` when set, else `sonnet` and `default`) until a request has answered.
- **`ym info`'s row** for a saved program whose consent is not saved and that the run did not name: such a run does not probe the program (13.1), so the program still reads `checking`, which `Activity::decide` reads as `not_consented` (10.4); the row says `Not available (you haven't agreed to send photos to Claude Code yet)` (`AutoNote` keeps a mapping for the same case, which the core now makes itself).
- **Ctrl-C.** The advisors of a run are registered in a static (`RUNNING_ADVISORS`), so the interrupt handler can stop them even while the main thread waits in the core: it halts the registry and waits for the programs (9.4: up to 4 s) before it sets the flag the command polls and before any `exit(130)`.
- **The review of 2026-10-29 (the core and `ym`).** (1) `Core::cancel_auto`, `Core::shutdown`, the session's drop and `ym`'s Ctrl-C handler wait for the programs to be gone (9.4: `ProviderAdvisors::halt` and `wait_idle`, bounded), and `ym` registers its advisors before the start-up probe. (2) A saved cloud choice with no consent saved, not named by the run, is not probed by `Session::refresh_apple_auto` or by a change of a switch (`AppleState::chosen_may_act`): `info --json` says `not_consented` every time and no program of the double is started (3.3, 10.4). (3) An advisor's change notification is read with `Session::read_auto_availability`, which calls no `refresh()`. (4) `Core::settle_auto` and `ym`'s start-up settle wait for the chosen advisor only (`ProviderAdvisors::settle_provider`). (5) A cancelled request stores nothing (9.4). (6) `Core::auto_status_cached()` is the status without a probe (12.6). **Owner's decision of 2026-10-29 (O7): start the programs only when they are needed.** (7) `Core::open` and the setters of the choice, the consent and the preference start no probe (`Session::change_choice` only reads); `Session::refresh_apple_auto` (a session opening) asks the chosen advisor to check when it could act and its reading is not `available`; `Session::refresh_auto_for_export` (`Core::export_start`, `ym`'s `auto_notice_before_export`) only when it still reads `checking`; Preferences opening still asks every advisor (3.3). The app builds its advisors with `StartProbe::Never` (`LAUNCH_PROBE` in `src-tauri/src/api.rs`).
- **`--auto-instructions`** is the option, with `--apple-auto-instructions` as a visible clap alias; giving both is a usage error. `--auto-provider` takes `yellowmoon|apple|claude|agy` in any case (`--help` shows the four names; the possible values are hidden so the options stay two lines each); `--apple-auto on|off` and `--auto-provider` together are "Use --auto-provider or --apple-auto, not both." (exit 2).
- **The A/B kit** (`tools/eval/ab`, section 17). `ab.sh prepare --provider apple|claude|agy` renders with `--auto-provider <p>` and `--auto-provider yellowmoon`, passes the instructions with `--auto-instructions`, reads `auto_advisor` (or `apple_auto` of an older `ym`), and stops unless `ym doctor --json` says `auto.providers.<p>.state` is `available` ("<Name> is not available on this computer (<clause>). The test needs it signed in."; Apple keeps its Mac sentence). `--jobs` defaults to 1 for the programs. The plan's `provenance.apple_auto` keeps its name for the plans already made and gains `provider`, `tool_version`, `schema` and `image` for the programs; `answers` (mean seconds per photo of the whole `ym` call, the picture included, and the model ids that answered with their counts) is written next to the plan's other counts, outside its id, so the same plan keeps one id. `--resume` asks again for an answer of another provider. The summary and `score` name the provider, the program's version and the models; the page and the trials are unchanged. `YELLOWMOON_CLAUDE_MODEL`, `YELLOWMOON_CLAUDE_EFFORT`, `YELLOWMOON_AGY_MODEL` and `YELLOWMOON_AGY_EFFORT` pass through to `ym` with `YELLOWMOON_TEST_AUTO_PROVIDERS`. `tools/oracle/compare_metadata.py` asks `ym info --auto-provider yellowmoon` (it never read `ym:AppleAuto`).
- **What this lane did not do.** `src-tauri` is lane C's: it needs the removed names above replaced and `src-tauri/tests/app_cli_parity.rs` needs `YELLOWMOON_TEST_AUTO_PROVIDERS=none` (lane C did it). The macOS runs of (Windows is not run for now, the owner 2026-10-29) `ym-cli`'s `auto_providers` tests are the joint pass of 18.6. Everything section 19 lists needs a signed-in computer.

## 12. The app

### 12.1 The Preferences row

Placed after Display rendering, where the Apple row was. Built from glasscn's `Select`, `Badge` and `Button` and the existing `Row` (`docs/11` §Preferences).

| Item | Value |
|---|---|
| Shown | while `auto.visible` (3.1), **or while the saved choice is Claude Code or Antigravity** (a program the person agreed to must stay visible so that it can be withdrawn, even when it was uninstalled and nothing else is listed; 10.2); hidden otherwise, the browser build by default included, and with a saved Apple Intelligence (the default), which sends nothing anywhere; the instructions go with it |
| Title | **Auto grading** (the `Select`'s accessible name too) |
| Control | a `Select`, 220 px wide, whose value is `auto.effective`, **except** while the saved choice is a program that is listed but cannot be chosen (signed out, too old, not working), or whose first check has not ended (below): then the value is that program, shown as its own disabled entry with its tag, so that choosing Default Auto is a real change (Radix sends no change for the value the `Select` already shows). The trigger then writes the name and the tag in place of the Experimental badge. Its list holds the entries below, in this order |
| The entries | **Default Auto** (always); **Apple Intelligence** (while available); **Claude Code** and **Antigravity (agy)** (while listed). Antigravity carries an **Experimental** `Badge` in the list and in the trigger. |
| An entry that is not available | listed **disabled** (`aria-disabled`, not selectable) with a muted tag after its name: **Not signed in** (`signed_out`), **Update needed** (`too_old`), **Not working** (`unavailable`), and, for the saved program only, **Checking…** (`checking`, below). The hint goes under the row (12.2), because a disabled item shows no tooltip. |
| The hint | under the title, shown whole (`longHint`): the hint of the **effective** provider (12.2) |
| Choosing | an available entry (and Default Auto always, including when it is how a saved program is withdrawn): a provider that needs no consent (`yellowmoon`, `apple`, or a cloud one with its consent saved) is saved at once with `preferences_set({ auto_provider })`; a cloud provider without consent opens the consent dialog (12.3) and the `Select` keeps showing the previous value until the person agrees. The core applies it at once (10.5). |
| Default | `apple` where available, else the dropdown shows Default Auto (the saved choice stays `apple`) |

**While a program is being checked (O7, 2026-10-29).** Nothing runs when the app starts, so when Preferences opens a program has usually not been checked yet: it reads `checking` (3.1) until the check that opening Preferences starts ends, a second or two later (`auto_status_changed` then tells the webview). The row must not misbehave in that time, so:

- **The row's visibility is the usual rule** (the table above) and a program that is only being checked does not make the row appear: a computer with no program would see it come and go. A computer that has one, and no Apple Intelligence and no saved program, sees the row appear when its check ends.
- **A saved program that is being checked** keeps the row, and the dropdown shows it as itself, disabled, with the tag **Checking…** (its entry in the list too, with the `aria-label` "Claude Code, checking…"). Default Auto can still be chosen, which is a real change. The hint under the title is Default Auto's, because Default Auto grades until the program answers (as for a program that is signed out).
- **A line under the row** (a muted `RowNote`, `role="note"`) names the programs that are being checked: "Checking Claude Code…", "Checking Antigravity…" or "Checking Claude Code and Antigravity…". It goes when the checks end; the entries that are known are in the list meanwhile.
- **Nothing says the program is unavailable while it is being checked**: no saved-choice note ("…isn't available now…") and no **Use Default Auto** button for it. They come if the check finds it unavailable.
- Apple Intelligence is never "being checked": it is an operating-system call, read at once.

### 12.2 The hints and the notes

**The hint of the effective provider** (the exact text, one paragraph; British spelling):

| Provider | Hint |
|---|---|
| `yellowmoon` | Default Auto sets the exposure and the white by a fixed rule and leaves every other value at 0. It is the same on every computer and sends nothing anywhere. |
| `apple` | Apple's on-device model looks at each photo and grades it: exposure, highlights, shadows, black level, contrast, warmth, tint, saturation and the colours of the shadows and highlights. It runs on this Mac and sends nothing anywhere. |
| `claude` | Claude Code looks at a small copy of each photo and grades it the same way. Yellowmoon runs the Claude Code installed on this computer and hands it the copy, 768 pixels on the long side, and your instructions. Claude Code sends them to Anthropic, or to the service it is set up to use, with your own sign-in. Yellowmoon itself makes no connection. Answers can differ from run to run. |
| `agy` | Antigravity looks at a small copy of each photo and grades it the same way. Yellowmoon runs the Antigravity (agy) installed on this computer and hands it the copy, 768 pixels on the long side, and your instructions. Antigravity sends them to Google with your own sign-in. Google may keep and review them unless you opt out in Antigravity's own settings, and Google's Antigravity terms apply. Yellowmoon itself makes no connection. Answers can differ from run to run. |

**The Experimental note** (a muted `RowNote` under the hint, with the badge, only while Antigravity is the effective provider): "Experimental. Antigravity has no way to take a picture on its input, so its agent opens the copy as a file. That is slower and less predictable than Claude Code, and the agent may open other tools. Yellowmoon runs it in a sandbox, in an empty folder, and never skips its permission checks."

**Notes for providers that are listed but not usable**, one muted line each (`role="note"`), under the row, in the dropdown's order: the hint column of 3.4, for example "Claude Code is installed but not signed in. Open Terminal, run claude and sign in." (`{version}` filled in).

**The checking line** (12.1, *While a program is being checked*): "Checking Claude Code…", "Checking Antigravity…" or "Checking Claude Code and Antigravity…", a muted `RowNote` (`role="note"`) after the notes above, while a program reads `checking`.

**The saved-choice note** (one line, when the saved `auto_provider` is Claude or Agy and `effective` is `yellowmoon`, **and the saved program is not being checked for the first time**): "Your saved choice, Claude Code, isn't available now, so Default Auto is used until it is." (the provider's name from `AutoProvider::name`). No note when the saved choice is Apple (as before: the row hides when Apple is unavailable). **Beside the note, at the right end of its line, a small text button (`LinkButton`) "Use Default Auto"** that saves `auto_provider: "yellowmoon"` with `preferences_set` (no consent is involved; the saved consent stays). It is how the person withdraws a saved program when the dropdown cannot do it: the program is not listed (uninstalled, or not found any more), or it is available but its consent was never saved, so the dropdown already shows Default Auto and choosing it again would be no change. While the program is listed but cannot be chosen the dropdown does it too (12.1, Control). Once the choice is Default Auto the note and the button go, and with them the row, when nothing else is listed (12.1, Shown). This is what "the person can choose another provider at any time" (10.2) means in the app.

### 12.3 The consent dialog

A glasscn `Dialog` (`src/screens/dialogs/AutoConsent.tsx`), 480 px, opened over Preferences (rendered inside `Preferences` while `store.state.consentFor` is set, so Esc closes only it). It is asked when a cloud provider without consent is chosen (10.2). `data-autofocus` is on **Cancel**.

| | Claude Code | Antigravity |
|---|---|---|
| Title | Send photos to Claude Code? | Send photos to Antigravity? |
| Body | To grade a photo, Yellowmoon runs Claude Code on this computer and hands it a copy of the photo, 768 pixels on the long side, and your instructions. Claude Code sends them to Anthropic, or to the service it is set up to use, with your own account. The original file, its name and its place on disk stay here. Whatever Anthropic keeps depends on your account's privacy settings. Yellowmoon itself makes no connection. | To grade a photo, Yellowmoon runs Antigravity (agy) on this computer and hands it a copy of the photo, 768 pixels on the long side, and your instructions. Antigravity sends them to Google with your own account. Google may keep and review them unless you opt out in Antigravity's own settings, and Google's Antigravity terms apply. The original file, its name and its place on disk stay here. Antigravity is an agent that can open other tools, so Yellowmoon runs it in a sandbox, in an empty folder, and never skips its permission checks. This is experimental. Yellowmoon itself makes no connection. |
| Buttons | **Cancel** (secondary) and **Use Claude Code** (primary) | **Cancel** and **Use Antigravity** |

On Windows each body ends with one more sentence: "The temporary folder's path includes your Windows user name, and the program may send it along." (D11; the webview knows the platform from `app_info`.) **Use …** sends the one patch of 10.2 and closes the dialog; **Cancel**, Esc and a click outside close it and change nothing ("Cancel keeps the previous choice"). The question belongs to Preferences: `consentFor` is cleared whenever the overlay becomes anything but Preferences (About from the native menu, ⌘W's close question, Import .cube, or no dialog), so the next ⌘, never opens Preferences with the dialog already showing.

### 12.4 The instructions field

Unchanged (`docs/19` section 2: the glasscn `Textarea`, the count, **Reset to default**, saving on blur or after 0.8 s, the limit, a change asking every photo again) except that it shows while the dropdown is visible **and the effective provider is not `yellowmoon`**, its label and hint follow the provider, and its text is one for all providers:

| Effective provider | Label | Hint (the count follows: "1689 of 2000 characters.") |
|---|---|---|
| `apple` | Instructions for Apple Intelligence | Apple's model reads these before each photo. The answer form stays the same. |
| `claude` | Instructions for Claude Code | Claude reads these before each photo. The answer form stays the same. |
| `agy` | Instructions for Antigravity | Antigravity reads these before each photo. The answer form stays the same. |

Switching provider keeps the text and the saved instructions; it asks nothing again for a provider whose answers are kept (10.5).

### 12.5 Everywhere else the provider is named

`{Name}` is `AutoProvider::name()` of the provider the photo's state belongs to (the answer's `info.provider`, or `auto.effective` while `pending`).

| Place | Text |
|---|---|
| Automatic card, answered | Set with {Name}. |
| Automatic card, pending | {Name} is looking at this photo… |
| Automatic card, failed | {Name} couldn't look at this photo, so Default Auto is used. |
| Automatic card, `unavailable` with a reason of a program the person chose (`not_installed`, `too_old`, `signed_out`, `probe_failed`, `unsafe_launcher`) | {Name} isn't available right now, so Default Auto is used. (Nothing for Apple, as before; nothing for `off`, `checking` or `not_consented`.) |
| Automatic card, plan | the plan in curly quotes, as before |
| Start screen, a photo that waits | Graded by {Name}… |
| Start screen, the hint | {Name} is looking at each photo. You can start looking while it finishes. |
| Start screen, the title | Grading N photos with {Name} |
| `ym` | section 13 |

### 12.6 The shell and the wire (`src-tauri`)

- **Commands** `auto_status()` (replaces `apple_auto_status()`): `Core::auto_status()`, never blocking on a process (it asks every advisor to probe, in the background, and returns what is cached; `agy`'s sign-in check, which contacts its service, runs then). **Only Preferences opening asks it** (3.3, D7). `auto_status_cached()`: `Core::auto_status_cached()`, the same `AutoStatus` as each advisor last said it, with no probe started. The webview reads it after the preferences change (the choice, the consent or the instructions: typing instructions saves a preference about every five seconds) and on `auto_status_changed`. Both run on `spawn_blocking` and return the same wire type.
- **Event** `auto_status_changed` (no payload; defined like `looks_changed`): emitted from `Core::on_auto_changed`. The webview reads `auto_status_cached()`.
- **Config.** `AppConfig::advisors: ProviderAdvisors` replaces `advisor`: the app builds it from `ym_apple::system_advisor()` and `ym_agents::system_advisors(StartProbe::Never)` (`LAUNCH_PROBE`; O7: nothing runs at launch, 3.3); the shell's tests give none, or scripted ones. The shell calls `Core::cancel_auto()` when the app quits.
- **Wire types** (`wire.rs`, mirrored by `src/api/types.ts` and checked by `contract.ts`): `AutoProvider` (`"yellowmoon" | "apple" | "claude" | "agy"`); `AutoConsent { claude: boolean, agy: boolean }`; `ProviderState`; `ProviderStatus { provider, state, reason: AppleAutoReason | null, info: AppleAutoInfo | null, tool_version: string | null, consent, experimental, listed, selectable }` (no `path`); `AutoStatus { providers: ProviderStatus[], preference, chosen, effective, active, visible, consent, table, instructions, instructions_sha256, own_instructions, default_instructions, instructions_max_chars }`. `AppleAutoReason` gains the seven new reasons of section 11 and `AppleAutoFailure` gains `not_signed_in` and `cli_failed`; `AppleAutoInfo` gains `provider`, `tool_version` (`string | null`), `api` (`string | null`) and `prompt` (`number`), and its `os` is `""` for the programs. `Preferences` has `auto_provider`, `auto_consent` and `apple_auto_instructions` (no `apple_auto`); `PreferencesPatch` the same three, optional. `AppleAutoStatus` and `AppleAutoMode` leave the wire. The photo's `estimates.apple`, `apple_steps`, `apple_pending` and the `apple` stage keep their names (D3).

### 12.7 The store

`state.auto: AutoStatus | null` replaces `appleAuto`; `refreshAuto()` replaces `refreshAppleAuto()` and reads `auto_status()` (the full status: asked only when Preferences opens); `refreshAutoCached()` reads `auto_status_cached()` (asked after a preference change that touches the three fields, and on `auto_status_changed`); `state.consentFor: "claude" | "agy" | null`; `chooseProvider(provider)` (the flow of 12.1) and `answerConsent(agree: boolean)`. `setPreferences` calls `refreshAutoCached()` when `auto_provider`, `auto_consent` or `apple_auto_instructions` is in the patch.

### 12.8 The browser mock

`src/api.mock.ts` plays the providers. `?mockProviders=<list>` takes a comma list of `name[:state]` with `name` one of `apple`, `claude`, `agy` and `state` one of `available` (the default), `signed_out`, `too_old`, `unavailable`, `checking`; `mockProviders=none` is the default (every provider absent, the row hidden: every existing screenshot stays as it is). `?mockAppleAuto=none|unavailable|pending|answered|failed` keeps its meaning for the photos' states and now applies to the effective provider; a value other than `none` implies `apple` in `mockProviders` when that parameter is absent. `?mockAutoProvider=claude` sets the saved choice (default `apple`) and `?mockConsent=claude,agy` the saved consents (default none). `MOCK_APPLE_INFO` gains a `claude` and an `agy` sibling (model `claude-sonnet-5-5`, tool version `2.1.310`; model `default`, tool version `1.4.7`). The mock's `auto_status` follows the rules of 3.1 and 10.4, `preferences_set` follows 10.2 (including the refusal), and `mockApi.setProviders(list)` plays a provider appearing, signing out or leaving (`auto_status_changed`).

### 12.9 Tests and states

`src/screens/AppleAuto.test.tsx` (extended) and a new `src/screens/AutoProviders.test.tsx`: the row hidden with no provider and shown with any; the list in order with Apple only when available; disabled entries with their tags and the notes under the row; Antigravity's badge and note; the hint of each provider; choosing a provider without consent opens the dialog and changes nothing until **Use …**, which sends one patch; Cancel and Esc keep the previous value; the saved-choice note and its **Use Default Auto** button; Default Auto chosen in the dropdown while the saved program is listed but cannot be chosen, and the row kept (with the note and the button) for a saved program that was uninstalled; the consent dialog dropped when another dialog replaces Preferences; the instructions' label and hint per provider; typing the instructions reads the cached status and never the full one; the Automatic card's and the Start screen's lines per provider. `e2e/auto-providers.spec.ts`: the same flow in a browser with `mockProviders`; `e2e/a11y.spec.ts`: names of the select, its entries (`Claude Code, not signed in`), the notes and the dialog's buttons; the glass audit gets two states, **Preferences with every provider listed** (a `Select` open, Claude Code available, Antigravity signed out) and **the consent dialog**. The shell: `src-tauri/src/api/tests.rs` (`auto_status_*` with scripted advisors, the patch refusal, `auto_status_changed`, that `auto_status_cached` asks no advisor to probe while `auto_status` asks every one, and that quitting stops a running request, 12.10; **and, with the doubles recording every run, that launching the app with a saved, consented Claude Code starts no process, that opening photos checks the chosen program (`--version`, the sign-in check) and then asks it once per photo, that a second opening starts no new check, that opening Preferences checks the other program, that no request folder is left, and that the app builds its advisors with `StartProbe::Never`**, O7) and `commands.rs` (the wire shapes, both commands). The frontend tests of the row while a program is being checked (12.1): the saved program shown with **Checking…**, the line "Checking Claude Code…", no "isn't available now" and no **Use Default Auto** for it, the row that is not shown for a program that is only being checked, and the settling when the check ends (`src/screens/AutoProviders.test.tsx`, `src/state/autoProviders.test.ts`, `e2e/auto-providers.spec.ts`).

### 12.10 As built (lane C)

The app and the frontend follow sections 12.1 to 12.9. Where the text was silent or the code needed something it does not name, this is what was done:

- **The operating system.** `app_info()` gains `os` (`std::env::consts::OS`: `macos`, `windows`, `linux`; `AppInfo::os` in `wire.rs`, `AppInfo.os` in `types.ts`), because 12.3 says the webview knows the platform from `app_info` and the command named none. The store asks for it when Preferences opens (`state.os`); the mock says `macos` or `?mockOs=windows|linux`. It words "a terminal window" in the hints (3.4) and adds the Windows sentence to the consent text (D11).
- **When the webview asks again.** `auto_status()` makes the core refresh every advisor, and `agy`'s sign-in check contacts its service (3.3, D7). So only Preferences opening asks it. After a change of the choice, the consent or the instructions (typing instructions saves about every five seconds) and on `auto_status_changed` the store reads `auto_status_cached()`, which asks no program to probe, at any time (the status is only shown in Preferences, but the read is cheap and keeps `state.auto` current). Nothing else needs the whole status: the card names the provider by the photo's own `estimates.provider` (14.1; the wire's `AutoEstimates` carries it, the mirror test of `src-tauri` requires it) and `estimates.apple`, and the Start screen, which has no estimates yet, reads the saved choice (`preferences.auto_provider`), which is the provider that waits while a photo is `pending` (only an acting provider is waited for).
- **The wire of the answer's info.** The core writes what a program says about itself as `tool: { version, api, auth }` (11, `AdvisorInfo`); the webview reads the flat `tool_version` and `api` of 12.6. The wire type `AppleAutoInfo` therefore reads either shape (a manual `Deserialize` in `wire.rs`, the `Serialize` is flat), tested in `src-tauri/src/api/tests.rs`. `AutoSource` on the wire has the four values of 14.1. `AutoConsent` on the wire has both fields required (no `#[serde(default)]`: the core always writes both and the webview always sends both).
- **The shell and quitting.** `Core::shutdown` calls `Core::cancel_auto`, which kills every program Auto started (a request or a probe) and waits, for at most 4 s (`AUTO_CANCEL_WAIT`), until they are reaped and their temporary folders removed (and the session asks nothing more). The app's quit path (`AppApi::shutdown` on `RunEvent::Exit`) therefore returns only then, and the process may end at once; the shell has no call of its own for it. `src-tauri/src/api/tests.rs` proves it (`quitting_stops_a_request_that_is_running_and_leaves_no_program_and_no_folder`): a request of the `slow` scenario of `ym-fake-claude` is in flight, `AppApi::shutdown()` returns, and the double's pid is gone (not even a zombie), its grandchild is gone and no `ym-auto-*` folder of that run is left. Unix only; where the double is not built (a plain `cargo test -p yellowmoon-app`) it prints `SKIPPED` unless `YM_REQUIRE_FAKE_CLI` is set, as `ym-cli`'s tests do. `AppConfig::advisors` is built by `system_advisors()` in `api.rs`: Apple's advisor, then `ym_agents::system_advisors(LAUNCH_PROBE)`, which is `StartProbe::Never` (O7, 2026-10-29: the app starts no program at launch; it was `StartProbe::Now` before).
- **The dropdown.** The `Select` shows `auto_status().effective`, or the saved program while that is listed but cannot be chosen (`dropdownValue`): choosing the value it already shows does nothing (Radix sends no change), so without this a person whose saved program is signed out or too old could not choose Default Auto at all. A saved program that is not listed has no entry to show, so the saved-choice note carries a **Use Default Auto** button (12.2), and the row stays (`autoRowShown`) while the saved choice is a program. In the trigger the tag of the shown program ("Not signed in", "Update needed", "Not working") is drawn tighter than in the list (4 px of padding) so that "Claude Code" and the longest tag fit its 220 px whole; Antigravity's Experimental badge is left out there while the tag shows. Each entry has an `aria-label` that says why it is disabled ("Claude Code, not signed in"; Antigravity's is "Antigravity (agy), experimental"). The trigger, 220 px wide, writes Antigravity as "Antigravity" with a small Experimental badge: the full name and badge are cut off there (the list has both in full). The consent dialog is mounted by `Overlays` beside Preferences, not inside it; the focus goes back to the dropdown when it closes.
- **The mock.** `MockOptions.providers` is `null` by default and then derived from `appleAuto` (so every screen and test of Apple Intelligence is unchanged); `mockProviders` names the computer and the mode of `appleAuto` then plays the photos of the provider that acts, `answered` when it says `none` or `unavailable`. A chosen provider that is not in `mockProviders` is `unavailable` with the reason `not_installed`, which the card names for a program ("Antigravity isn't available right now, …", 12.5) and not for Apple. The mock keeps no per-provider answers: a switch gives every photo the new provider's answer at once.
- **The name "Default Auto"** (owner, 2026-10-26: "Instead of 'Yellowmoon's own Auto' just say 'Default Auto'"). The app says Default Auto for the `yellowmoon` provider in the dropdown, its hint ("Default Auto sets the exposure and the white by a fixed rule …") and the saved-choice note ("… isn't available now, so Default Auto is used until it is."). The wire value stays `yellowmoon`. The card's lines (12.5) first kept "Auto uses Yellowmoon's own settings."; the owner agreed on 2026-10-29 ("yea") that they say "Default Auto is used." as well, like `ym`'s sentences (13.2). `AutoProvider::name`, `ym`'s text and the tables of sections 1 and 11 are the core's and follow lane B.
- **Screenshots.** The two new states are in `design/screenshots-c2/` as `photos-preferences-providers.png` and `photos-preferences-consent.png`, and in the glass audit as `40-dialog-preferences-providers` and `41-dialog-consent`.

## 13. `ym`

### 13.1 Options

`render` and `info` develop photos, so they take these (in `### Run` of `docs/10`):

| Option | Meaning |
|---|---|
| `--auto-provider yellowmoon\|apple\|claude\|agy` | Who grades each photo for this run (any case). Default: the app's setting, `auto_provider`, which acts where that provider is available and, for a cloud provider, where its consent is saved (10.4); otherwise Default Auto. **Naming `claude` or `agy` is the consent for this run** (never saved): `ym` then sends a 768 px picture of each photo and the instructions to that service, through the program installed on this computer. |
| `--apple-auto on\|off` | Kept as an alias: `on` is `--auto-provider apple`, `off` is `--auto-provider yellowmoon`. Giving it with `--auto-provider` is a usage error (exit code 2): "Use --auto-provider or --apple-auto, not both." |
| `--auto-instructions FILE` | The instructions for this run, from a UTF-8 text file, instead of the `apple_auto_instructions` preference. An alias of `--apple-auto-instructions FILE`, with the same rules (the final line break is not part of them; an empty file is the default; a file that cannot be read or is over 2000 characters is a usage error, exit 2). Giving both is a usage error. |

`render --help` says that answers from Claude Code and Antigravity can differ from run to run. `ym` hands the core `ProviderAdvisors` built from `ym_apple::system_advisor()` and `ym_agents::system_advisors(StartProbe::Never)`. It reads the saved choice with `PreferencesStore` first (as `doctor` does), works out the chosen provider (the option, else the saved choice) and, when that is Claude Code or Antigravity **and the run may use it** (it named it, or the saved consent covers it), calls `refresh` and `settle(30 s)` on its advisor alone before the core opens (all providers for `doctor`), so no run starts with `checking` and no run touches a program it does not use: a saved cloud choice whose consent is not saved starts no program at all, in any command (the core reads it as `not_consented`, 10.4). This is the "chosen program, checked once before its first grading" of O7 for `ym`: the core then finds the program `available`, so opening the photos and the export start no second check (3.3); a program the check found signed out is checked again by the session's opening, and the advisor's 5 s minimum interval swallows that repeat. The advisors are registered with the Ctrl-C handler first (9.4), so a Ctrl-C during that wait stops the probe too.

### 13.2 What it prints on stderr

Once per run each, as warnings (`--quiet` hides them), after the core opens:

- **Not available**, only when the run **named** the provider (`--auto-provider`, `--apple-auto on`): for Apple the sentence of `docs/19` section 8 unchanged ("Apple Auto is not available on this computer (<words>). Default Auto is used."); for Claude Code and Antigravity with the reason `not_installed`: "Claude Code is not available on this computer (it isn't installed). Default Auto is used."; with any other reason the hint of 3.4 followed by " Default Auto is used." (for example "Claude Code is installed but not signed in. Open Terminal, run claude and sign in. Default Auto is used."). A run that only follows the saved choice says nothing, as the app hides its row. `render` reads the availability again just before it queues the export and prints the same sentence if the provider went away since (`docs/19` section 8).
- **A cloud provider acts** (named or saved): "Claude Code will send a 768 px picture of each photo and the instructions to Anthropic." (when `apiProvider` is not `firstParty`: "…to the service your Claude Code is set up to use."); "Antigravity (agy) will send a 768 px picture of each photo and the instructions to Google." (`ym_core::auto_provider::cloud_notice`).
- **The breaker** (9.2): "Claude Code failed three times in a row (<the last failure in words>). Photos it doesn't answer use Default Auto."

While the answers come the progress line reads `Asking Claude Code [bar] 3/12` (the provider's `name()`).

### 13.3 `doctor`

`ym doctor` refreshes every advisor and waits at most 30 s for them (a program that hangs ends at its own probe limit). It reports the saved choice, the consent and each provider:

```json
"auto": {
  "preference": "claude", "effective": "claude", "consent": { "claude": true, "agy": false },
  "instructions": "default", "instructions_sha256": "932b96907b63", "table": 2,
  "providers": {
    "yellowmoon": { "state": "available", "reason": null },
    "apple":  { "state": "unavailable", "reason": "not_a_mac", "model": null, "os": null, "prompt": 3 },
    "claude": { "state": "available", "reason": null, "path": "/Users/me/.local/bin/claude", "tool_version": "2.1.310",
                "model": "sonnet", "api": "firstParty", "auth": "oauth_token", "prompt": 1, "schema": 1, "image": "jpeg444q92@768" },
    "agy":    { "state": "signed_out", "reason": "signed_out", "path": "/Users/me/.local/bin/agy", "tool_version": "1.4.7",
                "model": "default", "api": null, "auth": null, "prompt": 1, "schema": 1, "image": "jpeg444q92@768" }
  }
},
"apple_auto": { … }
```

Keys are sorted; `model` is the alias or slug until a request has answered; `path` is the person's own path and appears here only (this is a report for a bug, `docs/10` *doctor*; the sentence that says what it includes names the two programs' places). The old `apple_auto` object stays for one version, unchanged (`docs/19` section 8, now computed for the Apple provider only).

The text rows, after **Render path**: **Auto grading** (`Claude Code (saved in Preferences)`, or `Default Auto`, with `· not agreed to yet; naming it with --auto-provider is enough for ym` when its consent is not saved), **Apple Intelligence** (the row `docs/19` calls "Apple Auto", same text), **Claude Code** and **Antigravity (agy)**:

- `available · Claude Code 2.1.310 · sonnet · prompt 1 · schema 1 · table 2 · /Users/me/.local/bin/claude` (with ` · edited instructions (<digest>)` as 14.3);
- `installed but not signed in · 1.4.7 · /Users/me/.local/bin/agy`; `too old (2.1.100; 2.1.169 or later is needed) · <path>`; `didn't answer its check · <path>`; `found only as a script that Yellowmoon can't run safely`; `not installed`; `still checking`.

### 13.4 Reports

`info --json` and `render --json` carry `auto_source`, `auto_advisor` and the legacy `apple_auto` (section 14); the text row of `ym info` is renamed **Auto grading** and names the provider:

- `Claude Code · Brighter +0.3 EV · highlights -20 · contrast +15 · teal shadows (medium) · scene "Harbour in morning light" · plan "Cool, crisp morning with teal shadows"` (the steps as `docs/19` section 8; the provider's `name()` first and joined with ` · `, the first word of the steps still capitalised; ` (shared with its burst)` as there);
- `Default Auto` (provider `yellowmoon`); `Not available (Claude Code isn't signed in)` (the clause of 3.4); `Waiting for Claude Code`; `Failed (Claude Code reported a rate limit); Default Auto` (9.1; with ` its burst's answer: …` as there).

### 13.5 What does not use a provider

As `docs/19` section 8 says for Apple: `doctor --bench`, `looks bake --preview`, the golden tool, `tests/parity.rs` and `parity_gpu.rs` (except their fake-provider legs), the JSON snapshots, the evaluation tools. All pass `--auto-provider yellowmoon` (or `--apple-auto off`, which is the same) and the tests of `ym` run the binary with **both** `YELLOWMOON_TEST_APPLE_AUTO=none` and `YELLOWMOON_TEST_AUTO_PROVIDERS=none` (`common::NO_AUTO_PROVIDERS`), so that a developer's own Claude Code is never started by a test.

## 14. Reports and metadata

### 14.1 `--json` (D2)

- `auto_source`: `rule`, `apple`, `claude` or `agy`: which Auto made the photo's automatic values (a failed photo that takes its burst's steps counts as the provider).
- `auto_advisor` (new, after `apple_auto`; every photo `info` develops, and in `render` every photo a provider acted on): keys sorted, all present:

```json
{ "api": "firstParty", "image": "jpeg444q92@768", "instructions": "default", "instructions_sha256": "932b96907b63",
  "model": "claude-sonnet-5-5", "os": null, "plan": "Cool, crisp morning with teal shadows", "prompt": 1, "provider": "claude",
  "reason": null, "reproducible": false, "scene": "Harbour in morning light", "schema": 1, "state": "answered",
  "steps": { "…": "as docs/19 section 8" }, "table": 2, "tool_version": "2.1.310" }
```

`provider` and `state` (`off`, `unavailable`, `pending`, `answered`, `failed`) are always set; `reason` is the wire name of the unavailable reason or of the failure; `model`, `os`, `tool_version`, `api`, `instructions` and `instructions_sha256` come from the `AdvisorInfo` of the answer or the failure and are null otherwise; `prompt`, `schema`, `image`, `table` and `reproducible` are the provider's constants (null for `yellowmoon`; `schema` and `image` null for Apple; `reproducible` true for Apple on one Mac and macOS build, false for the programs); `steps`, `scene` and `plan` only when answered. The CLI's path is not in it.
- `apple_auto` (kept **one version**, deprecated): exactly what `docs/19` section 8 describes, for the provider `apple`; for every other provider it is `{ "state": "off" }`. So a script that looks for `apple_auto.state == "answered"` still finds the photos Apple graded, and none of Claude Code's.
- `auto` (`AutoSummary`) keeps `apple` and `apple_steps` (D3), gains `provider` (the chosen provider's wire name) and its `source` takes the four values; the `info` inside `auto.apple` gains `provider`, `prompt` and, for the programs, `tool` (`{ "version", "api", "auth" }`; absent for Apple), and its `os` is `""` for the programs.
- The key order is `…`, `auto`, `auto_source`, `apple_auto`, `auto_advisor`, `settings`, … and keys below the top level are sorted, as `docs/10` says.

### 14.2 XMP (`docs/09`)

`ym:Auto` is `rule`, `apple`, `claude` or `agy`. **`ym:AutoAdvisor` replaces `ym:AppleAuto`** (after `ym:Models`, before `ym:Settings`; compact JSON, keys sorted), written only when a provider acted on the photo (answered or failed); a key without a value is left out, except a step the model left empty, which is `null`. `ym:AppleAuto` is no longer written. A file made before this version carries `ym:AppleAuto`, which is `ym:AutoAdvisor` for the provider `apple` without `provider` and `reproducible`; readers should accept both.

```text
{"api":"firstParty","image":"jpeg444q92@768","instructions":"default","instructions_sha256":"932b96907b63","model":"claude-sonnet-5-5","plan":"…","prompt":1,"provider":"claude","reproducible":false,"result":"answered","scene":"…","schema":1,"steps":{"blacks":null,"contrast":1,"exposure":1,"highlight_color":"warm","highlight_color_strength":1,"highlights":-1,"saturation":0,"shadow_color":"teal","shadow_color_strength":2,"shadows":null,"tint":null,"white":-1},"table":2,"tool_version":"2.1.310"}
{"instructions":"default","instructions_sha256":"932b96907b63","model":"AFM 3 Core","os":"27.0.1 (27A404)","plan":"…","prompt":3,"provider":"apple","reproducible":true,"result":"answered","scene":"…","steps":{…},"table":2}
{"api":"firstParty","image":"jpeg444q92@768","instructions":"default","instructions_sha256":"…","model":"sonnet","prompt":1,"provider":"claude","reason":"timeout","reproducible":false,"result":"failed","schema":1,"table":2,"tool_version":"2.1.310"}
```

The program's path, its messages and the working folder are never written. `ym:Settings` is unchanged: contrast, saturation, highlights, shadows and black and the two wheels are pinned to the numbers Auto resolved to, white balance stays Auto, and the nudges are in `ym:AutoAdvisor`'s steps (`docs/19` section 9). A photo whose answer had not arrived when its session closed under its export is written without `ym:AutoAdvisor`. With Default Auto the file carries `ym:Auto` = `rule` and no `ym:AutoAdvisor`.

### 14.3 The version string (`AdvisorInfo::version_string`)

| Provider | String |
|---|---|
| Apple | `Apple Intelligence · AFM 3 Core · macOS 27.0.1 (27A404) · prompt 3 · table 2` (unchanged) |
| Claude Code | `Claude Code 2.1.310 · claude-sonnet-5-5 · prompt 1 · schema 1 · table 2` |
| Antigravity | `Antigravity 1.4.7 · default · prompt 1 · schema 1 · table 2` |

each with ` · edited instructions (<first 12 hex digits of their SHA-256>)` when the person wrote their own.

### 14.4 Determinism (hard rule 8)

- **Cloud answers are not reproducible.** Neither program has a temperature or a seed, models behind an alias change, and two calls on one picture can answer differently. So the same photo, settings and Yellowmoon version can export two different files with Claude Code or Antigravity chosen, and `ym` and the app, two processes with no shared cache, can grade one photo differently. Every export therefore records which Auto made it (14.2), `ym info` and `ym doctor` print the provider, the program's version and the model, and `reproducible: false` says so. The hint (12.2) and `ym render --help` say "Answers can differ from run to run".
- **The exported picture is reproducible from its own metadata**: `ym:Settings` pins every Auto value to the number it resolved to (`docs/19` section 5), so `ym render --settings <the file's settings>` writes the same pixels whatever any provider says. Only "ask the provider again" is not deterministic, and the metadata says that.
- **Tests, the golden tool and the parity tests never reach a provider**: they run with Default Auto, or with the in-process fakes of 15.3 (app and `ym` write identical bytes with the same fake).
- **Use of the providers' output.** As for Apple (`docs/19` section 7): their answers never train, label or tune anything of ours, and the table's values are chosen by eye, never fitted to a provider's answers. The blind A/B (17) uses them only to judge.

## 15. The crate `ym-agents` and its test doubles

### 15.1 Layout

`crates/ym-agents`, a workspace member, depended on by `ym-cli` and the app (`src-tauri`), never by `ym-core` (D3, like `ym-apple`). It implements `ym_core::apple_auto::AutoAdvisor` twice.

```text
crates/ym-agents/
  Cargo.toml        features: test-support (the helpers of testing.rs)
  src/lib.rs        crate docs, system_advisors(), TEST_PROVIDERS_ENV, re-exports
  src/tool.rs       Tool { Claude, Agy }: names, minimum versions, constants
  src/discovery.rs  section 4
  src/process.rs    section 5
  src/probe.rs      the cache and single-flight probe, refresh/settle/on_change
  src/advisor.rs    what the two advisors share: the reading, the probes of both programs, details(), the skeleton of a request
  src/breaker.rs    9.2
  src/jpeg.rs       6.1
  src/schema.rs     6.2 and 6.3
  src/claude.rs     ClaudeAdvisor, section 7
  src/agy.rs        AgyAdvisor, section 8
  src/env_hook.rs   YELLOWMOON_TEST_AUTO_PROVIDERS, 15.3
  src/testing.rs    (test-support) fake_cli_path(Tool), scenario helpers
  src/fakes.rs      shared by the two test doubles (#[path] include)
  src/bin/ym-fake-claude.rs, src/bin/ym-fake-agy.rs      the test doubles, 15.2
  tests/            discovery.rs, process.rs, process_env.rs, claude.rs, agy.rs, schema.rs, jpeg.rs, breaker.rs, probe.rs, env_hook.rs, doubles.rs, no_network.rs
```

Public API (fixed between the lanes, section 18):

```text
pub enum StartProbe { Now, Never }               // the app and `ym`: Never (O7: nothing runs at launch; the core checks the program it uses, 3.3); Now is kept for tests
pub fn system_advisors(start: StartProbe) -> Vec<(AutoProvider, Arc<dyn AutoAdvisor>)>
    // YELLOWMOON_TEST_AUTO_PROVIDERS unset: [(Claude, ClaudeAdvisor::system()), (Agy, AgyAdvisor::system())]; with Now each is already
    // probing in the background (Claude fully; Agy up to its version); with Never (the app and `ym`) nothing has probed. Set: the advisors of 15.3 (`start` has no effect on the fakes).
pub struct ClaudeAdvisor;  impl ClaudeAdvisor { pub fn system() -> Arc<Self>;  pub fn with_program(path: PathBuf) -> Arc<Self> }
pub struct AgyAdvisor;     impl AgyAdvisor    { pub fn system() -> Arc<Self>;  pub fn with_program(path: PathBuf) -> Arc<Self> }
pub const TEST_PROVIDERS_ENV: &str = "YELLOWMOON_TEST_AUTO_PROVIDERS";
// both advisors implement AutoAdvisor with concurrency() 2 and 1, refresh/refresh_asked/settle/cancel/wait_idle/on_change/details as 3.3, 9.4;
// `with_program` skips discovery (tests, and `exec:` of 15.3).
#[cfg(feature = "test-support")] pub mod testing { pub fn fake_cli_path(tool: Tool) -> PathBuf; /* … */ }
```

Everything else is private. No `unwrap` or `expect` in library code outside tests (`AGENTS.md`).

**Dependencies** (each licence in the PR, `docs/14` rows below):

| Crate | Use | Licence | Notes |
|---|---|---|---|
| `which` 8 | `PATH` and `PATHEXT` lookup (`which_global`) | MIT | new (4.4.2 is in the lock as a build tool only) |
| `process-wrap` 10, `default-features = false`, features `std`, `process-group`, `job-object`, `creation-flags` | process group, job object, `CREATE_NO_WINDOW` | Apache-2.0 OR MIT | new (10.0.1). Its `kill-on-drop` feature is for Tokio only and is not enabled |
| `windows` 0.62, Windows only, `default-features = false`, feature `Win32_System_Threading` | the `CREATE_NO_WINDOW` constant that `CreationFlags` takes | MIT OR Apache-2.0 | already in the lock (process-wrap, Tauri) |
| `zune-jpeg` 0.5 | the test doubles decode the picture they are given | MIT OR Apache-2.0 OR Zlib | also a dev-dependency of `ym-export` and `ym-color`; nothing in the library uses it |
| `tempfile` 3 | the private temp folders | MIT OR Apache-2.0 | workspace dependency |
| `jsonschema` 0.58, `default-features = false` | the answer's validation | MIT | add to `[workspace.dependencies]` (ym-core and ym-profiles name it themselves today); **no default features** (they pull an HTTP client) |
| `jpeg-encoder` 0.7 | the picture | (MIT OR Apache-2.0) AND IJG | workspace dependency, allowed in `deny.toml`; the IJG sentence is already in `NOTICE` |
| `base64` 0.22 | the image block | MIT OR Apache-2.0 | already in the lock |
| `semver` (workspace), `serde`, `serde_json`, `thiserror`, `tracing`, `sha2`/`hex` if needed | | MIT/Apache-2.0 | workspace |
| dev: nothing but this crate itself with `test-support` (*As built*, 15.7) | tests | | |

`cargo deny check` passes with `deny.toml` as it is (all licences are already allowed); no change expected. `docs/14` gets the rows of 15.6.

### 15.2 The test doubles

Two small binaries, `ym-fake-claude` and `ym-fake-agy`, built with the crate (no feature: `cargo test -p ym-agents` builds them; they are not part of any shipped build, which never builds another package's bins). They imitate the output of 2.1.310 and 1.4.7 as these sections describe it and **check what they are given**, so a change of the request that the real program would reject fails a test: the `claude` double requires `-p`, `--input-format stream-json`, `--output-format stream-json`, `--safe-mode`, `--tools` followed by an empty argument, `--no-session-persistence`, a `--json-schema` that parses and has `scene` and `plan` required, a `--system-prompt-file` that exists, a working folder that is empty, and a stdin line whose image block decodes to a JPEG; the `agy` double requires `--sandbox`, `--json-schema` naming an existing file, no `--dangerously-skip-permissions`, a prompt that contains `@photo.jpg`, and a working folder holding exactly `photo.jpg` (a JPEG). A failed check prints the reason on stderr and exits 99.

Controlled by environment variables:

| Variable | Meaning |
|---|---|
| `YM_FAKE_SCENARIO` | the behaviour (below); default `signed_in` |
| `YM_FAKE_RECORD=<folder>` | writes what the double saw: `argv.txt` (one argument per line), `cwd.txt`, `stdin.bin`, `env.txt` (the values of `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC` and `NO_COLOR` and whether `CLAUDECODE` is set), `files.txt` (the names in its working folder) |
| `YM_FAKE_PIDFILE=<file>` | the `slow` scenarios write their own pid and the pid of a grandchild they start, so a test can see that both are gone |
| `YM_FAKE_VERSION` | replaces the version it prints |
| `YM_FAKE_FAILS=<n>` | the first n requests fail (for the breaker) |
| `YM_FAKE_PLAN=<steps>` | what each request does, by its index, as a comma list: `ok`, `fail` (`cli_failed`), `rate` (`rate_limited`), `bad` (an answer that does not validate), `refused` (`claude` only), `signedout` (`not_signed_in`); requests after the list are `ok`. The breaker's tests use it to put a request the breaker does not count between two it does |

Scenarios (both doubles unless stated; "probe" is `--version`, `auth status` or `-p /usage`):

| Scenario | What it does |
|---|---|
| `signed_in` | the probes succeed; a request gets an answer computed from the picture (the mean luma maps to exposure and the other steps as `FakeAdvisor` does, with Apple's case names), a model id (`claude-fake-1`, an `agy` envelope without one) and, for `claude`, a few stream events before the `result` line |
| `signed_out` | `claude auth status` prints `loggedIn:false` and exits 1, a request prints the `is_error: true` result "Not logged in · Please run /login" and exits 1; `agy`'s `-p` prints "Error: authentication required. Run 'agy' to log in." on stderr and exits 1 |
| `too_old` | `--version` prints `2.1.100` or `1.0.9` |
| `probe_hang` | a probe never ends (killed at the probe limit); `claude` and `agy`. `signin_hang` hangs only the sign-in check. A hanging probe writes its pid to `YM_FAKE_PIDFILE` |
| `slow` | a request starts a grandchild, writes the pids and sleeps for ten minutes |
| `slow_deaf` | as `slow`, after ignoring `SIGTERM` (Unix; the grandchild inherits it): only `SIGKILL` ends it |
| `malformed` | the answer has an unknown case name (`muchBrightest`) |
| `no_structured` | success with prose and no `structured_output` (`claude`: `result` text; `agy`: `status` `SUCCESS`) |
| `refused` | `claude`: no `structured_output` and `stop_reason: "refusal"` |
| `rate_limit` | `claude`: an `api_retry` with `error: "rate_limit"`, then `is_error: true` and `api_error_status: 429`; `agy`: exit 3 with an `AGY_ERROR:` line naming `RESOURCE_EXHAUSTED` |
| `unauthenticated` | `claude`: an `api_retry` with `authentication_failed` and an error result; `agy`: exit 3 naming `UNAUTHENTICATED` |
| `status_error` | `agy`: exit 0, `status` `ERROR` with an `error` text |
| `print_timeout` | `agy`: exit 0, `status` `INTERRUPTED`, no `structured_output` |
| `exit1`, `exit2` | `agy`: the exit codes with stderr text |
| `denied` | `agy`: a valid answer with a non-empty `denied_actions` |
| `huge` | writes more than the stdout cap |
| `backpressure` | does not read stdin until it has written 2 MB to stdout |
| `crash` | exits 3 at once with a fake token (`sk-ant-FAKE0123456789`) in stderr |
| `late_signout` | the first request succeeds, later requests and `auth status` say signed out |

### 15.3 The test hook `YELLOWMOON_TEST_AUTO_PROVIDERS`

Like `YELLOWMOON_TEST_APPLE_AUTO` (`docs/19` section 11), read by `ym_agents::system_advisors()`:

| Value | Effect |
|---|---|
| unset | the real advisors with real discovery |
| `none` | no program is looked for; Claude Code and Antigravity are `not_installed` |
| a comma list of `provider=value`, `provider` `claude` or `agy`; a provider that is not listed is `not_installed` | `value` is a state, `available`, `signed_out`, `not_installed`, `too_old`, `unavailable` or `checking`: an in-process `FakeProvider` (no process, deterministic answers, model `Test Claude Code` or `Test Antigravity`, tool version `0.0.0-test`); or `exec:<absolute path>`: the real advisor over that program with discovery skipped (the full stack against a test double: `claude=exec:/…/ym-fake-claude`; the scenario comes from the environment of the test) |
| anything else | printed once on stderr as an error; the provider is `not_installed` |

Apple stays under `YELLOWMOON_TEST_APPLE_AUTO`. `ym-cli`'s tests find the doubles with `ym_agents::testing::fake_cli_path` (the folder of the test executable's `target` profile); a missing double is a failure when `YM_REQUIRE_FAKE_CLI` is set (the CI job sets it) and a printed `SKIPPED` otherwise, as the repository's other tests that need a tool do.

### 15.4 CI

A new workflow `.github/workflows/agents.yml` (on pushes to `main` and on pull requests that touch `crates/ym-agents`, `crates/ym-core/src/apple_auto.rs`, `crates/ym-core/src/auto_provider.rs`, `crates/ym-cli`, `Cargo.toml`, `Cargo.lock` or the workflow itself): a job `agents` with the matrix `os: [ubuntu-latest, macos-15]`, `fail-fast: false`, `YM_REQUIRE_FAKE_CLI: "1"`, steps: checkout, `rustup toolchain install` (reads `rust-toolchain.toml`), `Swatinem/rust-cache@v2`, `cargo clippy -p ym-agents --all-targets -- -D warnings`, `cargo test -p ym-agents --features test-support`, `cargo test -p ym-cli --test auto_providers`. It uses `shell: bash` so one script serves both systems. The main `rust` job (macOS) runs the whole workspace as before. Windows is not run for now (the owner, 2026-10-29: "no windows for now"): the Windows code paths stay in the crate and are compiled and tested nowhere for now.

### 15.5 Where each program's path comes from in tests

`ClaudeAdvisor::with_program(path)` and `exec:` skip discovery; the discovery tests build their own folder trees (4). Nothing in the tests looks for a real `claude` or `agy`.

### 15.6 `docs/14` rows

| Input | Licence | Handling |
|---|---|---|
| Claude Code (`claude`) | Anthropic's legal agreements; all rights reserved | **not shipped, not linked, not modified**: Yellowmoon starts the person's own copy (`docs/22`). The name is used in plain text to say what runs. |
| Antigravity CLI (`agy`) | Google Antigravity Additional Terms of Service; closed source | **not shipped, not linked, not modified**: Yellowmoon starts the person's own copy, marked Experimental (`docs/22`). |
| `which`, `process-wrap`, `base64` | MIT / Apache-2.0 | dependencies of `ym-agents` |

### 15.7 As built (lane A, 2026-10-26)

What the crate does where this file left a choice or turned out different. Each is a decision, not a deviation from a test of section 16.

- **Process handling.** `process-wrap` 10's standard-library API has no `KillOnDrop` (it is Tokio's) and its `JobObject` sets no `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`. `process::run` therefore owns the child in a guard that kills the group or the job and reaps it on every way out, a panic included. **A Windows process that ends without cancelling leaves the child running.** `process-wrap`'s `JobObject` makes its job with `make_job_object(handle, false)`, hard-wired, and keeps the handle private, so `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` cannot be set through it, and closing the handle of a job without that limit kills nothing. (Setting it ourselves would take a job object of our own, nested over `process-wrap`'s: code that nothing here can build or test, so it is not done.) What Yellowmoon can do on Windows is cancel: Ctrl-C and quitting go through `cancel()`, which terminates every job (9.4); a crash, or the task manager, does not. The same is true on Unix, where a child is never stopped by its parent's death. Neither program has a time limit of its own that would end such an orphan: Claude Code has none, and `agy`'s `--print-timeout` ends its turn. `ProcessGroupChild::try_wait` waits for the whole group, so the loop polls the native child and, once it has exited, kills what it left behind before reaping. The child is shared with the cancel source's list (`Arc<Mutex<…>>`) from its spawn to just before its reaping, which is what lets `cancel()` signal it from another thread (9.4). `ProcessGroupChild::try_wait` waits for the whole group, so the loop polls the native child and, once it has exited, kills what it left behind before reaping. Windows needs the `windows` crate (type of `CreationFlags`); `CreationFlags` and `JobObject` may be wrapped in either order.
- **A program that is being written.** `process::run` tries a program again for up to half a second when starting it fails with "text file busy" (`ETXTBSY`): the program's own updater may be replacing the file, and a test that copies a double and starts it at once meets it whenever another thread forks in between.
- **Temporary folders.** `tempfile` makes a folder with the process's umask, not 0700; `Workdir` sets mode 0700 on the folder and on `cwd/` and `in/` itself (a unit test reads the modes back).
- **Cancelling.** A request that is cancelled returns `Err(AdvisorError { kind: Other, message: "the request was cancelled" })` and changes nothing in the breaker. The core drops it (every cancel is followed by a session or instructions change). `CancelSource::cancel` signals the children itself (9.4): the source holds each child as `Arc<Mutex<Box<dyn ChildWrapper>>>` from the spawn (which takes the list's lock, checks the token and spawns in one go, so a cancel finds the child or stops it from starting) to just before the reaping, and sends `SIGTERM` and `SIGCONT` to the group, or terminates the job. `CancelToken` is therefore a handle on the source's state, not on a counter. The `SIGCONT` constant is 18 on Linux and 19 elsewhere (`process-wrap` takes a signal number and does not re-export its `Signal`). The run loop looks at its token **before** it looks at the child: a child that died of the `SIGTERM` that `cancel()` sent is `Cancelled`, not `Exited` with no exit code (which the breaker would count as a failure). The advisor counts its requests in flight; `Core::wait_idle` waits for that count and the probe to reach zero.
- **The breaker.** `Other` and `Unavailable` failures (our own: a temp folder that cannot be made, no program path) change nothing; `AutoAdvisor::details().breaker` is `Some` from the opening until a request that goes through closes the breaker, also after the 60 s.
- **Probes.** The start-up probe of `agy` (`StartProbe::Now`, which no front end passes since O7) does not count toward the 5 s minimum interval, and a full probe asked for while it runs follows it at once, so `auto_status()` right after the start still runs the sign-in check. A change is announced to the listeners **before** `settle` returns. The probes' programs obey `cancel()` too (`Tuning::cancel`, a token the advisor sets from its cancel source for each probe it starts; a request does not read it): the `--version` runs, the login shell, `claude auth status` and `agy -p /usage`. A cancelled probe's reading is dropped and the cache keeps what it had (the minimum interval still counts from its end). **The sign-out hold** of 3.3 and 9.2 is kept in the advisor, beside the cache, and applied to a probe's result under the cache's lock (`Prober::with_merge`), so a probe that ends just as a request finds the program signed out cannot overwrite it; the request side changes the reading with `Prober::modify`, which looks at what it is now. `ClaudeAdvisor::start()` / `AgyAdvisor::start()` (hidden) start the first probe of `StartProbe::Now`; `system_advisors` sweeps stale `ym-auto-*` folders for both values of it.
- **Discovery.** `$SHELL` must be an absolute path whose file name is one of the six; on Windows only a file with the extension `.exe` is started, anything else is a script (resolved for the npm layout, else dropped); `with_program` / `exec:` still run `--version` on the program and judge it as `locate` does, a path that is not a file is `not_installed`; the first `PATH` match that is an absolute path is the one used (a relative entry of `PATH` is skipped with its match; the other matches are not tried); `with_program` / `exec:` judge the file by the rules of the system they run on (`check_program` takes `Os::current()`; `check_program_on` takes the rules to apply, which is how the tests run Windows' on Linux); the login shell runs in a session of its own (`RunSpec::new_session`, `process-wrap`'s `ProcessSession`, feature `process-session`) and a `SIGTERM` to any group is followed by `SIGCONT`, so a stopped one leaves at once.
- **Claude Code's answer.** `api_retry` events decide only when `is_error` is true (a retry that worked is not a failure); the cached `info.model` is the alias until `served_model` says otherwise. A request line that is not valid is the doubles' business (exit 99), never the advisor's: the advisor builds the line by hand so its key order is the spec's.
- **Schema.** The compiled validator is built once per process, not per advisor (it holds no state). The text is built by hand from `schema::FIELDS`; `tests/schema.rs` compares the cases and guides with the Swift source and with the table of 6.2.
- **Redaction** works on words (letters, digits, `-_.+=/`): a word that starts with `sk-`, `AIza` or `ya29.` (also after an `=`), a JWT (`eyJ` and two dots), the word after `Bearer`, and any dot-free run of 40 or more letters, digits, `-_+=` become `…`; a run that holds a `/` (an AWS-style secret key) goes the same way only when it has upper case, lower case and a digit, so that `/usr/local/bin/claude` and a long lower-case path survive. A long mixed-case path with a digit in it (a macOS temporary folder) is redacted too: an accepted false positive.
- **Test hook.** An empty `YELLOWMOON_TEST_AUTO_PROVIDERS` counts as unset. `claude=available` and the other states give `FakeProvider`s whose `details().tool_version` is `0.0.0-test` (not `not_installed`).
- **Tests reach the internals.** The modules are private; with the feature `test-support` they are public and hidden from the documentation, and the crate has itself as a dev-dependency with that feature, so a plain `cargo test -p ym-agents` runs every test. `Tuning` (time limits, breaker, scenario variables for a double, the temporary folder, model and effort) and `ClaudeAdvisor::custom` / `AgyAdvisor::custom` (hidden) are how a test changes them; no test sets an environment variable except `env_hook.rs` and `process_env.rs`, which are single-test files.
- **The doubles** (`src/fakes.rs`) write each invocation to `YM_FAKE_RECORD/run-NNNN/` as well as to the top level (the last run wins there); `argv.json` is the exact argument list (an `agy` prompt holds line breaks, `argv.txt` does not survive them). More variables and scenarios than 15.2 lists: `YM_FAKE_DELAY_MS`, `YM_FAKE_COUNTER` (default: the record folder; where requests are counted for `YM_FAKE_FAILS` and `late_signout`), `YM_FAKE_API_PROVIDER` and `YM_FAKE_AUTH_METHOD` (what `auth status` says), `version.txt` next to a copy of a double (its version, so two folders can hold two versions), the scenarios `extra_keys`, `signin_hang` (the sign-in check hangs, `--version` does not), `early_exit`, and the first argument `--ym-fake-raw`, which runs a scenario without checking a request (the runner's tests). On Windows `env.txt` also has `CONSOLE_PROCESSES=<n>`, which the Windows test reads to see that `CREATE_NO_WINDOW` gave the child a console of its own.
- **CI.** All three steps are required on Linux and macOS (Windows is not run for now, 15.4). `cargo check --target x86_64-pc-windows-msvc` passed for `ym-agents` (library, doubles and tests, clippy with `-D warnings`) and for `ym-cli` on 2026-10-26, so the render stack type-checked on Windows then; no Windows runner has run the steps.

## 16. Tests and checks

**`ym-agents` (Linux and macOS, section 15.4; Windows is not run for now, the owner, 2026-10-29):**

- `discovery.rs`: section 4's tests.
- `process.rs`: section 5's tests, with the doubles; `cancel()` has sent the group its `SIGTERM` when it returns (the thread of the run is held back so that it cannot be the one), a token cancelled before the start starts nothing, a cancel that kills the child is `Cancelled`, a stopped group is ended at once by the time limit, a program that ignores `SIGTERM` is ended by the `SIGKILL`, a run with `new_session` leads its own session (Linux), and a program that answers and exits leaves nothing of its tree behind (`leaves_child`, a grandchild that the run kills with the group: "close them after finish", O7).
- `schema.rs`: the schema text is valid JSON Schema, has the properties in the table's order, `scene` and `plan` required, ASCII only and under 4 KB; its enums and descriptions equal the Swift source's and 6.2's table; mapping every enum name to its step and back; null and unknown keys dropped; an unknown case, a wrong type and a missing `scene` give `bad_answer`; strengths without a colour are ignored; `AdvisorAnswer::checked` still cuts scene and plan.
- `jpeg.rs`: 6.1's test; the same picture gives the same bytes; a picture that is not `w × h × 3` bytes is refused before encoding.
- `claude.rs` and `agy.rs`: every scenario of 15.2 through the real advisor (`with_program`): the argument list, stdin, working folder and environment recorded by the double equal sections 7.2 and 8.2 exactly; the answer maps to the steps the double computed; each failure maps to its kind (7.3, 8.3); a late sign-out ends as `not_signed_in` and flips `availability()` to `signed_out`; a timeout (the limits shortened for the test) kills the tree; cancel returns within 3 s and stores nothing, and after it `wait_idle` sees every child reaped and every folder gone (also for a program that ignores `SIGTERM`, and for a hanging probe); a sign-out that a request found is not undone by an automatic probe, and is by the person's own check, by a success, and after its hold; the breaker ignores `bad_answer`, `refused` and `not_signed_in` in the middle of failures (the double's `YM_FAKE_PLAN`) and opens when the middle request is a counted one; the breaker opens after three failures, refuses without starting a process (the record shows no new run), lets one trial through after the cool-down (shortened) and closes on success; two requests at once with Claude Code's concurrency of 2 run in parallel and share the breaker; the temp folder is gone after each request.
- `probe.rs`: `refresh` never blocks, coalesces calls within the minimum interval, `settle` waits, `on_change` fires once for a change and not for a repeat, a hanging probe ends at its limit with `probe_failed`, `agy`'s sign-in check is not run by the start-up probe, `not_installed` is searched again by `refresh`.
- `no_network.rs`: 2.1's test.
- The doubles' own checks (a bad argument list exits 99) are tested once by running them directly.

**`ym-core`** (`tests/auto_providers.rs`, `preferences.rs` unit tests, `tests/apple_auto.rs` adapted): the migration table; consent in a patch (accepted with it, refused without it, `yellowmoon` and `apple` always accepted, a file with a cloud choice and no consent acts as `yellowmoon`); `Activity::decide`'s six rules, each reachable; the status for every state and `visible` for every combination of providers (none: hidden; only a signed-out program: visible with the entry disabled; Apple alone: visible); `listed` and `selectable`; switching provider asks the photos that have no answer from the new provider, the photo on screen first, keeps the old provider's answers and reuses them when switched back (a scripted advisor counts calls); new instructions drop every provider's answers and cancel running requests; a provider that changes while a request runs stores the answer under the provider asked; two workers (a scripted advisor of concurrency 2 sees two requests at once and never the same photo twice); `retried(provider)`; a `not_signed_in` failure with the provider unavailable on re-read abandons the request and fails nothing, with it available fails the photo; `served_model` reaches `info.model`; a change notification from an advisor reaches the session (`Session::read_auto_availability`) and the `Core::on_auto_changed` callbacks and calls no `refresh()` of any advisor, a request-time sign-out then a re-read does not loop; a saved cloud choice with no consent is never refreshed (`Core::open`, a preference change, the status the webview reads, an export, the session opening), reads `not_consented`, and is refreshed once the run names it (at the session opening) or the consent is saved and photos are opened; **nothing is checked at launch (O7)**: with a saved, consented program that was never checked, `Core::open`, a change of the choice or the consent and the cached status start no check, opening photos starts exactly one (of the chosen program only) while the photos read `checking` and nothing is asked, they are asked and answered by that program when the check ends (not given Default Auto's values), a second opening and an export while it reads `available` start none, an opening checks again a program that was found signed out while an export does not (it checks only a program that still reads `checking` and waits for it), and Preferences opening (`Core::auto_status`) checks every program once; `Core::settle_auto` settles the chosen advisor and not a slow unchosen one, while `ProviderAdvisors::settle` settles all; `cancel_auto` and the session's drop cancel **and wait** (a scripted advisor whose requests take 300 ms to end after the cancel: `cancel_auto` returns only after them, the wait is bounded by `AUTO_CANCEL_WAIT`, nothing is asked afterwards, the cancelled requests store nothing and the halted session is idle), `halt` on a registry alone does the same for the session that holds it; the words (`failure_words`, `unavailable_hint`, `unavailable_words`) for every provider and the three operating systems; `AdvisorInfo::version_string` for the three providers; `advisor_json` for each provider in each state (golden strings of 14.2); golden 61 unchanged (Default Auto).

**`ym-export`:** `ym:AutoAdvisor` after `ym:Models`, before `ym:Settings`, `ym:AppleAuto` absent; `ym:Auto` for each provider.

**`ym-cli`** (`tests/auto_providers.rs`, run on Linux and macOS by 15.4, with the in-process fakes and the doubles through `exec:`; `tests/apple_auto.rs`, `info.rs`, `render.rs` adapted): the options (`--auto-provider` in any case, the alias and its conflict, `--auto-instructions` and its conflict with the old name, an unknown provider is a usage error); the default follows the saved choice only with consent; naming a provider is consent for the run and is not saved; the stderr sentences of 13.2 for each provider and reason, once per run and hidden by `--quiet`; `render --json` and `info --json` `auto_advisor` and legacy `apple_auto` shapes (snapshots), `auto_source`; the row of `ym info`; `doctor` JSON and text with each provider in each state; a full `render` against `ym-fake-claude` (two photos at once, proved by the start times of the two *requests* that the double records (an Antigravity run of two photos, which takes one at a time, shows them a request apart), answers in the file's XMP, the cloud notice, Ctrl-C leaves no child running, also with a double that ignores `SIGTERM` and during the start-up probe, with no program and no `ym-auto-*` folder left when `ym` has ended); a saved cloud choice with no consent saved starts no program (the double records nothing) and `info --json` says `not_consented` every time; a run that names a program checks it once before the core opens and starts no second check when the photos open or the export starts (the double records one `--version`, one sign-in check, then the requests); the breaker's sentence; app and `ym` write identical bytes with the same in-process fake `claude` (the parity test's new leg, JPEG, 16-bit TIFF, DNG); every existing test that spawns `ym` sets `YELLOWMOON_TEST_AUTO_PROVIDERS=none`.

**The app** (`src/`, `src-tauri`, `e2e/`): 12.9.

**Checks that need a signed-in machine** (section 19) are listed under "Waiting on the owner" in `docs/STATUS.md`; no runner has an account, and V1 takes no measurements of its own (`AGENTS.md`).

## 17. What is provisional, and the A/B

| Value | Now | Decided by |
|---|---|---|
| Claude Code model and effort | `sonnet`, `low` | the blind A/B below, per provider |
| Antigravity model and effort | its default model, `low` | the same |
| Hard stops | 90 s and 120 s | the first real timings (19) |
| Concurrency | 2 and 1 | the same; two at once is not tried on a real account |
| The breaker | 3 failures, 60 s | experience |
| Picture encoding | JPEG 4:4:4, q 92 | the sizes and answers on real photos |
| The prompt for `agy` and the answer form's wording | prompt 1, schema 1 | the A/B; any change bumps the version |
| `TABLE` | version 2, shared | `docs/19` section 13 |

**The blind A/B per provider.** `tools/eval/ab` (`docs/19` section 13) gains `--provider apple|claude|agy` on `ab.sh prepare` (default `apple`): the renders use `--auto-provider <p>` (the kit's `ab.sh` passes it, and the `apple` leg of today is `--auto-provider apple`), the kit first runs `ym doctor --json` and stops unless `auto.providers.<p>.state` is `available` ("<Name> is not available on this computer (<clause>). The test needs it signed in."), `plan.json` records the provider, the program's version and the model, one provider is rendered at a time (`--jobs 1` by default for the programs; each call is one photo), and the same provisional gate applies: the provider's win share against the rule at least 60 % (ties as half), the Wilson lower bound above 50 %, no stratum below 45 %, no more "needs a fix" than the rule. It reports the seconds per photo and, for Claude Code, the model ids that answered. About 150 fresh photos per provider; the programs bill the person's own account, so the maintainer chooses when to run it. The model and effort are changed for a run with the developer variables of D18, never by editing code. A provider fails its gate: its row of the dropdown stays but is not recommended in the release notes, and the owner decides (a failed gate is reported, never hidden: `AGENTS.md`).

## 18. Implementation plan

Three lanes, each one task per branch and pull request (`AGENTS.md`), merged in the order below. The interfaces between them are fixed here; a lane that finds one wrong stops and writes the conflict under "Questions for the owner".

### 18.1 The interfaces

| # | Interface | Owner | Used by |
|---|---|---|---|
| I1 | The types of section 11 (`ym_core::auto_provider`, the changes to `apple_auto.rs`, `Preferences`, `PreferencesPatch`, the trait with its new methods and defaults, `ProviderAdvisors`, `FakeProvider`, the words, the constants). **Step 0** below. | B | A, C |
| I2 | `ym_agents`' public API (15.1: `system_advisors(StartProbe)`, `ClaudeAdvisor`, `AgyAdvisor`, `TEST_PROVIDERS_ENV`, `testing::fake_cli_path`) and the hook of 15.3 | A | B (`ym`), C (the shell) |
| I3 | `Core`'s API (11.3): `CoreOptions::{advisors, auto_choice}`, `auto_status`, `set_auto_choice`, `on_auto_changed`, `cancel_auto`, `settle_auto`, and `AutoStatus` with its JSON (11) | B | C |
| I4 | The IPC: `auto_status`, `auto_status_changed`, the preference and patch fields (12.6) | C | the webview, the mock |
| I5 | The `ym` JSON and text shapes (13, 14) and the XMP (14.2) | B | `tools/eval/ab`, the tests |
| I6 | The test hooks: `YELLOWMOON_TEST_AUTO_PROVIDERS` (15.3), the doubles and their variables (15.2), `YM_REQUIRE_FAKE_CLI`, `common::NO_AUTO_PROVIDERS` | A, B | all |

### 18.2 Step 0, the contract (first, small, by B with a skeleton by A)

One pull request (`cli-auto-00-contract`): lane B adds `auto_provider.rs` and the changes of section 11 to `apple_auto.rs`, `preferences.rs` (the new fields and the migration, without the session behaviour), `Preferences`/`PreferencesPatch` and the doc comments; everything compiles and the existing tests pass with the renamed types (`AppleAutoMode` → `AutoChoice`, `AppleAutoStatus` → `AutoStatus`, built from one advisor as before). Lane A adds the `ym-agents` crate with `system_advisors(StartProbe)`, the hook of 15.3 with the in-process fakes only, and `ClaudeAdvisor`/`AgyAdvisor` that report `not_installed`. After it, A, B and C work in parallel against real types.

### 18.3 Lane A: the crate `ym-agents` (Sonnet developer 1)

Tasks: **A1** `tool`, `discovery`, `process`, `probe`, the doubles and their tests (the base of everything); **A2** `jpeg`, `schema`, `claude` with its tests; **A3** `breaker`, concurrency, cancellation and the probes' laziness for `agy`; **A4** `agy`; **A5** `agents.yml`, `no_network.rs`, `docs/14` rows, the licences in the PR descriptions.

Files owned: `crates/ym-agents/**`, `Cargo.toml` (member and workspace dependencies: `which`, `process-wrap`, `base64`, `jsonschema`), `Cargo.lock`, `deny.toml` (only if a licence needs it), `.github/workflows/agents.yml`, `docs/14-licensing.md` (the rows of 15.6), and sections 3 to 9 and 15 of this file when a decision turns out ambiguous (with a note in `docs/STATUS.md`).

Done when: `cargo test -p ym-agents --features test-support` and clippy pass on Linux and macOS in `agents.yml`; the doubles' checks pass for the exact argument lists of 7.2 and 8.2; `cargo deny check` passes; the no-network test passes.

### 18.4 Lane B: the core and `ym` (Sonnet developer 2)

Tasks: **B1** (after step 0) the preference: migration, consent in a patch, `PreferencesPatch::check`, the status, `Activity::decide`; **B2** the session: the provider switch, per-provider answers, workers by concurrency, cancel, the `not_signed_in` rule, `served_model`, change notifications; **B3** reports and metadata: `advisor_json`, `ym:AutoAdvisor`, `auto_advisor`, the legacy `apple_auto`; **B4** `ym`: options, notices, `doctor`, the info row, progress line; **B5** tests and tools: `tests/auto_providers.rs`, the parity leg, snapshots, `common::NO_AUTO_PROVIDERS` in every spawn, `tools/eval/ab`.

Files owned: `crates/ym-core/src/{auto_provider.rs, apple_auto.rs, preferences.rs, session.rs, app.rs, lib.rs, export.rs, auto.rs}` and their tests (`crates/ym-core/tests/{apple_auto.rs, auto_providers.rs, core.rs, export.rs, session_remove.rs}`), `crates/ym-export/src/xmp.rs` and its tests, `crates/ym-cli/**` (`Cargo.toml` gains `ym-agents`; `src/{cli,runtime,doctor,info,render,report,main,develop}.rs`; `tests/**`), `tools/eval/ab/**`, `tools/oracle/compare_metadata.py`, and the docs `00`, `01`, `09`, `10`, `13`, `16`, `19` and `README.md`'s Auto section (once the behaviour exists).

Done when: the checks of section 16 for `ym-core`, `ym-export` and `ym-cli` pass; golden 61 is unchanged; app and `ym` parity holds with the fake provider; `cargo fmt`, clippy and `cargo test --workspace` pass.

### 18.5 Lane C: the app and the frontend (Sonnet developer 3)

Tasks: **C1** the shell: `wire.rs` types, `auto_status`, `auto_status_changed`, `AppConfig::advisors`, `cancel_auto` on quit, bindings, `contract.ts`; **C2** the Preferences row: `Select`, badge, notes, hints, the instructions' label, the saved-choice note; **C3** the consent dialog and the store (`consentFor`, `chooseProvider`, `answerConsent`); **C4** the card's and the Start screen's lines; **C5** the mock (`mockProviders`, `mockAutoProvider`, `mockConsent`), vitest, Playwright, a11y and the glass audit's two new states, `docs/11`.

Files owned: `src-tauri/{Cargo.toml, src/api.rs, src/commands.rs, src/wire.rs, src/bindings.rs, src/events.rs, src/app.rs, src/testing.rs, src/api/tests.rs, tests/app_cli_parity.rs}`, `src/api/{types.ts, bindings.ts, contract.ts}`, `src/api.ts`, `src/api.mock.ts`, `src/api/appleAutoMock.ts` and its test, `src/state/store.ts`, `src/screens/dialogs/{Preferences.tsx, AutoConsent.tsx, Overlays.tsx}`, `src/screens/tools/LookTab.tsx`, `src/screens/Start.tsx`, their tests, `e2e/**`, `design/screenshots-c2/` (the new states), and `docs/11-ui.md`.

Done when: `npm test`, `npm run build`, `npm run e2e` pass; the shell's tests pass with scripted advisors; the macOS shell type-checks (`cargo check --target aarch64-apple-darwin -p yellowmoon-app`, `docs/13`); what needs a Mac stays under "Waiting on the owner".

### 18.6 Order and dependencies

```text
step 0 (B+A skeleton) ──┬─ A1 → A2 → A3 → A4 → A5
                        ├─ B1 → B2 → B3 → B4 → B5          (B4 needs A's I2 from step 0; B5's exec: tests need A1's doubles)
                        └─ C1 (needs I3 from B1) → C2 → C3 → C4 → C5   (C2–C5 run against the mock and need no real core)
final: a joint pass on Linux and macOS (agents.yml; no Windows for now, the owner 2026-10-29), the Mac checks of section 19, STATUS.
```

Every lane updates the spec sections it changes and `docs/STATUS.md` in the same pull request (`AGENTS.md`), keeps `docs/STATUS.md`'s "Auto providers (branch cli-auto)" paragraph current and moves finished checks out of "Waiting on the owner".

## 19. What a signed-in machine must still check

None of this can run in CI (no runner has an account), and V1 takes no measurements of its own, so each is a [HUMAN] check listed in `docs/STATUS.md` (AP-1 for Claude Code, Windows and Linux included, AP-2 for Antigravity). The research marked all of them **unverified**.

**Claude Code** (a Mac, then Windows and Linux): a real request with the picture, the schema and `--effort low` on `sonnet`: does the model follow the `StructuredOutput` tool reliably, how often does it answer in words, the time and tokens per photo; `--safe-mode` with a real subscription login (the macOS keychain item read from a Finder-started app without a prompt; the app is not sandboxed); `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1` leaves the OAuth refresh working; `claude auth status` makes no network request; the list of parent-session variables of section 5; `--json-schema` as one argument on Windows; the three Windows installs (native, WinGet, npm with its shim); a rate limit mid-batch.

**Antigravity:** `agy -p` signed out (the exit code and message of 8.3), the `-p /usage` check (that it starts no agent turn and what it prints), that `@photo.jpg` in `-p` gives the model the pixels (the owner's statement, 2026-10-26), the workspace-trust behaviour in an untrusted temp folder, `--sandbox` on macOS, Linux and Windows (if it is refused on one, the provider is `unavailable` there), the conversation files the program leaves and whether a script may remove them, an environment switch for its telemetry and self-update, `AGY_ERROR`'s field names, `--json-schema`'s enforcement, the default model's slug (`agy models --output-format json`), and the tool-less custom agent (`excludeDefaultComponents`, 1.2.1) as a way to remove the agent's tools. If `@photo.jpg` or the untrusted folder fails in `-p`, the `agy` provider is dropped rather than worked around.

**Both:** the login-shell lookup against real zsh, bash and fish set-ups; the seconds and the tokens per photo and the effect of two requests at once on a plan's limits; that a hard stop of 90 and 120 s is long enough; that a person with Claude Code on Bedrock or Vertex gets the notice's "service your Claude Code is set up to use".

