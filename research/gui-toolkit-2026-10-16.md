# Rebuilding the Yellowmoon interface in Slint, and what else could fix it

For Mykyta, 2026-10-16. Research only. No existing file was edited and nothing was built; this file is the only thing written to the repository. Everything external was read on 2026-10-16 from this machine; sources are tagged `[S1]`, `[S2]` and listed with URLs and dates at the end. "Repo:" paths are files in this repository. There is no Mac here, so every statement about how something feels or looks on macOS comes from documentation, changelogs and source code, not from running it. Licence remarks are my reading of the texts, not legal advice.

## 0. Short answer

1. **Recommendation: do not move to Slint now.** Both symptoms you named are missing rules in our own page, not limits of WKWebView, and the "sluggish" part may well sit in our per-event frame path rather than in the toolkit (section 4.1; I read the code, I did not run it on a Mac). Fix the two symptoms first (about half a day), measure the rest (about half a day), then decide. If you still want to leave the webview afterwards, Slint can do the job, but only with an explicit licence exception from you (below), and it has three open technical points that decide the case: colour tagging of the photo, wgpu texture interop, and the feel of macOS scrolling. Section 5 is a two-day spike that tests those first. If you refuse the licence exception, run the same spike on egui instead (MIT or Apache-2.0, already on wgpu 30).
2. **Licence verdict.** Slint is offered under three licences, chosen by the user. GPLv3 would make the distributed app binary GPLv3 (the Apache-2.0 source files may stay Apache-2.0, but the combined work must be GPL), which `deny.toml` and `docs/14` forbid. The commercial licence is paid per seat. The Royalty-free licence 2.0 costs nothing and does not touch our source files, but it is not an open-source licence (desktop, mobile and web only, no embedded use, attribution required, notices must stay). It is therefore **not "compatible with Apache-2.0" in the sense of AGENTS.md rule 1 and `deny.toml`; adopting it needs your explicit exception** and changes to `docs/14`, `deny.toml`, `NOTICE` and the About dialog. Forks stay possible: Slint's own FAQ says a forker may pick any of the three Slint licences. Details in section 1.
3. **Top five risks of a Slint rewrite.**
   1. Licence: a policy exception, an attribution duty on every build and every fork, and no embedded use (section 1).
   2. Colour: Slint draws into a Metal layer it does not tag with a colour space and has no setting for it. Our viewport is Display P3 today. wgpu's own Metal code resets the layer's colour space on every surface reconfigure, so an outside fix must be re-applied after each resize (section 2.2).
   3. wgpu interop is an `unstable-wgpu-30` feature whose name changed in six of the last seven minor releases, imports only `Rgba8Unorm` and `Rgba8UnormSrgb`, and had a macOS Skia bug (pink flashes, issue 12271, fixed in 1.17.1). Today's viewport is not zero-copy either, so the gain needs work in `ym-gpu` (section 2.1).
   4. macOS feel: Slint's own tracking issue says scroll inertia "feels wrong" on macOS; the app menu is a fixed default; smart zoom and Finder drops need winit hooks; accessibility rests on AccessKit and a known per-frame cost (issue 3867); our a11y suite has 15 tests that the DOM gives for free (section 2).
   5. Size and verification: about 18.7k lines of TypeScript, 2.2k of CSS, 13.9k of Vitest and 3.8k of Playwright to redo, with every Mac-only behaviour needing a round trip with your Mac. And the rewrite may not remove the lag, because the render path stays (section 3).
4. **Estimate.** About 16 agent-days for the spike plus the rewrite to parity (range 11 to 25), plus 5 to 8 Mac round trips that are not in that number. Phased plan in section 3.3. Keeping Tauri and fixing the web page and the input path costs about 1 day for the page fixes and 4 to 6 more for native input and GPU pan and zoom, if the measurements ask for them (section 4.1).

## 1. Licence

### 1.1 What Slint offers

The crates on crates.io (`slint` 1.18.1, 2026-09-21) declare `GPL-3.0-only OR LicenseRef-Slint-Royalty-free-2.0 OR LicenseRef-Slint-Software-3.0` [S7]. The repository's `LICENSE.md` says: "You can use the Slint framework under ***any*** of the following licenses, at your choice" [S3]:

| Licence | What `LICENSE.md` says it permits | Cost |
|---|---|---|
| Royalty-free 2.0 | "Permits use in **proprietary** desktop, mobile, and web applications **at no cost**. Use in embedded systems is excluded." | none |
| GNU GPLv3 | "Permits use in **open source software** under GPL-compatible terms, **at no cost**, for desktop, mobile, and web applications, as well as for embedded systems." | none |
| Commercial (Slint Software License 3.0.5) | "Permits use in **proprietary** applications, including desktop, mobile, web, and embedded systems." | paid plan, per seat; royalties for embedded [S5] |

The documentation, examples and some assets are MIT; contributions to the Slint repository are MIT-0 [S3].

### 1.2 The text that decides it

Royalty-free licence 2.0, the parts that matter [S4]:

- Grant (section 1): "SixtyFPS hereby grants You a world-wide, royalty-free, non-exclusive license to use, reproduce, make available, modify, display, perform, distribute the Software as part of a Desktop, Mobile, or Web Application."
- Attribution (section 2): "You may distribute the Software as part of an Application, modified or unmodified, provided that You do either of the following: (a) Display the `AboutSlint` widget in an "About" screen or dialog that is accessible from the top level menu of the Application. In the absence of such a screen or dialog, display the widget in the "Splash Screen" of the Application. (b) Display the Slint attribution badge on a public webpage, preferably where the binaries of your Application can be downloaded from, in such a way that it can be easily found by any visitor to that page."
- Limits (section 3): "The License does not permit to distribute or make the Software publicly available alone and without integration into an Application." "The License does not permit the use of the Software within Embedded Systems." "The License does not permit the distribution of Application that exposes the APIs, in part or in total, of the Software." "You may not remove or alter any license notices ... contained within the source code form of the Software."

`AboutSlint` is a widget that "displays a 'Made with Slint' badge" [S23]. The badge file is `MadeWithSlint-logo-whitebg.png` in the Slint repository.

### 1.3 What each choice means for Yellowmoon

**GPLv3.** Slint's FAQ: "While your software modules can remain under the MIT-license, the work as a whole must be licensed under the GPL" [S6]. `LICENSE.md` says the same in its own words: "Use Slint for free under the GPLv3 on any platform; your own files can stay MIT or Apache-2.0" [S3]. Legally this works in one direction: the Apache Software Foundation says "Apache 2 software can therefore be included in GPLv3 projects ... However, GPLv3 software cannot be included in Apache projects" [S8]. So the Apache-2.0 source stays Apache-2.0, but the app binary that links Slint is GPLv3, and anyone who wants a proprietary fork must drop Slint or buy a licence (Slint FAQ, "My MIT-licensed program links to Slint GPLv3. Can someone fork my program to build and distribute a proprietary program?" [S6]). This contradicts the header of `deny.toml` ("Shipped crates must be Apache-2.0 compatible. LGPL/GPL/AGPL are never allowed in shipped crates") and `docs/14` ("Everything shipped must be compatible with it"). **Verdict: not usable under the repository's rules.**

**Commercial.** Needs a paid plan with one seat per person who designs, develops or tests the app (section 2(a) of the licence [S5]). It cannot be handed to forks. **Verdict: not a fit for a free, forkable project.**

**Royalty-free 2.0.** No fee and no copyleft, so our own source files and the `ym` CLI stay Apache-2.0. But the licence restricts the field of use (no embedded systems) and adds duties on whoever distributes a binary, and the Open Source Definition says a licence "must not restrict anyone from making use of the program in a specific field of endeavor" [S9]. It is not an open-source licence, it is not in the `deny.toml` allow list, and AGENTS.md rule 1 ("Ship only code and data compatible with Apache-2.0") and the first line of `docs/14` ("Everything shipped must be compatible with it") are not met in the plain sense of a permissive dependency. It is legal to combine with an Apache-2.0 app: Apache-2.0 sets no limit on what the app links to. So the honest verdict is: **usable at no cost, but only if you decide that a free source-available licence with an attribution duty is an acceptable exception to rule 1.**

### 1.4 Forks

Slint's FAQ covers it directly: "If the forked application also uses Slint, then the person forking the application can choose to use Slint under any one of the licenses - Royalty-free, GPLv3, or Commercial license" [S6]. So a fork of an Apache-2.0 Yellowmoon that runs on desktops is fine under the Royalty-free licence if it keeps the attribution. A fork for an embedded device cannot use the Royalty-free licence; it would need GPLv3 (and then the fork is GPL as a whole) or a paid licence. The FAQ also says each person using Slint under a proprietary licence must have acquired it themselves.

### 1.5 If you accept the Royalty-free exception: what has to change

- AGENTS.md rule 1, `docs/14` (new table row and a rule), `deny.toml` (the SPDX expression contains `LicenseRef-...` names that are not on the SPDX list; each Slint crate needs an exception or clarification; the exact syntax is for the task that does it), `NOTICE`.
- The licence texts must be in the bundle: `tools/licences/generate.py` (cargo-about) has to learn the two `LicenseRef` names.
- About: put `AboutSlint` in an in-window About dialog reachable from the app menu (the current About is the system panel, `src-tauri/src/about.rs`, plus an in-window dialog, `src/screens/dialogs/About.tsx`). Also put the badge in `README.md` and the release notes, which satisfies option (b).
- Pin the Slint version: the licence text is per release (1.0.1 became 2.0 and added "Mobile"; the repository says it "commit[s] to the larger Slint community to provide Slint under a Royalty-free license" [S6]).
- Keep Slint types out of any published library crate (the "exposes the APIs" limit). Our `ym-*` crates do not use Slint; only the app crate would.
- Ask `info@slint.dev` to confirm the reading for an Apache-2.0 project (Slint's own `LICENSE.md` invites licensing questions there [S3]).

## 2. Technical fit on macOS

Slint 1.18.1 is the latest release (2026-09-21); 1.18.0 was 2026-09-16, 1.17.1 2026-07-07, 1.17.0 2026-06-24, 1.16.0 2026-04-16 [S1]. The changelog on `master` has a 1.19.0 section marked "Unreleased" [S2]. The winit backend is the default on all platforms since 1.16.0 [S2]. On macOS, Slint 1.18 renders with Skia (Metal), FemtoVG (OpenGL, or Metal through `renderer-femtovg-wgpu`) or the experimental Vello; the default features are FemtoVG plus software, and Skia is picked first at run time when it is compiled in [S10, S22].

Status words: **works today**, **works with effort** (how), **missing**.

### 2.1 Showing our wgpu output on screen

**Works with effort.**

What exists [S10, S11]:
- Cargo feature `unstable-wgpu-30` (and `unstable-wgpu-29`), module `slint::wgpu_30`, `BackendSelector::require_wgpu_30(WGPUConfiguration)`. `WGPUConfiguration::Manual { instance, adapter, device, queue }` lets Slint use a device we opened; `Automatic(WGPUSettings)` lets Slint open it with required features and limits we give.
- `slint::Image::try_from(wgpu::Texture)` wraps a texture as an image; `Image::to_wgpu_30_texture()` gets it back; `Window::set_rendering_notifier` gives callbacks with the device and queue for an underlay or overlay.
- Only `Rgba8Unorm` and `Rgba8UnormSrgb` can be imported, and the usage must include `TEXTURE_BINDING` and `RENDER_ATTACHMENT` (`TextureImportError`).
- Renderers that take it: Skia (wgpu), FemtoVG (wgpu) and experimental Vello [S2, S10, S22]. `require_wgpu_30` fails fast without a GPU adapter ("Set SLINT_WGPU_CPU=1 to allow CPU adapters"), which our CPU-only fallback would have to catch [S22].
- Our workspace uses `wgpu = "30"`; wgpu 30.0.1 is current (2026-08-22) [S26]. Slint 1.18 is on wgpu 30, so one copy of the crate can serve both.

Stability:
- The module documentation says it "is behind a feature flag and may be removed or changed in future minor releases, as new major WGPU releases become available" [S10]. In practice the flag changed in six of the last seven minor releases: wgpu 24 in 1.12.0, 26 in 1.13.0, 27 in 1.14.0, 28 in 1.15.0, 29 in 1.17.0, 30 in 1.18.0 (1.16.0 had none) [S2].
- Issue 12271: on macOS with Skia and wgpu 29, windows showed "periodic flashes of pink". The maintainer's cause: wgpu had removed the accessor for its Metal command queue, so Slint made its own queue and the two queues were not synchronised. Fixed for 1.17.1 once wgpu 29.0.4 re-added the API [S15]. This is exactly the shape of our use (our compute and Slint's drawing sharing a device).
- Open issue 12030 (title seen in the issue list): a rendering notifier with the FemtoVG renderer recreates render pipelines on every frame [S24]. Avoid FemtoVG for a notifier-based viewport.

What our code does today, so that "without a copy" is measured against the right thing: `RenderedFrame.rgba` is a `Vec<u8>` in CPU memory (`crates/ym-core/src/pipeline/frame.rs`); `src-tauri/src/native.rs` opens its own `wgpu::Instance` and device, uploads the bytes with `queue.write_texture` and draws one triangle. `docs/gpu-plan.md` section 4.8 records that this stays: "own device, uploads the core's RGBA8 ... one 5.5 to 22 MB copy per frame; a shared surface is a Mac tuning item". So Slint would not be worse (`Image::from_rgba8` is the same copy), and a true zero-copy path needs `ym-core` to render the viewport frame into a texture on a device that Slint also uses (either pass `ym-gpu`'s device in `Manual`, or take Slint's device and hand it to `ym-gpu`). That change is worth doing for any toolkit, and it is independent of the choice.

### 2.2 Display P3 and colour management

**Missing in Slint; works with effort, unproven.**

Today: the native viewport is a `CAMetalLayer` configured through wgpu with `SurfaceColorSpace::DisplayP3`, so macOS converts our P3 frames to whatever display the window is on (`docs/01` Native viewport, point 4; `native.rs`).

In Slint:
- A search of Slint's wgpu surface code (`internal/core/graphics/wgpu_30.rs`, `internal/renderers/skia/wgpu_30_surface.rs`, `internal/renderers/femtovg/wgpu.rs`) finds no colour-space setting. It picks `Rgba8Unorm` or `Bgra8Unorm` and `PresentMode::AutoVsync` [S11]. Slint's changelog has no entry about P3, wide gamut or colour spaces [S2].
- wgpu 30's Metal code maps `SurfaceColorSpace::Srgb` to "the layer's default, which treats contents as sRGB" (its own comment) and `DisplayP3` to `kCGColorSpaceDisplayP3`; and it calls `setColorspace` on **every** `configure` [S12]. Apple's documentation of the property says the opposite about the default: "The default value is `nil`, indicating that the rendered content isn't color-matched" [S14]. Which of the two is true on a P3 panel decides whether Slint's own UI colours (for example the yellow `#FFC20E`) show correctly, and it must be measured.
- Fix paths: (a) after Slint creates the window, find the layer through the winit window handle with `objc2` and set `colorspace` to Display P3; because wgpu resets it on each reconfigure, re-apply after each resize; then convert every UI colour of `design/tokens.json` to P3-encoded values and export thumbnails in P3 (all UI colour is flat tokens, so this is mechanical). (b) Output the viewport in sRGB when Slint's layer is sRGB. Our default export is sRGB (`docs/09`), so the viewport would match the default export exactly and lose only the part of P3 outside sRGB. This changes `docs/01` Native viewport point 4. (c) Keep our own `NSView` and layer behind a transparent Slint window, as now. Slint 1.18 makes a window transparent when its background is translucent or it has no frame [S2], but winit makes Slint's view the content view, so our layer would have to sit in a child window or a re-parented view; not tested, and Spaces, full screen and Mission Control checks come with it.
- GPUI has the same gap: its Metal layer is `BGRA8Unorm`, and I found no colour-space setting in its renderer or macOS window files [S30]. GPUI's `surface(CVPixelBuffer)` element is the only route that carries colour attachments per picture.

### 2.3 HiDPI

**Works today.** Scale factor comes from winit; Slint has a physical-pixel length (`phx`, "an amount of physical pixels") [S23]. For a 1:1 photo the image must be placed at integer device pixels, with `image-rendering: pixelated` if needed (the `Image` element has `image-rendering`, default `smooth`) [S23]. `docs/01` already requires an exact 100% origin; a spike check, not a doubt.

### 2.4 Trackpad: two-finger scroll with momentum, pinch, smart zoom

- **Two-finger scroll with momentum: works with effort.** winit's macOS view handles `scrollWheel:` with precise deltas and gives the momentum phase priority over the touch phase, so the OS's momentum arrives as scroll events (winit 0.30.12 `view.rs`, lines 657 to 690) [S18]. Slint converts them to logical pixels and passes them on; `TouchArea.scroll-event` gives `delta-x`, `delta-y` and `modifiers` [S17, S21]. A photo pan built on `scroll-event` therefore follows the OS momentum. The sidebar and look grid would use Slint's `Flickable`, whose wheel scrolling is animated since 1.17.0 [S2] and which has open work: tracking issue 9803 ("ScrollView Tracking Issue", opened 2025-10-20) lists "Inertia feels wrong on Windows, macOS, Android, iOS, and Wasm" (issue 9652) and no native overscroll or overlay scrollbars [S24]. Flickable fixes keep landing (issue 13542, a jump at the start of a touchpad scroll, merged 2026-09-21; 1.18.1 "Nested Flickables can be scrolled with a touch pad again") [S2, S24]. Plan: write our own scroll component on `scroll-event` so the OS provides the feel, or accept Flickable after a feel test.
- **Pinch zoom: works today.** `ScaleRotateGestureHandler` (added in 1.16.0) "supports touchscreens on all platforms, and additionally supports trackpad gestures on macOS and iOS"; `scale` is cumulative from 1.0 and `center` is the cursor position for trackpad gestures [S16]. Slint uses the last cursor position because winit's pinch event has no position [S21].
- **Smart zoom (two-finger double tap): works with effort.** winit has `WindowEvent::DoubleTapGesture` ("On a Mac, smart magnification is triggered by a double tap with two fingers") [S18]. Slint's winit adapter ignores it (its event match ends in `_ => {}`) [S21], but `Window::on_winit_window_event` (feature `unstable-winit-030`) lets us handle it ourselves [S10, S22]. AppKit's own hook is `smartMagnify(with:)` [S27].
- **No rubber-banding of the whole window, no selectable labels: works today.** A Slint window is not a scrolling document, `Text` is not selectable, and only `TextInput` can show a selection.

### 2.5 Native macOS menu bar

**Works with effort.**
- `MenuBar` in Slint builds a native bar with `muda`; `MenuItem.shortcut` exists since 1.16.0 and the shortcuts show in the native menu since 1.16.1; `ContextMenuArea` is native on macOS since 1.16.0 [S2, S22].
- Limit: "Until we have menu roles, always create an app menu on macOS": Slint builds a fixed app menu (About, Services, Hide, Hide Others, Show All, Quit) [S22]. Our spec's app menu also has Preferences… (⌘,), Check for Updates… and Install Command-Line Tool… (`docs/11`, `src-tauri/src/menu.rs`, 628 lines). Two ways: put those items in another menu, or turn Slint's default bar off (`with_default_menu_bar(false)` exists on the winit backend builder, with the comment "Set this to false if you're providing your own menu bar") and build the bar with `muda` ourselves from the `spec()` data we already have [S22]. Whether that builder option is reachable from the public `slint` crate is not verified; it is a spike item.
- A native "About" panel with credits (the IJG and Adobe lines, `about.rs`) would be ours with `muda`'s predefined About item, plus an in-window dialog holding `AboutSlint`.

### 2.6 Window chrome: overlay title bar with the traffic lights

**Works with effort.** winit 0.30 has the same switches Tauri's `titleBarStyle: Overlay` sets: `with_titlebar_transparent`, `with_title_hidden`, `with_fullsize_content_view` [S19]. Slint exposes them through `BackendSelector::with_winit_window_attributes_hook` (feature `unstable-winit-030`) [S10]. `WindowMoveArea` (new in 1.18.0) starts an interactive window move from any element, on macOS too [S2, S20]. The `Window.safe-area-insets` property is documented as covering "the window buttons on macOS on windows that extend their content over the titlebar", but Slint's winit adapter implements it for iOS only [S20, S21]; keep our 84 px padding as a constant. `Window.no-frame`, `full-screen`, `min-width`, `min-height` exist [S20]. Slint's issue 10451 (open) asks for a first-click-goes-to-the-app flag on macOS; winit has `with_accepts_first_mouse` for the hook [S19, S24].

### 2.7 File dialogs

**Works with effort.** Slint has no dialog element. `rfd` 0.17.2 (MIT, 2026-01-12) is the crate behind `tauri-plugin-dialog` (it is in that plugin's dependency list) [S26]; call it with the winit window as parent (`with_winit_window`) from the event loop thread.

### 2.8 Dropping files and folders from Finder

**Works with effort.** winit emits `HoveredFile`, `HoveredFileCancelled` and `DroppedFile` per path [S18]; Slint's winit adapter does not forward them (no match arm) [S21]; Slint's `DragArea` and `DropArea` (1.17.0) are for drags within a window, and 1.18.0 added native drag-out only for the Qt backend [S2]. Handle the three events in `on_winit_window_event` and call `Core::open_paths`, as the webview does now. egui does this out of the box (`egui-winit` turns them into `hovered_files` and `dropped_files`) [S28].

### 2.9 Keyboard shortcuts, IME and text input

**Works today for shortcuts, with effort for the rest.** `KeyBinding`, `@keys(...)` and `MenuItem.shortcut` landed in 1.16.0; `FocusScope` handles plain keys, and a focused `TextInput` takes keys first, which is what `docs/11` wants for ⌘Z and single keys [S2]. IME: winit delivers it; 1.18.0 fixed "IME updates when a focused TextInput moves" and added `input-method-hints` [S2]. We have few text fields (search, value boxes, file name template), so IME is a low risk, and untested here.

### 2.10 Accessibility and VoiceOver

**Works with effort.** The `accessibility` feature is on by default and exposes the element tree through AccessKit [S10]; `accesskit_macos` 0.27.1 was published 2026-09-25 [S26]. Slint has roles including `slider`, `tab`, `tab-list`, `switch`, `radio-button`, `list-item`, `window-title-bar` and landmark roles (`banner`, `main`, `navigation`, `search` and others), plus `accessible-label`, `-value`, `-value-minimum`/`-maximum`/`-step`, `-checked`, `-expanded` and the actions `increment`, `decrement` and `set-value` [S23]. 1.18.0 exposes text selection to assistive technology; 1.19 (unreleased) fixes text positions [S2]. Open issue 3867: the AccessKit code made large `ListView`s slow; on 2026-10-11 a contributor measured on macOS that list length no longer matters but every row created or destroyed rebuilds the tree, about 0.2 to 1.5 ms per frame [S24]. What we have to rebuild: `e2e/a11y.spec.ts` checks names, one main landmark, keyboard-only flows, dialog focus traps and focus rings on every Tab stop (15 tests); the DOM gives most of that for free and Slint needs every custom control annotated. VoiceOver itself can only be checked on a Mac.

### 2.11 Our own widgets

**Works with effort for all, nothing is missing outright.** Slint has `Rectangle`, `Path` (SVG path commands, strokes, 1.19 adds dashes), `Image`, `TouchArea`, `FocusScope`, `Flickable`, `ListView`, `PopupWindow`, conic gradients and animations [S2, S23].

| Our widget | How in Slint |
|---|---|
| Slider with a detent, fill bar, ValueBox | custom: `TouchArea` plus rectangles and a `TextInput`; the standard `Slider` has no detent. `onChange(value, final)` contract becomes two callbacks |
| Segmented, Switch, Tag, Card, TabBar | rectangles plus `TouchArea` and `accessible-*` |
| Curve editor (320 × 220, up to 16 points, drag off the plot to delete) | `Path` for the spline (our `curve.ts` math moves to Rust and returns an SVG command string), `TouchArea` for points |
| Three colour wheels | conic gradient for hue, a radial gradient for chroma, or a generated image; the puck is a `TouchArea` |
| Histogram | `Path` (our `histogramPath` already makes a path string) |
| Look grid, up to ~600 thumbnails, lazy | `ListView` instantiates only visible rows ("elements are only instantiated if they are visible, which guarantees stable performance with a practically unlimited number of items") [S23]; make each row hold N tiles and request the thumbnail from the tile's init callback; set the `Image` when the PNG is ready |
| Filmstrip, hundreds of 60 × 40 thumbnails, horizontal | the documented `ListView` is a vertical list; use a `Flickable` and a windowed model that follows `content-x`. Not verified that no horizontal virtual list exists |
| Tabular figures in ValueBox | `Text` has `letter-spacing` and `font-weight` but no OpenType feature switch [S23]; our CSS uses `font-variant-numeric: tabular-nums`. Workaround: digits in fixed-width cells, or a font build with tabular figures as default |
| Figtree | we ship variable `woff2` (`design/fonts/`); the Slint documentation I found shows `.ttf` imports; `woff2` and variable-axis support not verified, so plan static TTF instances |

### 2.12 Animation, start-up time, memory

- Animation: **works today** (`animate`, easing curves, a `spring` curve added in 1.18.0, animated Flickable) [S2]. Our UI has few animations (progress fills, update pill).
- Start-up and memory: **no data**. I found no published desktop numbers for Slint. 1.19 (unreleased) says system fonts are now found in the background for faster start-up [S2]. The Skia renderer has "heavy disk-footprint compared to other renderers" [S22]. The spike measures cold start to first frame, idle resident memory and binary size for the three renderers against the current Tauri build. A WKWebView app carries a separate web-content process; a Slint app does not; the photo pipeline's own budget (up to 4 GiB GPU, 6 GiB CPU, `docs/01`) dominates either way.

### 2.13 Testing

**Works with effort; coverage will be different, not equal.**
- `i-slint-backend-testing` simulates a windowing system with no pixels ("No pixels are rendered and text is measured by a fixed font size"), can advance mock time, and its `ElementHandle` clicks, hovers, drags and scrolls elements found by query (1.19 adds `ElementQuery::match_accessible_label`). Its README says it is internal, "does not follow the semver convention" and must be pinned as `=x.y.z` [S25]. 1.18.0 made those pointer helpers aim at the transformed centre of scaled or rotated elements [S2].
- Screenshots: `Window::take_snapshot()` returns an RGBA8 buffer [S25]; `slint-viewer --screenshot` exists [S2]. For pixel goldens on Linux CI use the software renderer. Text renders differently from WebKit, so `e2e/components.spec.ts` (pixel-exact against the mockups) becomes a tolerance comparison against `design/screenshots/`.
- An MCP server (feature `mcp`, started only when `SLINT_MCP_PORT` is set) lets an agent inspect and drive the running UI, including `dispatch_pointer_scroll` and modifier keys; "It is not recommended to enable it in production builds", and it binds to `127.0.0.1` [S2, S10]. It must stay out of release builds (hard rule 7).
- Our Playwright suite (20 files, 246 tests), `e2e/real/shell-smoke.mjs` (39 checks) and 1,131 Vitest cases would be replaced by Rust unit tests for the logic (math, state, undo, wizard model), the Slint testing API for interaction, and snapshots for looks.
- egui has `egui_kittest` (AccessKit based queries and snapshots) [S28]; GPUI has `TestAppContext` [S30].

### 2.14 Self-update

**Missing in Slint; works with effort, and our signing flow survives.** Slint has no updater. The current flow is `tauri-plugin-updater` reading `latest.json` and a minisign-signed `.app.tar.gz` (`docs/12`). Options:
1. **A small module of our own** (about 300 to 400 lines) with `minisign-verify` (0.3.0, MIT, 2026-09-25), `reqwest`, `tar`, `flate2` and `semver`: the same crates `tauri-plugin-updater` 2.14.0 depends on [S26]. Keeps `latest.json`, the `.app.tar.gz` and the existing key and signatures unchanged; installed 0.1.x copies should update into the new build, because the current updater replaces the whole bundle (to be confirmed in phase 5). Keeps our guarded `restart_app`. The public key in `tauri.conf.json` is base64 of the whole `.pub` file; `minisign-verify` needs it decoded first (one spike check, can be done on Linux).
2. `cargo-packager-updater` 0.2.3 (Apache-2.0 OR MIT, last release 2025-07-21): "Updater for apps that was packaged by `cargo-packager`", same endpoint and signature scheme as Tauri's, uses `minisign-verify` 0.2 [S26, S31]. Usable, but unreleased for a year.
3. **Sparkle 2**: Obj-C framework with an RSS appcast (`SUFeedURL`) and its own EdDSA keys (`SUPublicEDKey`, `generate_appcast`); it recommends Developer ID signing and notarisation "if possible" [S31]. We ship ad-hoc signed, and Sparkle's behaviour for that case is not verified. It would replace `latest.json` and the key. Not recommended.
Packaging also moves: the Tauri CLI makes the `.app`, `.dmg` and updater artefacts today. `cargo-packager` 0.11.8 (Apache-2.0 OR MIT, 2025-11-27) makes `.app` and `.dmg` and "a compatible updater", or a script (`lipo`, `codesign -s -`, `hdiutil`) does it [S31].

## 3. Size of the job

### 3.1 What exists (measured with `wc` on 2026-10-16)

| Part | Files | Lines |
|---|---|---|
| Rust crates (`crates/*`) | | 250,217 (ym-gpu 77.3k, ym-core 56.5k, ym-color 35.6k, ym-looks 18.9k, ym-raw 17.6k, ym-export 15.2k, ym-cli 11.7k, ym-noise 9.0k, ym-profiles 4.7k, ym-ml 3.7k) |
| `tools/` Rust | | 12,967 |
| Frontend TypeScript and TSX, no tests (`src/`) | | 18,652 |
| of which `src/api` (IPC types, generated bindings) | | 2,518 |
| of which `api.mock.ts` and `api.ts` | | 2,059 and 310 |
| of which `src/components` (34 exported components) | | 3,498 |
| of which `src/screens` | | 4,412 |
| of which `src/state` (`store.ts` alone 1,449) | | 2,223 |
| of which `src/export`, `src/wizard`, `updater.ts` | | 492, 617, 317 |
| of which `src/dev` (component gallery and screen replicas) | | 2,042 |
| CSS (`src/styles`) | 8 files | 2,184 |
| Vitest | 69 files, 1,131 cases | 13,931 |
| Playwright (`e2e/`) plus real-shell smoke | 20 files, 246 tests | 3,800 |
| `src-tauri/src` without tests | 17 files | about 9,400 (`wire.rs` 1,859, `viewport.rs` 1,467, `api.rs` 979, `commands.rs` 808, `menu.rs` 628, `native.rs` 617, `wizard.rs` 572, `cli_link.rs` 521, `system.rs` 380, `bindings.rs` 264, `thumbs.rs` 209, `logging.rs` 224, `events.rs` 158, `about.rs` 86) |
| `src-tauri` tests | | 2,412 (`api/tests.rs`) and 432 (`app_cli_parity.rs`); 108 `#[test]` |
| Mockups and references | 5 HTML pages (1,705 lines), 25 screenshots at 2880 × 1800 | |

Counts of what the UI contains (from `docs/11` and `src/`):
- **Screens:** 3 full-window (Start, Photos, Import ARRI looks wizard with 5 steps). 2 panels that replace the tool panel (Export with idle, running and finished states; Crop). 6 tool tabs with 14 cards (Look 3, Light 3, Colour 2, Detail 2, Curve 1, RAW 3).
- **Dialogs:** 4 (About, Preferences, Confirm close, Import .cube) plus the wizard's input question.
- **Overlays and popups:** pick-white crosshair, crop frame with handles, look chip, camera line, Before label, ⌥ pixel readout, zoom menu, update pill, error bar.
- **Components:** 34 (MoonMark, Button, KeyHint, LinkButton, ResetLink, AutoToggle, Tag, ProgressBar, ValueBox, Slider, Segmented, Switch, AutoRow, Card, Field, InfoRow, RowNote, Rows, SwitchRow, TitleBlock, TabBar, SearchField, LookTile, Wheel, WheelGroup, CurvePlot, Histogram, Filmstrip, PhotoCell, UpdatePill, DevelopList, ExportQueueList, DropZone, StepRail).

### 3.2 What carries over and what is rewritten

**Carries over unchanged:** all of `crates/*` and `tools/` (263k lines), `data/`, the CLI, the specs in `docs/` (apart from 00, 01, 11, 12, 14). `Core`, `Session`, `DevelopSettings`, the render graph, the export queue, preferences and looks stay as they are; the app would call `ym_core::app::Core` directly in-process.

**Carries over with changes (about 5.5k lines of `src-tauri`):** `viewport.rs` (the viewport controller and its `FrameSink` trait; only the sink changes), `app.rs`, the logic in `api.rs`, `wizard.rs`, `cli_link.rs`, `system.rs`, `about.rs`, `logging.rs`, and the menu `spec()` of `menu.rs`.

**Goes away (about 3.9k lines):** the IPC layer (`commands.rs`, `wire.rs`, `bindings.rs`, `events.rs`, `thumbs.rs`, the `ym-frame` scheme), `native.rs` (replaced by the Slint image path), the TypeScript contract tests that compare field names with `wire.rs`, `src/api/*`, `api.mock.ts`. Thumbnails and look tiles would go straight from the core to `slint::Image` without PNG bytes or an address scheme. The drift between `types.ts` and the Rust types disappears because there is one language.

**Rewritten:** all of `src/` (18.7k lines of TS and TSX, 2.2k of CSS), its tests (13.9k Vitest, 3.8k Playwright), and the packaging and update path. State handling shrinks: `store.ts` (1,449 lines) mostly mirrors the core across IPC; in-process it becomes a thin model over `Core`, plus the parts that really are UI state (undo of 100 steps, zoom mirror, picks, current tab, crop session, wizard model).

### 3.3 Phased plan and estimate (agent-days)

An agent-day here is the size of the days in this repository's history: 253 commits in five days (2026-10-12 to 2026-10-16) produced the Rust core, the frontend and the shell. Mac verification is not included.

| Phase | Work | Days | Exit check |
|---|---|---|---|
| 0 | (Not part of the rewrite.) Fix the web-page symptoms, profile the render path (section 4.1) | 0.5 to 1 | symptoms gone; three latency numbers |
| 1 | Spike (section 5) | 2 | gates G1 to G7 |
| 2 | Foundations: app crate with Slint window and event loop, `Core` wiring, state model, tokens as Slint globals, fonts, viewport component, menu, drop, dialog, updater, `AboutSlint` | 3 | Start opens files; Photos shows a photo with pan and zoom |
| 3 | Components: 34 `.slint` components with Rust helpers (curve, wheel, formats), a gallery and snapshot goldens | 3 | gallery matches `design/screenshots/` within tolerance |
| 4 | Screens: Start, Photos (header, cell, filmstrip, 6 tabs and 14 cards), Export, Crop, Pick white, wizard, dialogs, shortcuts, undo | 3.5 | `docs/11` behaviour checklist |
| 5 | Packaging and release: replace `tauri build`, universal binary, ad-hoc signing, dmg, `.app.tar.gz`, minisign signature, `latest.json`, cask, CI | 2 | an installed 0.1.x updates to the new build on a Mac |
| 6 | Tests, docs, clean-up: port the logic tests to Rust, Slint-API interaction tests, snapshots; delete React, Tauri and Playwright; update `docs/00`, `01`, `11`, `12`, `14`, `NOTICE`, `THIRD-PARTY.md` | 2.5 | CI green on Linux and macOS |
| | Total phases 1 to 6 | **16** (range 11 to 25) | |

Low end: the spike is clean and components map one to one. High end: colour needs the child-window route, or scrolling and the menu need custom native work. Phases 3 and 4 can be split between two or three agents (about two calendar weeks). Budget 5 to 8 Mac round trips for trackpad feel, colour on a P3 and an sRGB display, VoiceOver, the menu, the title bar and the update. The 0.1.0 to 0.1.1 fix of the blank viewport (`docs/STATUS.md`) shows what one costs; Linux CI cannot run Metal, a trackpad or a P3 display.

### 3.4 Risks specific to the rewrite

1. The licence decision (section 1) has to be made before the work starts.
2. Colour, wgpu interop and scroll feel (sections 2.1, 2.2, 2.4) can each fail late.
3. The rewrite may not fix "sluggish": the frame path stays (section 4.1). The Mac numbers for it are still unmeasured (M0-H4 is waiting on you).
4. Loss of free web features: DOM accessibility, text services, DevTools-style inspection, Playwright's maturity.
5. Slint's own churn: wgpu flag renamed every minor release, the testing crate is not semver, Flickable and AccessKit issues still being fixed in 1.19.
6. Skia builds fetch prebuilt binaries and need a large target directory; the disk here is at 88%.
7. Our hard rule 7 (no network): the Slint MCP server and `system-testing` feature must never be in a release build.

## 4. Alternatives

### 4.1 Keep Tauri and fix responsiveness

What the repository shows (all checked in the files, none run on a Mac):

| Symptom | What is in the code | Likely fix |
|---|---|---|
| Text is selectable | `src/styles/global.css` has no `user-select` rule on `body`; `user-select: none` appears on 9 specific rules (slider, wheel, curve, photo image, crop) only | `body { -webkit-user-select: none; user-select: none; cursor: default }` and `input, textarea { -webkit-user-select: text; user-select: text }` |
| Two-finger pan moves the whole app | No `overscroll-behavior` anywhere in `src/` or `index.html`; `.ym-screen` is `height: 100vh; overflow: hidden`, but `html` and `body` have no overflow rule. The wheel handler is React's `onWheel` (`src/screens/PhotoView.tsx`, line 169), and at Fit it does nothing, so the event reaches the document. React keeps `onWheel` passive: "`e.preventDefault()` in them stays broken" (React PR 19654) [S32] | `html, body { overflow: hidden; overscroll-behavior: none }` (Safari 16 and later according to caniuse, one of whose tables marks it partial; which WebKit a Mac on macOS 12 runs depends on the Safari update installed, not verified) [S33]; a native `addEventListener("wheel", ..., { passive: false })` that calls `preventDefault()` on the window except inside the tool panel and the filmstrip |

I could not confirm that CSS alone stops elastic scrolling in WKWebView on macOS; search results disagree and mostly concern iOS. The wheel listener is the sure part.

"Sluggish" probably has other causes, in this order of suspicion:
1. **Every input is a frame render.** A wheel or drag event goes over IPC to the viewport worker (`src-tauri/src/viewport.rs`), which asks the core for a new RGBA8 frame at the size of the cell, uploads it, and presents it. `docs/01` says the same ("M0's spike measures latency; if it is above one frame, move pan/zoom handling to native event monitoring"). Measured on Linux (4 shared cores), a warm fit frame took 849 ms on the CPU path (output sharpening, which still runs on the CPU, accounts for 306 of those) and 174 ms with the GPU colour pass on llvmpipe before that stage existed; Mac numbers are pending (`docs/01` Preview strategy; M0-H4).
2. **Every slider move is a full frame.** "The app has no way to tell a drag from a click on `update_settings`, so there is no cheaper frame during a drag" (`docs/01`).
3. React commit time on the 400 px panel for each `render_stats` and settings event (`store.ts` is 1,449 lines). Unmeasured.
4. A transparent full-window WKWebView over a Metal layer (`transparent: true`, `macOSPrivateApi`). Unmeasured.

None of 1 and 2 is changed by a new toolkit. Cheap tests first: stage timings with `YELLOWMOON_LOG=debug`, Safari's Web Inspector timeline on the WKWebView, and three numbers (pan input to frame, slider input to frame, React commit per slider move).

If the measurements point at 1 and 2, the fixes are in our own code and work with any toolkit: pan and zoom as a shader transform of a cached frame (cheap frame while dragging, sharp frame after), and input over the photo handled by a native view above the webview (`scrollWheel:`, `magnifyWithEvent:`, `smartMagnifyWithEvent:` straight into `Viewport`, no IPC), which also stops the webview from ever seeing those events [S27]. Estimate: page fixes and measuring 0.5 to 1 day; native input view 1.5 to 2; GPU pan and zoom of a cached frame 2 to 3 (my guesses, Mac needed for the first two).

Also: `docs/00` lists the shell (Tauri 2) among the settled decisions ("Don't reopen these without new evidence"). This report is that evidence if you choose to reopen it.

### 4.2 Native AppKit through `objc2`

- Licence: `objc2` 0.6.5 (MIT, 2026-10-05), `objc2-app-kit` 0.3.2 (Zlib OR Apache-2.0 OR MIT, 2025-10-04) [S26]. Clean. `src-tauri/src/native.rs` already uses both.
- macOS maturity: everything is first class. NSEvent `magnifyWithEvent:` and `smartMagnifyWithEvent:` [S27], `scrollWheel:` with phases, `NSDraggingDestination`, `NSOpenPanel`, `NSMenu`, NSAccessibility, `NSTextInputClient`, and `CAMetalLayer.colorspace` [S14]; Sparkle fits.
- wgpu and Metal: the viewport we have is this approach already; a `CAMetalLayer` per view can carry its own colour space.
- Trackpad: best possible.
- Cost: no declarative UI. The "bold grid" would be custom views (34 components) drawn by hand in `objc2`, which is unsafe-heavy, or the same work in Swift/SwiftUI with the Rust core behind a C interface (a language break; every UI line then compiles only on macOS, so Linux CI cannot build it). My estimate 25 to 40 agent-days. Best result for macOS feel, worst for the agent workflow and for Linux CI.

### 4.3 GPUI (Zed's toolkit)

- Licence: `gpui` 0.2.2 and its helper crates (`gpui_util`, `gpui_collections` and others) are Apache-2.0 on crates.io (2025-10-22) [S26, S30]; `gpui_macos` is Apache-2.0 too. A fully clean licence. The crates.io version is a year old; the Zed `main` branch has split into `gpui_platform`, `gpui_macos`, `gpui_apple`, `gpui_wgpu`, so a current build is a git dependency, which `deny.toml` forbids (`unknown-git = "deny"`).
- Maturity: the README says "GPUI is still in active development as we work on the Zed code editor, and is still pre-1.0. There will often be breaking changes between versions" and "the best way to learn about these APIs is to read the Zed source code" [S30].
- macOS: "Rendering uses Metal and is always available" [S30]; `BGRA8Unorm` layer with no colour space set [S30]. Accessibility through AccessKit is wired for macOS (`gpui_macos` depends on `accesskit_macos`) [S30].
- wgpu interop: on macOS GPUI draws with Metal, not wgpu. The route for our pixels is its `surface(CVPixelBuffer)` element (an IOSurface-backed picture) [S30]; we would have to create our wgpu texture on an IOSurface through `wgpu-hal`. Possible, low level.
- Trackpad: `ScrollWheelEvent` with `TouchPhase` and a `PinchEvent` exist [S30]; I found no smart-zoom handler.
- Verdict: clean licence, strong macOS feel, unstable API and no wgpu sharing. Not now.

### 4.4 egui (eframe, egui-wgpu)

- Licence: `egui` 0.37.0 (MIT OR Apache-2.0, 2026-10-15) [S26]. Clean.
- Maturity: the README says "If you want a GUI that looks native, egui is not for you. If you want something that doesn't break when you upgrade it, egui isn't for you (yet)", and also "egui can be used to create professional looking applications, like the Rerun Viewer" [S28]. Immediate mode.
- wgpu: egui 0.37 is on wgpu 30 (its workspace comment: "wgpu 30's wgpu-hal pins glow 0.17") [S28]; `egui_wgpu::Callback` runs our own wgpu code inside egui's render pass on egui's device. The closest fit to a custom GPU viewport; sharing `ym-gpu`'s device would need the same API work as for Slint. No colour-space setting found in `egui-wgpu` [S28], so the layer-tag fix of section 2.2 applies.
- Trackpad and input: `egui-winit` handles `MouseWheel` (source `Trackpad`), `PinchGesture`, `HoveredFile` and `DroppedFile`, and Ime; it does not act on `DoubleTapGesture` [S28]. Text selection of labels is a setting.
- Accessibility: AccessKit, "currently implements the native accessibility APIs on Windows and macOS", on by default in eframe; tests with `egui_kittest` [S28].
- Menu: none in eframe; `muda` separately. Look: "bold grid" is easy with custom painting; curve editor, wheels and histogram are natural in a `Painter`.
- Verdict: the only candidate that is licence-clean, already on wgpu 30, and quick for custom widgets. The price is pre-1.0 churn and a look that is ours to build. My estimate 12 to 20 agent-days. Run the section 5 spike on it if Slint's licence is refused.

### 4.5 iced

- Licence: `iced` 0.14.0 (MIT, 2025-12-07) [S26]. Maturity: "Iced is currently experimental software." [S29] Master (0.15.0-dev) is on wgpu 30 (I did not check which wgpu the released 0.14.0 uses), and master pins a git fork of winit (`deny.toml` forbids git sources) [S29]. `iced::widget::shader` renders custom wgpu pipelines inside the widget tree [S29]. No AccessKit dependency in the workspace manifest [S29], so VoiceOver support is not there. Not suitable.

### 4.6 Freya, Dioxus native and Blitz, Xilem

- **Freya** 0.4.3 (MIT, 2026-10-11) [S26, S31]: Skia through `freya-skia-safe`, OpenGL (`gl`, `glutin-winit`) in the workspace manifest, winit 0.30, AccessKit. The 0.5 line is at release candidate. I did not verify its Metal path or trackpad handling.
- **Dioxus native and Blitz** (`dioxus-native` 0.7.10, `blitz-dom` 0.2.4, MIT OR Apache-2.0, October 2026) [S26]: Blitz says "currently in a **beta** state ... usable for making apps if you are an early adopter and willing to live on the bleeding edge". It uses Vello on wgpu 30 with `winit =0.31.0-beta.3`, has a "WGPU texture integration example" and plans AccessKit [S31]. It is HTML and CSS rendered natively (our CSS could carry over in part), but written as Dioxus (Rust), not React. Too young.
- **Xilem and Masonry** (`xilem` 0.4.0, Apache-2.0, 2025-10-29): "An experimental Rust architecture for reactive UI", on winit, Vello, wgpu, Parley and AccessKit [S31]. Experimental. Not suitable yet.

### 4.7 Comparison

| | Licence | macOS maturity | wgpu 30 sharing | Trackpad | A11y | Verdict |
|---|---|---|---|---|---|---|
| Keep Tauri, fix | Apache-2.0 or MIT | in use | own device, one copy | page fixes plus native input view | DOM | **do first** |
| Slint 1.18.1 | GPL / Royalty-free / paid | usable, gaps (menu roles, scroll feel, colour) | `unstable-wgpu-30`, RGBA8 only | pinch yes; smart zoom and drops by hook | AccessKit, 1 perf issue | only with a licence exception, after a spike |
| AppKit (`objc2` or Swift) | MIT / Zlib / Apache | best | n/a (own layer) | best | native | best feel, largest job |
| GPUI | Apache-2.0 | strong, pre-1.0 | no (Metal; IOSurface route) | scroll and pinch | AccessKit | not now |
| egui 0.37 | MIT or Apache-2.0 | adequate, pre-1.0 | yes (wgpu 30 callbacks) | scroll, pinch, drops | AccessKit | fallback if Slint's licence is refused |
| iced 0.14 | MIT | experimental | custom shader widget | not checked | none found | no |
| Freya, Blitz, Xilem | MIT / Apache | young | partial | not checked | AccessKit | no |

## 5. Recommendation and spike plan

### 5.1 Recommendation

1. Do phase 0 now (section 4.1): the two page fixes, the non-passive wheel listener, and the three latency numbers. About a day, and it settles whether the toolkit is the problem.
2. If, after that, you still want to leave the webview, decide the licence question first (section 1.5). Then run the spike below. Slint is a reasonable candidate for the look and the in-process design, but it is not ahead of the alternatives on macOS feel, and its licence is the only one in this list that is neither permissive nor GPL-clean.
3. Do not start the rewrite before the spike's gates G1 to G3 pass.

### 5.2 Spike plan, two days, on a Mac with a trackpad

Pin `slint = "=1.18.1"` with features `unstable-wgpu-30`, `unstable-winit-030`, `renderer-skia`. A throwaway crate outside the workspace, so nothing in `Cargo.lock` or `deny.toml` changes.

**Day 1: the riskiest points.**

| Gate | Test | Pass |
|---|---|---|
| G1 colour | Show a Display P3 test chart (P3 primaries, sRGB primaries, a grey ramp) through `Image::try_from(texture)` at 1:1 device pixels, on a built-in P3 display and an external sRGB display. Capture with `screencapture` (it embeds the display profile) and compare in Lab with the same chart in the current Tauri viewport and in Preview. Try three variants: untouched; layer tagged P3 with `objc2` and re-applied after each window resize; viewport in sRGB. Also compare Slint's UI yellow `#FFC20E` against Preview's | one variant within 1.0 ΔE00 on both displays, UI colours included |
| G2 interop | Pass `ym-gpu`'s device to Slint with `WGPUConfiguration::Manual` (or give Slint features and limits with `Automatic` and hand its device to `ym-gpu`). Render real frames, 2880 × 1920, from a texture with no CPU readback, 60 per second. Try Skia and FemtoVG. Pan, zoom and resize for 10 minutes | no flashes or tearing (issue 12271 class); no unexpected `ym-gpu` limit failure; frame time logged |
| G3 input | Over the photo: `TouchArea.scroll-event` pan with momentum continuing after the fingers lift; `ScaleRotateGestureHandler` pinch about the cursor; `DoubleTapGesture` through `on_winit_window_event` toggling Fit and 100%; labels not selectable; no rubber-band. Log scroll-event timestamps against presented frames; compare with the same gestures in the current app and Preview | pan input to photo within one frame plus render time; momentum feels like Preview |
| G6 footprint | Cold start to first frame, idle resident memory and binary size for Skia, FemtoVG-wgpu and software builds, against the current Tauri build | recorded; no gate unless 2× worse |
| G7 updater (Linux, 1 hour) | `minisign-verify` accepts a `.sig` made by the current Tauri signer for a real `latest.json` entry, using the key in `tauri.conf.json` | verifies |

**Day 2: shell and widgets.**

| Gate | Test | Pass |
|---|---|---|
| G4 shell | Overlay title bar through `with_winit_window_attributes_hook` (transparent title bar, hidden title, full-size content view) with `WindowMoveArea` and 84 px padding; Finder drop of 3 files and 1 folder through `HoveredFile` and `DroppedFile` with a hover state; `rfd` open dialogs with the Slint window as parent; `MenuBar` with shortcuts, and an app menu with Preferences…, Check for Updates… and Install Command-Line Tool… either through `with_default_menu_bar(false)` and `muda` or in another menu; `AboutSlint` in an in-window About | each works, or its workaround is under a day |
| G5 widgets and a11y | A slider with a detent and a typed ValueBox; a 600-tile `ListView` with lazily set images, scrolled at 60 fps with `accessibility` on (issue 3867); VoiceOver reads the tab list, a slider's value, and `increment` moves the slider | 60 fps; VoiceOver usable |

**Decision rule.** If G1, G2 or G3 fails and the fallback needs more than a day, stop and write it up. G4 and G5 failures are costed, not fatal. If you refuse the licence exception, run the same table on egui (`eframe` with `egui-wgpu`, wgpu 30), replacing the Slint-specific calls (`Image::try_from` becomes an `egui_wgpu::Callback`; the winit hook is `egui-winit` itself).

## 6. Questions for you

1. Is a free, source-available, non-OSI licence (Slint Royalty-free 2.0: attribution, no embedded use) an acceptable exception to AGENTS.md rule 1 and `deny.toml`? If no, the spike goes to egui.
2. Is it acceptable that every fork of Yellowmoon that keeps Slint has to keep `AboutSlint` or the badge, and that an embedded fork cannot use the Royalty-free licence?
3. By "sluggish": pan and zoom, slider drags, switching photos, or opening a folder? It decides which of the four suspects in section 4.1 to measure first.
4. May the viewport be sRGB-only if Slint cannot tag P3 (the default export is sRGB), or is P3 in the viewport a requirement?
5. Which Mac and displays (a P3 laptop panel and an sRGB external, a trackpad) can run the spike, or may it be queued under "Waiting on Mykyta" as M0-H4 is?
6. If the rewrite goes ahead, may the Tauri bundler be replaced by `cargo-packager` or a script (section 2.14)?

### Answers (Mykyta, 2026-10-16)

1 and 2: no exception: "my software is Apache 2", so Slint's Royalty-free licence is not used and a spike, if any, runs on a licence-clean toolkit (egui). 4: an sRGB-only viewport is acceptable. 5: the owner's M4 Mac ("Mac Pro M4"; read as a MacBook Pro with its built-in display until he says otherwise). 3 and 6 are open. Recorded in `docs/16`.

## Sources

All read 2026-10-16. Dates are of the release or commit read, or "master" for files read from the default branch that day.

- S1 Slint releases, https://github.com/slint-ui/slint/releases (1.18.1 on 2026-09-21; 1.18.0 on 2026-09-16; 1.17.1 on 2026-07-07; 1.17.0 on 2026-06-24; 1.16.x on 2026-04-16 and 04-23; 1.15.x on 2026-02-04 and 02-12).
- S2 Slint changelog, https://raw.githubusercontent.com/slint-ui/slint/master/CHANGELOG.md (master, with 1.19.0 unreleased; per-release sections 1.12.0 to 1.18.1).
- S3 Slint `LICENSE.md`, https://raw.githubusercontent.com/slint-ui/slint/master/LICENSE.md (master).
- S4 Slint Royalty-free licence 2.0, https://raw.githubusercontent.com/slint-ui/slint/master/LICENSES/LicenseRef-Slint-Royalty-free-2.0.md (master).
- S5 Slint Software License 3.0.5, https://raw.githubusercontent.com/slint-ui/slint/master/LICENSES/LicenseRef-Slint-Software-3.0.md (master).
- S6 Slint licensing FAQ, https://raw.githubusercontent.com/slint-ui/slint/master/FAQ.md (master).
- S7 `slint` on crates.io, https://crates.io/crates/slint (1.18.1, 2026-09-21; licence expression read through the crates.io API).
- S8 Apache Software Foundation, "Apache License v2.0 and GPL Compatibility", https://www.apache.org/licenses/GPL-compatibility.html.
- S9 Open Source Definition, https://opensource.org/osd (clause 6).
- S10 Slint Rust documentation: https://docs.slint.dev/latest/docs/rust/slint/wgpu_30/ , `.../wgpu_30/enum.WGPUConfiguration.html` , `.../wgpu_30/enum.TextureImportError.html` , `.../struct.BackendSelector.html` , `.../docs/cargo_features/` , `.../winit_030/` (the "latest" documentation, which also says it is for the next version).
- S11 Slint source, master: https://raw.githubusercontent.com/slint-ui/slint/master/internal/core/graphics/wgpu_30.rs , `.../internal/renderers/skia/wgpu_30_surface.rs` , `.../internal/renderers/femtovg/wgpu.rs`.
- S12 wgpu-hal Metal surface, https://raw.githubusercontent.com/gfx-rs/wgpu/v30.0.1/wgpu-hal/src/metal/surface.rs (v30.0.1).
- S13 wgpu `SurfaceColorSpace`, https://wgpu.rs/doc/wgpu/enum.SurfaceColorSpace.html .
- S14 Apple, `CAMetalLayer.colorspace`, https://developer.apple.com/documentation/quartzcore/cametallayer/colorspace .
- S15 Slint issue 12271 "V1.17 - Skia Renderer + Pink Flash On Opened Windows" (2026-06-27, closed 2026-07-02), https://github.com/slint-ui/slint/issues/12271 .
- S16 Slint `ScaleRotateGestureHandler`, https://docs.slint.dev/latest/docs/slint/reference/gestures/scalerotategesturehandler/ .
- S17 Slint `TouchArea`, https://docs.slint.dev/latest/docs/slint/reference/gestures/toucharea/ .
- S18 winit 0.30.12: https://docs.rs/winit/0.30.12/winit/event/enum.WindowEvent.html ; macOS view source https://raw.githubusercontent.com/rust-windowing/winit/v0.30.12/src/platform_impl/macos/view.rs (`scroll_wheel`, `magnify_with_event`, `smart_magnify_with_event`).
- S19 winit macOS window attributes, https://docs.rs/winit/0.30.12/winit/platform/macos/trait.WindowAttributesExtMacOS.html .
- S20 Slint `Window`, https://docs.slint.dev/latest/docs/slint/reference/window/window/ ; `WindowMoveArea`, https://docs.slint.dev/latest/docs/slint/reference/window/windowmovearea/ .
- S21 Slint winit adapter, https://raw.githubusercontent.com/slint-ui/slint/master/internal/backends/winit/winitwindowadapter.rs (master: `MouseWheel`, `PinchGesture`, `safe_area_inset` for iOS only, no file-drop arms).
- S22 Slint backends: https://docs.slint.dev/latest/docs/slint/guide/backends-and-renderers/backends_and_renderers/ ; `https://raw.githubusercontent.com/slint-ui/slint/master/internal/backends/winit/muda.rs` ; `.../internal/backends/winit/lib.rs` ; `.../internal/backends/selector/api.rs` (master).
- S23 Slint element and widget documentation: `ListView` https://docs.slint.dev/latest/docs/slint/reference/std-widgets/views/listview/ ; `AboutSlint` https://docs.slint.dev/latest/docs/slint/reference/std-widgets/misc/aboutslint/ ; `Text` https://docs.slint.dev/latest/docs/slint/reference/elements/text/ ; `Image` https://docs.slint.dev/latest/docs/slint/reference/elements/image/ ; common properties and `accessible-*` https://docs.slint.dev/latest/docs/slint/reference/common/ ; numeric types (`phx`) https://docs.slint.dev/latest/docs/slint/reference/property-types/numeric-types/ .
- S24 Slint issues: 9803 https://github.com/slint-ui/slint/issues/9803 ; 3867 https://github.com/slint-ui/slint/issues/3867 ; pull request 13542 https://github.com/slint-ui/slint/issues/13542 ; open-issue lists (10451, 12030) https://github.com/slint-ui/slint/issues?q=is%3Aissue+is%3Aopen+macOS+wgpu and `...+trackpad+OR+touchpad+OR+scroll+macOS` (titles only).
- S25 Slint testing: https://raw.githubusercontent.com/slint-ui/slint/master/internal/backends/testing/README.md ; `Window::take_snapshot` https://docs.slint.dev/latest/docs/rust/slint/struct.Window.html .
- S26 crates.io (API, 2026-10-16): `slint` 1.18.1; `wgpu` 30.0.1 (2026-08-22); `winit` 0.30.13 (2026-09-04); `egui` 0.37.0 (2026-10-15); `iced` 0.14.0 (2025-12-07); `freya` 0.4.3 (2026-10-11); `dioxus-native` 0.7.10 (2026-10-14); `blitz-dom` 0.2.4 (2026-10-12); `xilem` 0.4.0 (2025-10-29); `gpui` 0.2.2 (2025-10-22) and its dependencies `gpui_*`; `tauri` 2.12.4; `tauri-plugin-updater` 2.14.0 (2026-10-14, dependencies include `minisign-verify`, `tar`, `flate2`, `reqwest`, `semver`); `tauri-plugin-dialog` 2.8.3 (depends on `rfd`); `rfd` 0.17.2 (MIT, 2026-01-12); `muda` 0.21.2 (2026-10-08); `minisign-verify` 0.3.0 (MIT, 2026-09-25); `cargo-packager` 0.11.8 (2025-11-27); `cargo-packager-updater` 0.2.3 (2025-07-21); `objc2` 0.6.5; `objc2-app-kit` 0.3.2; `accesskit` 0.25.1; `accesskit_macos` 0.27.1 (2026-09-25).
- S27 Apple, `smartMagnify(with:)`, https://developer.apple.com/documentation/appkit/nsresponder/smartmagnify(with:) ; `magnify(with:)` https://developer.apple.com/documentation/appkit/nsresponder/magnify(with:) .
- S28 egui: https://raw.githubusercontent.com/emilk/egui/main/README.md ; `crates/egui-winit/src/lib.rs` ; `crates/egui-wgpu/src/renderer.rs` ; workspace `Cargo.toml` ; `crates/egui_kittest/README.md` (main).
- S29 iced: https://raw.githubusercontent.com/iced-rs/iced/master/README.md ; `Cargo.toml` ; `widget/src/shader.rs` (master).
- S30 GPUI: https://raw.githubusercontent.com/zed-industries/zed/main/crates/gpui/README.md ; `crates/gpui/Cargo.toml` ; `crates/gpui/src/interactive.rs` ; `crates/gpui/src/elements/surface.rs` ; `crates/gpui/src/_accessibility.rs` ; `crates/gpui_apple/src/metal_renderer.rs` ; `crates/gpui_macos/Cargo.toml` (main).
- S31 Freya https://raw.githubusercontent.com/marc2332/freya/main/README.md and `Cargo.toml`; Blitz https://raw.githubusercontent.com/DioxusLabs/blitz/main/README.md and `Cargo.toml`; Xilem https://raw.githubusercontent.com/linebender/xilem/main/README.md; `cargo-packager` https://raw.githubusercontent.com/crabnebula-dev/cargo-packager/main/README.md and `crates/updater/README.md`; Sparkle documentation https://sparkle-project.org/documentation/ (main).
- S32 React pull request 19654, "Keep onTouchStart, onTouchMove, and onWheel passive", https://github.com/facebook/react/pull/19654 .
- S33 caniuse, CSS `overscroll-behavior`, https://caniuse.com/css-overscroll-behavior (from a search result; the page itself was not opened).

Repository files read: `AGENTS.md`, `docs/00-overview.md`, `docs/01-architecture.md`, `docs/11-ui.md`, `docs/12-distribution.md`, `docs/14-licensing.md`, `docs/gpu-plan.md`, `docs/09-export.md`, `docs/16-risks-and-decisions.md`, `deny.toml`, `src-tauri/tauri.conf.json`, `src-tauri/Cargo.toml`, `src-tauri/src/native.rs`, `src-tauri/src/viewport.rs`, `crates/ym-core/src/pipeline/frame.rs`, `src/styles/*.css`, `src/screens/PhotoView.tsx`, `index.html`, `e2e/a11y.spec.ts`.

Scratch: downloaded sources and the helper scripts are in `/tmp/claude-0/-home-user-yellowmoon/21063226-7ac2-5ae0-bec4-1d31a6166c2b/scratchpad/` (`slint-cl/`, `lic/`, `slintsrc/`, `web/`, `alts/`, `gpui/`, `wgpusrc/`, `winitsrc/`, `apple/`).

## What I could not check

- Anything that needs a Mac: colour on a P3 panel, trackpad feel, momentum, VoiceOver, the menu, the title bar, Finder drops, start-up time and memory. Each is a gate in section 5.
- Whether Slint's `with_default_menu_bar(false)` is reachable from the public `slint` crate; whether winit's momentum events reach Slint's `scroll-event` unchanged (the winit source says they are sent as scroll events); font format and variable-font support in Slint; whether a transparent Slint window can sit above our own Metal view.
- Whether CSS `overscroll-behavior` alone stops elastic scrolling inside WKWebView on macOS.
- The `tauri` menu implementation, Freya's Metal path, trackpad handling in iced, Freya, Blitz and Xilem (marked "not checked").
- Legal effect of the licences for an Apache-2.0 project; Slint's contact for licensing questions is `info@slint.dev` [S3].
