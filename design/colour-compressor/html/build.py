#!/usr/bin/env python3
"""Builds the pages of design/colour-compressor/html/pages/ (task CC-01; the owner reviews them at CC-H1).

Same method as design/skin/html/build.py and design/mixers/html/build.py: the app's DOM and CSS captured in design/nodes/html/pages/ (header, photo cell,
toolbar, panel head, tabs, cards, sliders), the graph and the panel rebuilt from the app's own classes. This file imports the skin builder (which imports the
mixers builder) for the shared pieces and adds the Colour Compressor node (kind `color_compressor`, docs/32). The pictures are the drawing of design/skin/
with the maths of docs/32 section 3 applied by compress_ref.py (compress_demo.py writes portrait-before/after/affected.jpg and assets/target.json).
Run: python3 build.py   (needs bs4, numpy). Not product code.
"""
import importlib.util
import json
import math
import os
import re
import sys

from bs4 import BeautifulSoup

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "skin", "html"))
sys.path.insert(0, HERE)
spec = importlib.util.spec_from_file_location("skb", os.path.join(HERE, "..", "..", "skin", "html", "build.py"))
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
M = S.M
svg = M.svg
import compress_ref as R  # noqa: E402

AS = "../assets/"
T = json.load(open(os.path.join(HERE, "assets", "target.json")))
RGB = "rgb(%d,%d,%d)" % tuple(T["rgb"])

# the node's glyph: two arrows pointing at a line (a range pulled onto one colour)
CC_ICON = '<path d="M2 12h6"/><path d="m5 9 3 3-3 3"/><path d="M22 12h-6"/><path d="m19 9-3 3 3 3"/><path d="M12 4v16" stroke-dasharray="2 3"/>'
EYE = S.EYE
M.ICON["color_compressor"] = CC_ICON
INFO = '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>'
PICK_ICON = '<path d="m2 22 1-1h3l9-9"/><path d="M3 21v-3l9-9"/><path d="m15 6 3.4-3.4a2.1 2.1 0 1 1 3 3L18 9l.4.4a2.1 2.1 0 1 1-3 3l-3.8-3.8a2.1 2.1 0 1 1 3-3l.4.4Z"/>'
DROP = '<path d="M12 22a7 7 0 0 0 7-7c0-2-1-3.9-3-5.5s-3.5-4-4-6.5c-.5 2.5-2 4.9-4 6.5C6 11.1 5 13 5 15a7 7 0 0 0 7 7z"/>'


# ------------------------------------------------------------------------------------------------------------------------------ rows
_uid = [100]


def uid():
    _uid[0] += 1
    return f"cc{_uid[0]}"


def srow(label, value, lo, hi, default, text, disabled=False, bipolar=False, track=None):
    """The app's slider row (label, slider, number field, Reset when changed); `text` is what the number field reads."""
    r = BeautifulSoup(str(S.SLIDER), "html.parser").select_one("[data-slider]")
    i = uid()
    lab = r.select_one("span.text-\\[0\\.8125rem\\]")
    lab["id"] = i
    lab.string = label
    f = (value - lo) / (hi - lo)
    rng = r.select_one('[data-slot="slider-range"]')
    if track == "hue":
        rng["style"] = "inset-inline-start: 0%; width: 0%;"
        r.select_one('[data-slot="slider-track"]')["class"] = r.select_one('[data-slot="slider-track"]').get("class", []) + ["cc-hue-track"]
    elif bipolar:
        z = (0 - lo) / (hi - lo)
        a, b = min(f, z), max(f, z)
        rng["style"] = f"inset-inline-start: {a * 100:.2f}%; width: {(b - a) * 100:.2f}%;"
    else:
        rng["style"] = f"inset-inline-start: 0%; width: {f * 100:.2f}%;"
    thumb = r.select_one('[data-slot="slider-thumb"]')
    thumb.parent["style"] = f"transform: var(--radix-slider-thumb-transform); position: absolute; left: calc({f * 100:.2f}% + {8 - 16 * f:.2f}px);"
    thumb["aria-labelledby"] = i
    thumb["aria-valuemin"], thumb["aria-valuemax"], thumb["aria-valuenow"] = str(lo), str(hi), str(value)
    thumb["aria-valuetext"] = text
    inp = r.select_one("input")
    inp["value"] = text
    inp["aria-label"] = f"{label} value"
    if value != default and not disabled:
        inp["data-changed"] = ""
        last = r.select_one("div.flex.justify-end")
        btn = BeautifulSoup(str(S.RESET_LINK), "html.parser").find("button")
        btn["aria-label"] = f"Reset {label.lower()}"
        last.append(btn)
    dis = ' aria-disabled="true"' if disabled else ""
    return f'<div class="sk-row"{dis}>{r}</div>'


def head_of(title, tag=None, extra="", reset=None):
    t = f'<span class="sk-count">{tag}</span>' if tag else ""
    return f'<div class="ym-card__header"><h3 class="ym-card__title">{title}</h3>{t}{extra}{reset or ""}</div>'


def pick_button(on):
    """Pick colour: glasscn's Toggle (a header-variant button with the eyedropper), pressed while the crosshair is on."""
    b = BeautifulSoup(M.FOOT, "html.parser").find("button")
    b.clear()
    b.append(BeautifulSoup(svg(PICK_ICON, 16), "html.parser"))
    b.append(BeautifulSoup("<span>Pick colour</span>", "html.parser"))
    b["aria-pressed"] = "true" if on else "false"
    b["aria-describedby"] = ""
    b["data-state"] = "on" if on else "off"
    if on:
        b["class"] = b.get("class", []) + ["cc-pick-on"]
    if b.has_attr("aria-disabled"):
        del b["aria-disabled"]
    return str(b)


def eye(on, disabled=False):
    e = BeautifulSoup(str(S.EYE_BTN), "html.parser")
    b = e.find("button")
    b["aria-label"] = "Show affected area"
    b["aria-pressed"] = "true" if on else "false"
    b["data-state"] = "on" if on else "off"
    e.select_one(".sr-only").string = "Pick a colour first." if disabled else ("Hide the affected area" if on else "Show affected area")
    if on:
        b["class"] = b.get("class", []) + ["sk-eye-on"]
    if disabled:
        b["aria-disabled"] = "true"
    return str(e)


def colour_name(h, s):
    names = [(0, "Red"), (30, "Orange"), (60, "Yellow"), (120, "Green"), (180, "Cyan"), (240, "Blue"), (270, "Purple"), (300, "Magenta"), (360, "Red")]
    n = min(names, key=lambda a: abs(((h - a[0] + 180) % 360) - 180))[1]
    return n + (", a little pale" if s < 14 else "")


# ------------------------------------------------------------------------------------------------------------------------------ cards
def target_card(state="set", hue=None, sat=None, ev=None, note=None, rgb=RGB):
    """state: empty (no target, Pick colour on), picking (a target, Pick colour on), set, grey (a notice)."""
    hue = T["hue_deg"] if hue is None else hue
    sat = T["saturation"] if sat is None else sat
    ev = T["lightness_ev"] if ev is None else ev
    has = state != "empty"
    on = state in ("empty", "picking")
    if has:
        sw = f'<span class="cc-swatch" style="background:{rgb}" role="img" aria-label="Target colour: {colour_name(hue, sat).lower()}, hue {hue:g} degrees, saturation {sat:g}, lightness {ev:g} stops."></span>'
        name = f'<span class="cc-swatch__name">{colour_name(hue, sat)}</span><span class="cc-swatch__sub">Picked from the photo</span>'
    else:
        sw = '<span class="cc-swatch cc-swatch--empty" role="img" aria-label="No target colour"></span>'
        name = '<span class="cc-swatch__name">No colour yet</span><span class="cc-swatch__sub">Click a colour in the photo.</span>'
    reset = S.link_button("Reset") if has else ""
    top = f'<div class="cc-target">{sw}<div class="cc-swatch__text">{name}</div>{pick_button(on)}</div>'
    n = f'<div class="sk-note" role="note">{svg(INFO, 16)}<p>{note}</p></div>' if note else ""
    dis = not has
    rows = (
        srow("Hue", round(hue), 0, 359, round(hue), f"{round(hue)}°" if has else "–", disabled=dis, track="hue")
        + srow("Saturation", sat, 2, 50, sat, f"{sat:g}" if has else "–", disabled=dis)
        + srow("Lightness (EV)", ev, -4, 8, ev, (f"{ev:+.2f}".replace("-", "−")) if has else "–", disabled=dis, bipolar=True)
    )
    d = ' data-disabled' if dis else ""
    return f'<section class="ym-card" data-last="false">{head_of("Target", None, "", reset)}{top}{n}<div class="sk-rows"{d}>{rows}</div></section>'


def compress_card(h=60, s=60, l=0, disabled=False):
    rows = srow("Hue", h, 0, 100, 0, f"{h}%", disabled=disabled) + srow("Saturation", s, 0, 100, 0, f"{s}%", disabled=disabled) + srow("Lightness", l, 0, 100, 0, f"{l}%", disabled=disabled)
    line = "Pick a colour to compress toward." if disabled else "At 100 % every colour in the range takes the target's value."
    d = " data-disabled" if disabled else ""
    reset = S.link_button("Reset") if (h or s or l) and not disabled else ""
    return f'<section class="ym-card" data-last="false">{head_of("Compress", None, "", reset)}<div class="sk-rows"{d}>{rows}</div><p class="ym-row-note" style="margin:8px 0 0">{line}</p></section>'


RANGE_DEFAULT = (20, 5, 1.5, 50)


def range_card(hue=30, sat=5, ev=1.5, soft=60, eye_on=False, eye_disabled=False, last=True):
    rows = (
        srow("Hue", hue, 1, 85, 20, f"±{hue:g}°")
        + srow("Saturation", sat, 0.5, 30, 5, f"±{sat:g}")
        + srow("Lightness", ev, 0.1, 8, 1.5, f"±{ev:g} EV")
        + srow("Softness", soft, 0, 100, 50, f"{soft}%")
    )
    changed = (hue, sat, ev, soft) != RANGE_DEFAULT
    reset = S.link_button("Reset") if changed else ""
    return f'<section class="ym-card" data-last="{"true" if last else "false"}">{head_of("Range", None, eye(eye_on, eye_disabled), reset)}<div class="sk-rows">{rows}</div></section>'


def cards(target="set", compress=(60, 60, 0), rng=(30, 5, 1.5, 60), eye_on=False, **kw):
    no = target == "empty"
    return (
        target_card(target, **kw)
        + compress_card(*compress, disabled=no)
        + range_card(*rng, eye_on=eye_on, eye_disabled=no)
    )


# ------------------------------------------------------------------------------------------------------------------------------ panel
def tabs_cc():
    t = M.TABLIST.replace('data-tab="raw"', 'data-tab="color"').replace(">Blend</button>", ">Compress</button>").replace("ym-tab-raw-blend", "ym-tab-color-compress").replace("ym-tabpanel-raw", "ym-tabpanel-color")
    return t


def aside(headhtml, cardshtml):
    return (
        f'<aside aria-label="Tools" class="ym-tools" data-tab="color">{headhtml}{tabs_cc()}'
        f'<div aria-labelledby="ym-tab-color-compress" class="ym-tools__scroll" data-tab="color" id="ym-tabpanel-color" role="tabpanel">{cardshtml}</div>{M.FOOT}</aside>'
        '<div aria-live="polite" class="sr-only"></div></div></div>'
    )


def head(line, bypass_on=False, num="02"):
    h = M.head("color_compressor", "Compressor", line, pressed_on=bypass_on)
    return re.sub(r'<span class="ym-nodehead__chip">.*?</span>', f'<span class="ym-nodehead__chip" style="font-size:12px;font-weight:800">{num}</span>', h, count=1, flags=re.S)


# ------------------------------------------------------------------------------------------------------------------------------ the page
def page(title, inner):
    return (
        f'<!doctype html>\n{M.HTML_TAG}<head>\n<meta charset="utf-8"/>\n<meta content="width=device-width, initial-scale=1.0" name="viewport"/>\n<meta content="dark" name="color-scheme"/>\n'
        f'<title>Yellowmoon · {title}</title>\n<link href="../../../mixers/html/app.css" rel="stylesheet"/>\n<link href="../../../nodes/html/nv.css" rel="stylesheet"/>\n'
        f'<link href="../../../mixers/html/mixers.css" rel="stylesheet"/>\n<link href="../../../skin/html/skin.css" rel="stylesheet"/>\n<link href="../cc.css" rel="stylesheet"/>\n</head>\n{inner}\n{M.SCRIPT}</body></html>\n'
    )


def write(name, html):
    with open(os.path.join(HERE, "pages", name), "w", encoding="utf-8") as f:
        f.write(html)
    print("wrote", name, len(html))


def cc_card(g, key, x, y, num, thumb_, selected=False, bypass=False, dead=False, name="Compressor"):
    """The node's card: a corrector's, with the node's glyph in the footer where a mask glyph would be."""
    g.corrector(key, name, x, y, num, thumb_, masks=("mask",), selected=selected, bypass=bypass, dead=dead)
    h = g.html[-1]
    h = h.replace(f'data-glyph="mask">{M.HALF}', f'data-glyph="color_compressor">{svg(CC_ICON, 14)}')
    g.html[-1] = h


def photo(src, overlay="", name="Portrait", sub="2026-10-04 · 7 photos"):
    pre = S.photo_prefix("portrait", False, 28, name=name, sub=sub, overlay=overlay)
    pre = re.sub(r'(class="ym-photo-cell__image"[^>]*?src=")[^"]*(")', rf"\g<1>{AS}{src}\g<2>", pre)
    return pre


def graph(selected=True, bypass=False, dead=False, thumbs=("portrait-before.jpg", "portrait-after.jpg")):
    g = M.Graph()
    mid = 160
    xs = [120, 334]
    g.marker("input", "Input", xs[0] - 86, mid)
    g.corrector("main", "Main", xs[0], mid - 56, "01", AS + thumbs[0])
    cc_card(g, "cc", xs[1], mid - 56, "02", AS + thumbs[1], selected=selected, bypass=bypass, dead=dead)
    g.marker("output", "Output", xs[1] + 150 + 86, mid)
    for a, b in (("input", "main"), ("main", "cc"), ("cc", "output")):
        g.link(a, b)
    return g


def editor(graph_html, mask_on=False):
    bar = M.BAR06
    add = str(bar["add"])
    byp = M.pressed(str(bar["bypass"]), False)
    msk = M.pressed(str(bar["mask"]), mask_on)
    if mask_on:
        msk = msk.replace('data-state="off"', 'data-state="on"')
    exp = M.pressed(str(bar["expand"]), False)
    return (
        f'<section aria-label="Node editor" class="ym-filmstrip nv2-editor" data-tab="color"><div class="nv2-bar">{add}{byp}{bar["delete"]}'
        f'<span class="nv2-bar__sp"></span>{bar["switcher"]}{msk}{exp}</div>'
        f'<div class="nv2-canvas" style="--cw:1040px;">{graph_html}{M.ZOOM}</div></section>'
    )


def node_page(name, title, src, headline, cardshtml, overlay="", mask_on=False, bypass_on=False, scroll_end=False, extra=""):
    pre = photo(src, overlay)
    g = graph(bypass=bypass_on)
    inner = pre + editor(g.render(h=327), mask_on) + aside(head(headline, bypass_on), cardshtml) + extra
    if scroll_end:
        inner += '<script>const go=()=>{const s=document.querySelector(".ym-tools__scroll");if(s)s.scrollTop=s.scrollHeight;};addEventListener("load",()=>{go();setTimeout(go,100);});</script>'
    write(name, page(title, inner))


# ---- overlays ------------------------------------------------------------------------------------------------------------------------
def crosshair(x=600, y=590):
    """The pick crosshair and the 7 x 7 square the pick averages (scene units; the square is drawn larger than it is, to be seen)."""
    return (
        '<svg class="sk-over" viewBox="0 0 1500 1000" preserveAspectRatio="xMidYMid meet">'
        f'<g transform="translate({x},{y})" fill="none" stroke-linecap="round">'
        '<circle r="46" stroke="#000" stroke-opacity=".5" stroke-width="7"/><circle r="46" stroke="#fff" stroke-width="3.5"/>'
        '<path d="M-70 0H-54M54 0H70M0 -70V-54M0 54V70" stroke="#000" stroke-opacity=".5" stroke-width="7"/><path d="M-70 0H-54M54 0H70M0 -70V-54M0 54V70" stroke="#fff" stroke-width="3.5"/>'
        '<rect x="-12" y="-12" width="24" height="24" stroke="#ffc20e" stroke-width="3.5"/></g></svg>'
    )


def pill(text, left, top):
    return f'<div class="cc-pill" style="left:{left}px;top:{top}px">{svg(PICK_ICON, 14)}<span>{text}</span></div>'


# ---- 01 the node ------------------------------------------------------------------------------------------------------------------
def page01():
    node_page("01.html", "Colour Compressor node", "portrait-after.jpg", "Colour Compressor", cards("set"))


# ---- 02 picking the target -----------------------------------------------------------------------------------------------------------
def page02():
    pre = photo("portrait-before.jpg", crosshair())
    g = graph()
    inner = pre + editor(g.render(h=327)) + aside(head("Colour Compressor"), cards("empty", compress=(0, 0, 0))) + pill("Click a colour in the photo.", 560, 126)
    inner += '<div class="sk-callout" style="left:640px;top:440px;width:330px"><b>7 × 7 square.</b> The target is the mean colour of the pixels in the yellow square, so a noisy pixel is not the target.</div>'
    write("02.html", page("Picking the target", inner))


# ---- 03 Show affected area -------------------------------------------------------------------------------------------------------------
def page03():
    node_page("03.html", "Show affected area", "portrait-affected.jpg", "Colour Compressor", cards("set", eye_on=True), mask_on=True, scroll_end=True,
              extra='<div class="sk-callout" style="left:640px;top:440px;max-width:340px"><b>Show affected area</b> is Show mask (M) on this node. It draws the node\'s mask times its range, with the soft edge. Hair, eyes, lips, teeth and the mole are not in the range.</div>')


# ---- 04 Add node menu --------------------------------------------------------------------------------------------------------------------
def add_menu(highlight="Colour Compressor"):
    items = [
        ("Corrector", "corrector", "Grades all or part of the photo."),
        ("Film", "film", "Models a colour negative, a print and a scan."),
        ("Face Refinement", "face_refinement", "Refines each face: skin, under-eyes and teeth."),
        ("Colour Compressor", "color_compressor", "Pulls a range of colours toward one colour."),
        None,
        ("Parallel", "parallel", "Grade branches side by side."),
        ("Layer", "layer", "Blend one branch over another."),
        ("Splitter and combiner", "split-menu", "Split a picture, then join it again."),
    ]
    out = []
    for it in items:
        if it is None:
            out.append(M.SEP)
            continue
        nm, ic, line = it
        hl = ' style="background:color-mix(in oklch, var(--ink) 12%, transparent)"' if nm == highlight else ""
        out.append(
            f'<div class="{M.ITEM_CLASS}" data-slot="dropdown-menu-item" role="menuitem" tabindex="-1"{hl}>{svg(M.ICON[ic], 24)}'
            f'<span class="ym-menu-line"><span class="ym-menu-line__name">{nm}</span><span class="ym-menu-line__text">{line}</span></span></div>'
        )
    return (
        '<div data-radix-popper-content-wrapper="" style="position:fixed;left:12px;bottom:372px;z-index:50;">'
        f'<div aria-label="Add node" class="{M.MENU_CLASS} ym-add-node-menu" data-slot="dropdown-menu-content" data-state="open" role="menu" '
        f'style="width:356px;outline:none;--glass-liquid-opacity:90%;--glass-liquid-blur:22px">{"".join(out)}</div></div>'
    )


def page04():
    g = M.Graph()
    mid = 160
    xs = [120, 334, 548]
    g.marker("input", "Input", xs[0] - 86, mid)
    g.corrector("main", "Main", xs[0], mid - 56, "01", AS + "portrait-before.jpg")
    g.corrector("c2", "Warmth", xs[1], mid - 56, "02", AS + "portrait-before.jpg", selected=True)
    g.marker("output", "Output", xs[1] + 150 + 86, mid)
    for a, b in (("input", "main"), ("main", "c2"), ("c2", "output")):
        g.link(a, b)
    pre = photo("portrait-before.jpg")
    aside06 = M.BODY06[M.BODY06.find("<aside"): M.BODY06.find("</aside>") + 8].replace('src="../assets/', 'src="' + S.A).replace("Mountain", "Warmth").replace("Corrector · 2 masks", "Corrector · whole photo")
    tail = '<div aria-live="polite" class="sr-only"></div></div></div>'
    inner = pre + M.editor(g.render(h=327), M.BAR06, False, add_open=True, mask_off=False) + aside06 + tail + add_menu()
    inner += '<div class="sk-callout" style="left:400px;top:560px;max-width:340px"><b>One new item.</b> Adds a Colour Compressor after the selected node, selects it and turns Pick colour on, as Film and Face Refinement are added.</div>'
    write("04.html", page("Add node menu", inner))


# ---- 05 the simple view ------------------------------------------------------------------------------------------------------------------
def page05():
    s = open(os.path.join(M.NODES, "04.html"), encoding="utf-8").read()
    s = s.replace('src="../assets/', 'src="' + S.A)
    s = s.replace('src="' + S.A + 'photo-graded.jpg"', f'src="{AS}portrait-after.jpg"')
    s = s.replace("This photo has 4 nodes.", "This photo has 2 nodes.").replace("Provence", "Portrait")
    s = s.replace(S.A + "strip-2-graded.jpg", AS + "portrait-after.jpg")
    s = re.sub(r"Sony A1 · A1M\d+ · ISO \d+ · 1/\d+ · f/[\d.]+", "Sony A1 · A1M02214 · ISO 200 · 1/250 · f/2.0", s)
    s = re.sub(r'(<link[^>]*href=")\.\./(app\.css|nv\.css)', lambda m_: m_.group(1) + ("../../../mixers/html/app.css" if m_.group(2) == "app.css" else "../../../nodes/html/nv.css"), s)
    s = s.replace("</head>", '<link href="../../../mixers/html/mixers.css" rel="stylesheet"/><link href="../../../skin/html/skin.css" rel="stylesheet"/><link href="../cc.css" rel="stylesheet"/></head>', 1)
    s = s.replace("<title>Yellowmoon</title>", "<title>Yellowmoon · Simple view</title>")
    s = s.replace("</body>", '<div class="sk-callout" style="left:24px;top:132px;max-width:380px"><b>Simple view.</b> It has no Colour Compressor controls. The photo shows the result; the notice counts the node, and Open nodes goes to the node view.</div></body>', 1)
    write("05.html", s)


# ---- 06 how it works (a diagram) ---------------------------------------------------------------------------------------------------------
def plane_svg():
    """The chroma plane: the target, the selected wedge and ring, and what 50 % and 100 % do to a few colours (computed by compress_ref)."""
    import numpy as np

    W_ = 420
    cx = cy = W_ / 2
    ext = 0.20  # plane units at the edge of the drawing
    k = (W_ / 2 - 14) / ext
    px = lambda a, b: (cx + a * k, cy - b * k)
    tgt = dict(chroma=(T["chroma"][0], T["chroma"][1]), tone=T["tone"])
    ta, tb = tgt["chroma"]
    rt = math.hypot(ta, tb)
    th = math.atan2(tb, ta)
    rng = dict(hue_deg=30.0, chroma=0.05, tone_ev=1.5, softness=0.6)
    wh = math.radians(30.0)
    qh = max(0.6 * wh, 0.05)
    rin, rout = rt - 0.05, rt + 0.05
    out = []
    # the hue wheel, faded toward the centre
    out.append(f'<circle cx="{cx}" cy="{cy}" r="{ext * k:.1f}" fill="url(#cc-fade)"/>')
    for rr in (0.1, 0.2):
        out.append(f'<circle cx="{cx}" cy="{cy}" r="{rr * k:.1f}" fill="none" stroke="#fff" stroke-opacity=".14"/>')
    out.append(f'<path d="M{cx - ext * k} {cy}H{cx + ext * k}M{cx} {cy - ext * k}V{cy + ext * k}" stroke="#fff" stroke-opacity=".14"/>')

    def sector(r0, r1, a0, a1):
        p = [px(r0 * math.cos(a0), r0 * math.sin(a0)), px(r1 * math.cos(a0), r1 * math.sin(a0)), px(r1 * math.cos(a1), r1 * math.sin(a1)), px(r0 * math.cos(a1), r0 * math.sin(a1))]
        return f"M{p[0][0]:.1f} {p[0][1]:.1f}L{p[1][0]:.1f} {p[1][1]:.1f}A{r1 * k:.1f} {r1 * k:.1f} 0 0 0 {p[2][0]:.1f} {p[2][1]:.1f}L{p[3][0]:.1f} {p[3][1]:.1f}A{r0 * k:.1f} {r0 * k:.1f} 0 0 1 {p[0][0]:.1f} {p[0][1]:.1f}Z"

    # the soft edge, then the full range
    out.append(f'<path d="{sector(max(rin - 0.025, 0.02), rout + 0.025, th - wh - qh, th + wh + qh)}" fill="#ff6fd2" fill-opacity=".16" stroke="#ff6fd2" stroke-opacity=".5" stroke-dasharray="4 4"/>')
    out.append(f'<path d="{sector(rin, rout, th - wh, th + wh)}" fill="#ff6fd2" fill-opacity=".30" stroke="#ff6fd2" stroke-opacity=".9"/>')
    out.append(f'<circle cx="{cx}" cy="{cy}" r="{0.02 * k:.1f}" fill="#000" fill-opacity=".55" stroke="#fff" stroke-opacity=".5" stroke-dasharray="2 3"/>')
    # a few colours: before (ring) and after at 50 and 100 % on hue and saturation
    rs = np.random.default_rng(3)
    pts = []
    for _ in range(9):
        a = th + rs.uniform(-1.1 * wh, 1.1 * wh)
        r = rt + rs.uniform(-1.1 * 0.05, 1.1 * 0.05)
        pts.append((r * math.cos(a), r * math.sin(a)))
    for a, b in pts:
        v = tgt["tone"] * R.ONE + a * R.E1 + b * R.E2
        o50, _w = R.compress(v, tgt, 0.5, 0.5, 0.0, rng)
        o100, _w = R.compress(v, tgt, 1.0, 1.0, 0.0, rng)
        x0, y0 = px(a, b)
        c5 = px(o50 @ R.E1, o50 @ R.E2)
        c1 = px(o100 @ R.E1, o100 @ R.E2)
        out.append(f'<path d="M{x0:.1f} {y0:.1f}L{c1[0]:.1f} {c1[1]:.1f}" stroke="#fff" stroke-opacity=".5" stroke-width="1.5"/>')
        out.append(f'<circle cx="{c5[0]:.1f}" cy="{c5[1]:.1f}" r="3.2" fill="#ffc20e"/>')
        out.append(f'<circle cx="{c1[0]:.1f}" cy="{c1[1]:.1f}" r="3.2" fill="#fff"/>')
        out.append(f'<circle cx="{x0:.1f}" cy="{y0:.1f}" r="4.6" fill="none" stroke="#fff" stroke-width="1.6"/>')
    tx, ty = px(ta, tb)
    out.append(f'<circle cx="{tx:.1f}" cy="{ty:.1f}" r="8" fill="{RGB}" stroke="#fff" stroke-width="2.5"/>')
    out.append(f'<text x="{tx + 12:.1f}" y="{ty - 10:.1f}" fill="#fff" font-size="13" font-weight="700">Target</text>')
    out.append(f'<text x="{cx + 6}" y="{cy + 16}" fill="#fff" fill-opacity=".6" font-size="11">neutral: never selected</text>')
    defs = (
        '<defs><radialGradient id="cc-fade"><stop offset="0" stop-color="#1c1c1e" stop-opacity=".92"/><stop offset=".55" stop-color="#1c1c1e" stop-opacity=".62"/><stop offset="1" stop-color="#1c1c1e" stop-opacity=".25"/></radialGradient>'
        '<radialGradient id="cc-wheel" cx=".5" cy=".5" r=".5"><stop offset="0" stop-color="#777"/><stop offset="1" stop-color="#777"/></radialGradient></defs>'
    )
    wheel = (
        f'<foreignObject x="{cx - ext * k:.1f}" y="{cy - ext * k:.1f}" width="{2 * ext * k:.1f}" height="{2 * ext * k:.1f}"><div xmlns="http://www.w3.org/1999/xhtml" '
        f'style="width:100%;height:100%;border-radius:50%;background:conic-gradient(from 90deg, #ff4d4d 0deg, #e6e04a 60deg, #4fe05a 120deg, #2fe0e0 180deg, #4a6bff 240deg, #ff5fd0 300deg, #ff4d4d 360deg);transform:scaleY(-1)"></div></foreignObject>'
    )
    return f'<svg viewBox="0 0 {W_} {W_}" width="{W_}" height="{W_}" role="img" aria-label="The chroma plane with the target, the selected range and what the pull does">{defs}{wheel}{"".join(out)}</svg>'


def law_svg(kind):
    """The law on one axis: input against output for 0, 50 and 100 %, with the window drawn behind (computed from compress_ref's windows)."""
    import numpy as np

    W_, H_, pad = 420, 300, 34
    if kind == "hue":
        lo, hi, t, w, q = -90.0, 90.0, 0.0, 30.0, max(0.6 * 30, math.degrees(0.05))
        unit, xl = "°", "Hue of a colour, from the target's hue"
    else:
        lo, hi = -4.0, 4.0
        t, w, q = 0.0, 1.5, max(0.6 * 1.5, 0.02 / R.STOP)
        unit, xl = " EV", "Lightness of a colour, from the target's"
    xs = np.linspace(lo, hi, 721)
    win = 1.0 - R.smooth((np.abs(xs - t) - w) / q)
    px = lambda x: pad + (x - lo) / (hi - lo) * (W_ - 2 * pad)
    py = lambda y: H_ - pad - (y - lo) / (hi - lo) * (H_ - 2 * pad)
    out = []
    # the window behind
    area = f"M{px(lo):.1f} {py(lo):.1f}" + "".join(f"L{px(x):.1f} {H_ - pad - w_ * 0.28 * (H_ - 2 * pad):.1f}" for x, w_ in zip(xs[::6], win[::6])) + f"L{px(hi):.1f} {py(lo):.1f}Z"
    out.append(f'<path d="{area}" fill="#ff6fd2" fill-opacity=".20"/>')
    out.append(f'<path d="M{px(lo):.1f} {py(lo):.1f}L{px(hi):.1f} {py(hi):.1f}" stroke="#fff" stroke-opacity=".28" stroke-dasharray="3 5"/>')
    for kap, col, lab in ((0.5, "#ffc20e", "50 %"), (1.0, "#fff", "100 %")):
        ys = t + (xs - t) * (1.0 - kap * win)
        out.append(f'<path d="M' + "L".join(f"{px(x):.1f} {py(y):.1f}" for x, y in zip(xs[::3], ys[::3])) + f'" fill="none" stroke="{col}" stroke-width="2.4"/>')
    out.append(f'<path d="M{pad} {H_ - pad}H{W_ - pad}M{pad} {pad}V{H_ - pad}" stroke="#fff" stroke-opacity=".5"/>')
    out.append(f'<text x="{W_ / 2}" y="{H_ - 6}" text-anchor="middle" fill="#fff" fill-opacity=".7" font-size="12">{xl}</text>')
    out.append(f'<text x="12" y="{pad + 4}" fill="#fff" fill-opacity=".7" font-size="12" transform="rotate(-90 12 {H_ / 2})" text-anchor="middle">after</text>')
    out.append(f'<text x="{px(0):.1f}" y="{H_ - pad + 16}" text-anchor="middle" fill="#fff" fill-opacity=".8" font-size="11">0</text>')
    out.append(f'<text x="{px(w):.1f}" y="{H_ - pad + 16}" text-anchor="middle" fill="#ff6fd2" font-size="11">{w:g}{unit}</text>')
    out.append(f'<text x="{px(lo) + 4:.1f}" y="{py(lo) - 6:.1f}" fill="#ff6fd2" fill-opacity=".9" font-size="11">the range</text>')
    return f'<svg viewBox="0 0 {W_} {H_}" width="{W_}" height="{H_}" role="img" aria-label="Output against input on one axis">{"".join(out)}</svg>'


def page06():
    legend = '<span><i style="background:none;box-shadow:inset 0 0 0 2px #fff"></i>a colour before</span><span><i style="background:#ffc20e"></i>after, 50 %</span><span><i style="background:#fff"></i>after, 100 %</span><span><i style="background:#ff6fd2;opacity:.5"></i>the range (dashed: the soft edge)</span>'
    inner = (
        '<body><div class="mx-sheet sk-sheet">'
        '<h1 class="mx-sheet__title">Colour Compressor · how the pull works</h1>'
        '<p class="mx-sheet__sub">Drawn from the maths of <code>docs/32</code> §3 by the sketch <code>compress_ref.py</code>, not by hand. Each colour keeps the share <code>1 − κW</code> of its distance to the target on each axis: 0 % leaves it, 100 % puts it on the target. Outside the range <code>W</code> is 0, so nothing moves; at the soft edge it moves by a share.</p>'
        '<section><h2>The chroma plane: hue and saturation</h2><div class="mx-row cc-diagram">'
        f'<figure class="cc-fig">{plane_svg()}<figcaption>Nine colours near the target are pulled toward it, on hue and saturation together. Greys are never selected.<div class="sk-legend">{legend}</div></figcaption></figure>'
        f'<figure class="cc-fig">{law_svg("hue")}<figcaption><b>Hue, one axis.</b> A colour 20° from the target (inside ±30°) is pulled to 10° at 50 % and to 0° at 100 %. Outside the range the line is the diagonal: nothing moves. The line never turns back: no colour order is inverted, and below 100 % no two colours are made equal.</figcaption></figure>'
        f'<figure class="cc-fig">{law_svg("tone")}<figcaption><b>Lightness, one axis.</b> The same law on the mean of the three codes, in stops. At 100 % the colours inside the range share the target\'s lightness: that is the point of the tool, and why Lightness starts at 0 % and a face stays round.</figcaption></figure>'
        '</div></section></div>'
    )
    write("06.html", page("How the pull works", inner))


# ---- states -------------------------------------------------------------------------------------------------------------------
def sheet_card(caption, body, note=""):
    n = f"<small>{note}</small>" if note else ""
    return f'<figure class="sk-cell"><div class="sk-cell__stage sk-panel ym-tools" data-tab="color" style="width:400px">{body}</div><figcaption>{caption}{n}</figcaption></figure>'


def panel_fragment(headline, cardshtml, bypass=False):
    return head(headline, bypass) + f'<div class="ym-tools__scroll" data-tab="color" style="overflow:visible;height:auto">{cardshtml}</div>'


def node_cell(caption, note="", **kw):
    g = M.Graph()
    cc_card(g, "s", 47, 90, "02", AS + "portrait-after.jpg", **kw)
    return f'<figure class="mx-cell" style="width:250px"><div class="mx-cell__stage" style="height:230px">{g.render(250, 230)}</div><figcaption>{caption}{("<small>" + note + "</small>") if note else ""}</figcaption></figure>'


def states_page():
    cells = [
        node_cell("Normal", "numbered like Film; the node's glyph sits where mask glyphs do"),
        node_cell("Selected", selected=True),
        node_cell("Bypassed", "half-strength thumbnail and the slash", bypass=True),
        node_cell("Dead", "not on a path to the Output: 45 %", dead=True),
    ]
    grey_rgb = "rgb(126,120,116)"
    panels = [
        ("A new node (Pick colour on)", "Colour Compressor", cards("empty", compress=(0, 0, 0)), "no target: Hue, Saturation and Lightness are off and the Compress sliders say why; the Range card still works and its eye is off"),
        ("After the first click", "Colour Compressor", cards("picking", compress=(50, 50, 0), rng=RANGE_DEFAULT), "the first pick set Hue and Saturation to 50 %; Pick colour stays on for the next click; the Range card shows its defaults"),
        ("A target set, Pick colour off", "Colour Compressor", cards("set"), "the numbers can be typed: this is the target's shift"),
        ("A pick too grey for a hue", "Colour Compressor", cards("set", hue=22, sat=2, ev=0.1, rgb=grey_rgb, note="That colour is too grey for a hue. It was moved to Saturation 2."), "held at Saturation 2; the line goes at the next change"),
        ("The widest range", "Colour Compressor", cards("set", rng=(85, 30, 8, 100)), "Hue ±85° is the most; Saturation ±30 and Lightness ±8 EV take every saturation and every tone, which is the axis off"),
        ("Bypassed", "Bypassed", cards("set"), "the head's Bypass is on; the node passes its picture through"),
    ]
    pcells = [sheet_card(cap, panel_fragment(line, ch, bypass=(line == "Bypassed")), note) for cap, line, ch, note in panels]
    tips = [
        ("Add node › Colour Compressor", "Pulls a range of colours toward one colour."),
        ("Add node › Colour Compressor, at 64 nodes", "A photo can have up to 64 nodes."),
        ("Pick colour (the toggle)", "Click a colour in the photo. Esc stops."),
        ("The hint over the photo", "Click a colour in the photo."),
        ("The empty target", "No colour yet. Click a colour in the photo."),
        ("Hue, Saturation and Lightness of the target, no target", "Pick a colour first."),
        ("The swatch (its accessible name)", "Target colour: orange, hue 26 degrees, saturation 9.7, lightness minus 0.03 stops."),
        ("A pick too grey for a hue", "That colour is too grey for a hue. It was moved to Saturation 2."),
        ("Compress, no target", "Pick a colour to compress toward."),
        ("Compress, with a target", "At 100 % every colour in the range takes the target's value."),
        ("Range › Hue", "How far from the target's hue colours are still pulled."),
        ("Range › Saturation", "How far from the target's saturation colours are still pulled."),
        ("Range › Lightness", "How far from the target's lightness colours are still pulled."),
        ("Range › Softness", "How gradually the pull fades at the edge of the range."),
        ("Show affected area (the eye)", "Show affected area"),
        ("Show affected area, no target", "Pick a colour first."),
        ("Target › Reset", "Clears the target. The amounts stay."),
        ("Compress › Reset", "Sets Hue, Saturation and Lightness back to 0."),
        ("Range › Reset", "Puts the range back to ±20°, ±5, ±1.5 EV and 50 %."),
        ("Reset (the head)", "Reset this node. Masks stay."),
        ("The simple view's notice", "This photo has 2 nodes. These controls change the input, the output and node 1, Main."),
    ]
    tip_html = "".join(f'<div class="mx-tiprow"><span>{a}</span>{M.tip(b)}</div>' for a, b in tips)
    inner = (
        '<body><div class="mx-sheet sk-sheet">'
        '<h1 class="mx-sheet__title">Colour Compressor · states</h1>'
        '<p class="mx-sheet__sub">The panel is 400 px wide, as in the app. The face is a drawing made for design/skin/; the compressed picture is the maths of docs/32 §3 applied to it.</p>'
        f'<section><h2>Node card</h2><div class="mx-row">{"".join(cells)}</div></section>'
        f'<section><h2>Panel</h2><div class="mx-row sk-grid">{"".join(pcells)}</div></section>'
        f'<section><h2>Copy for tooltips and notes (drafts)</h2><div class="mx-tips">{tip_html}</div></section>'
        "</div>"
    )
    write("states.html", page("Colour Compressor states", inner))


if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "pages"), exist_ok=True)
    page01()
    page02()
    page03()
    page04()
    page05()
    page06()
    states_page()
