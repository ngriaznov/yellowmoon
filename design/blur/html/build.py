#!/usr/bin/env python3
"""Builds the pages of design/blur/html/pages/ (task BL-01). Same method as design/colour-uniform/html/build.py, which this file imports for the shared pieces
(the app's captured DOM and CSS: header, photo cell, toolbar, panel head, tabs, cards, sliders, the graph). It adds the Blur node (kind `blur`, docs/33).
The pictures are the synthetic scene of blur_demo.py with the maths of docs/33 section 3 applied by blur_ref.py. Run: python3 build.py  (needs bs4). Not product code.
"""
import importlib.util
import os
import re
import sys

from bs4 import BeautifulSoup

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "colour-uniform", "html"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "skin", "html"))
spec = importlib.util.spec_from_file_location("cub", os.path.join(HERE, "..", "..", "colour-uniform", "html", "build.py"))
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)
S, M = C.S, C.M
svg = M.svg
AS = "../assets/"

BL_ICON = '<circle cx="12" cy="12" r="3"/><circle cx="12" cy="12" r="6.5" stroke-dasharray="2 2.6"/><circle cx="12" cy="12" r="10" stroke-dasharray="1.2 3.6"/>'
M.ICON["blur"] = BL_ICON
PERSON = '<circle cx="12" cy="7" r="4"/><path d="M5.5 21a6.5 6.5 0 0 1 13 0"/>'
CROSS = '<circle cx="12" cy="12" r="3"/><path d="M12 2v4M12 18v4M2 12h4M18 12h4"/>'

TYPES = ["Soft", "Lens", "Motion", "Zoom"]
LINE = "Soft"


# ---------------------------------------------------------------------------------------------------------------- panel pieces
def segmented(labels, active, label):
    s = S.segmented(labels, active, 352, label)
    s = s.replace("min-w-max flex-none", "flex-1")
    return s


def pick_centre(on):
    b = BeautifulSoup(M.FOOT, "html.parser").find("button")
    b.clear()
    b.append(BeautifulSoup(svg(CROSS, 16), "html.parser"))
    b.append(BeautifulSoup("<span>Pick centre</span>", "html.parser"))
    b["aria-pressed"] = "true" if on else "false"
    b["aria-describedby"] = ""
    b["data-state"] = "on" if on else "off"
    if on:
        b["class"] = b.get("class", []) + ["bl-pick-on"]
    if b.has_attr("aria-disabled"):
        del b["aria-disabled"]
    return str(b)


def blur_card(kind="Soft", radius=20, blades=0, rotation=0, highlights=0, length=40, angle=0, amount=40, centre_on=False, centre_set=True, changed=True):
    k = TYPES.index(kind)
    rows = ""
    note = ""
    if kind in ("Soft", "Lens"):
        rows += C.srow("Radius", radius, 0, 100, 20, f"{radius}")
    if kind == "Lens":
        labs = ["Round", "5", "6", "7", "8", "9"]
        act = 0 if blades == 0 else blades - 4
        shape = f'<div class="bl-shape"><span>Shape</span>{segmented(labs, act, "Aperture shape: round, or the number of blades")}</div>'
        rows += shape
        rows += C.srow("Rotation", rotation, 0, 359, 0, f"{rotation}°", disabled=(blades == 0))
        rows += C.srow("Highlights", highlights, 0, 100, 0, f"{highlights}")
    if kind == "Motion":
        rows += C.srow("Length", length, 0, 100, 0, f"{length}")
        rows += C.srow("Angle", angle, 0, 179, 0, f"{angle}°")
    if kind == "Zoom":
        rows += C.srow("Amount", amount, 0, 100, 0, f"{amount}")
        txt = "Click the photo or drag the handle." if centre_on else ("Centre of the picture." if not centre_set else "Moved from the centre.")
        rows += f'<div class="bl-centre">{pick_centre(centre_on)}<p>{txt}</p></div>'
    quiet = {
        "Soft": "Radius is a share of the picture's size, so the view and every export look alike.",
        "Lens": "Bright lights become discs, or the shape of the blades. Highlights makes them stronger.",
        "Motion": "Length is a share of the picture's size. Angle 0° is along the picture's width.",
        "Zoom": "Pixels farther from the centre blur more. Amount 0 changes nothing.",
    }[kind]
    seg = f'<div class="bl-seg">{segmented(TYPES, k, "Blur type")}</div>'
    reset = S.link_button("Reset") if changed else ""
    return (
        f'<section class="ym-card" data-last="false">{C.head_of("Blur", None, "", reset)}{seg}<div class="sk-rows">{rows}</div>'
        f'<p class="ym-row-note" style="margin:8px 0 0">{quiet}</p></section>'
    )


def edges_card(respect=True, masked=True):
    sw = S.switch(respect, "Respect mask edges", "Colours from outside the mask are not blurred in, so there is no glow around the subject." if respect else "The whole picture is blurred, then the mask decides where the blur shows.")
    line = "" if masked else '<p class="ym-row-note" style="margin:6px 0 0">This node has no mask, so this changes nothing yet.</p>'
    return f'<section class="ym-card" data-last="true">{C.head_of("Mask edges", None, "", "")}{sw}{line}</section>'


def cards(**kw):
    respect = kw.pop("respect", True)
    masked = kw.pop("masked", True)
    return blur_card(**kw) + edges_card(respect, masked)


def mask_cards(selected=True, mask_view=False):
    """The Mask tab with one Subject row, inverted: the card is a corrector's (Mask tab of docs/23 7.6); built from the Colour Uniform page's by renaming the row."""
    h = C.mask_cards(1, selected, mask_view)
    s = BeautifulSoup(h, "html.parser")
    for r in s.select(".nv-maskrow"):
        r["aria-label"] = "Subject mask, inverted"
        r.select_one(".nv-maskrow__name").string = "Subject"
        r.select_one(".nv-maskrow__sub").string = "Inverted · background"
        g = r.find("svg")
        g.clear()
        g.append(BeautifulSoup(PERSON, "html.parser"))
    # replace the Colour range card by the Subject row's card: Invert (on), Feather, Refine edges, Opacity
    secs = s.find_all("section", recursive=False)
    for sec in secs:
        t = sec.select_one("h3")
        if t and t.text.strip() == "Colour range":
            rows = C.srow("Feather", 8, 0, 100, 0, "8") + C.srow("Refine edges", 50, 0, 100, 0, "50%") + C.srow("Opacity", 100, 0, 100, 100, "100%")
            inv = S.switch(True, "Invert", "Blur everything but the subject.")
            sec.replace_with(BeautifulSoup(f'<section class="ym-card" data-last="false">{C.head_of("Subject", None, "", S.link_button("Reset"))}{inv}<div class="sk-rows">{rows}</div></section>', "html.parser"))
    return str(s)


def tabs_bl(active="blur"):
    t = M.TABLIST.replace('data-tab="raw"', 'data-tab="color"' if active == "blur" else 'data-tab="mask"').replace(">Blend</button>", ">Blur</button>")
    t = t.replace("ym-tab-raw-blend", "ym-tab-color-compress").replace("ym-tabpanel-raw", "ym-tabpanel-color").replace('data-tab="raw"', 'data-tab="color"')
    if active == "mask":
        t = t.replace('aria-selected="true"', 'aria-selected="__A"').replace('aria-selected="false"', 'aria-selected="true"').replace('aria-selected="__A"', 'aria-selected="false"')
        t = t.replace('data-state="active"', 'data-state="__A"').replace('data-state="inactive"', 'data-state="active"').replace('data-state="__A"', 'data-state="inactive"')
        t = t.replace('tabindex="0" type="button">Blur', 'tabindex="-1" type="button">Blur').replace('tabindex="-1" type="button">Mask', 'tabindex="0" type="button">Mask')
    return t


def head(line, bypass_on=False, num="02"):
    h = M.head("blur", "Blur", line, pressed_on=bypass_on)
    return re.sub(r'<span class="ym-nodehead__chip">.*?</span>', f'<span class="ym-nodehead__chip" style="font-size:12px;font-weight:800">{num}</span>', h, count=1, flags=re.S)


C.tabs_cu = tabs_bl   # C.aside builds its tab row with this


# ---------------------------------------------------------------------------------------------------------------- the page
def page(title, inner):
    return (
        f'<!doctype html>\n{M.HTML_TAG}<head>\n<meta charset="utf-8"/>\n<meta content="width=device-width, initial-scale=1.0" name="viewport"/>\n<meta content="dark" name="color-scheme"/>\n'
        f'<title>Yellowmoon · {title}</title>\n<link href="../../../mixers/html/app.css" rel="stylesheet"/>\n<link href="../../../nodes/html/nv.css" rel="stylesheet"/>\n'
        f'<link href="../../../mixers/html/mixers.css" rel="stylesheet"/>\n<link href="../../../skin/html/skin.css" rel="stylesheet"/>\n<link href="../../../colour-uniform/html/cc.css" rel="stylesheet"/>\n<link href="../bl.css" rel="stylesheet"/>\n</head>\n{inner}\n{M.SCRIPT}</body></html>\n'
    )


def fix_paths(h):
    """The pages sit one folder deeper than colour-uniform's (html/pages/ here too, but the shared sheets are reached from design/blur/html/pages/)."""
    return h


def write(name, html):
    os.makedirs(os.path.join(HERE, "pages"), exist_ok=True)
    html = html.replace('data-tab="color"', 'data-tab="detail"')   # the Blur tab is in the Detail tab's colour (docs/33 2.6)
    with open(os.path.join(HERE, "pages", name), "w", encoding="utf-8") as f:
        f.write(html)
    print("wrote", name, len(html))


def bl_card(g, key, x, y, num, thumb_, selected=False, bypass=False, dead=False, name="Blur", masked=True):
    g.corrector(key, name, x, y, num, thumb_, masks=("cu", "mask") if masked else ("cu",), selected=selected, bypass=bypass, dead=dead)
    h = g.html[-1]
    h = h.replace(f'data-glyph="cu">{M.HALF}', f'data-glyph="blur">{svg(BL_ICON, 14)}')
    g.html[-1] = h


def photo(src, overlay="", name="Street at dusk", sub="2026-11-18 · synthetic scene"):
    pre = S.photo_prefix("portrait", False, 28, name=name, sub=sub, overlay=overlay)
    return re.sub(r'(class="ym-photo-cell__image"[^>]*?src=")[^"]*(")', rf"\g<1>{AS}{src}\g<2>", pre)


def graph(after, selected=True, bypass=False, dead=False, masked=True):
    g = M.Graph()
    mid = 160
    xs = [120, 334]
    g.marker("input", "Input", xs[0] - 86, mid)
    g.corrector("main", "Main", xs[0], mid - 56, "01", AS + "scene.jpg")
    bl_card(g, "bl", xs[1], mid - 56, "02", AS + after, selected=selected, bypass=bypass, dead=dead, masked=masked)
    g.marker("output", "Output", xs[1] + 150 + 86, mid)
    for a, b in (("input", "main"), ("main", "bl"), ("bl", "output")):
        g.link(a, b)
    return g


def node_page(name, title, src, cardshtml, overlay="", mask_on=False, bypass_on=False, active="blur", masked=True, extra="", after=None, line=LINE):
    pre = photo(src, overlay)
    g = graph(after or src, bypass=bypass_on, masked=masked)
    inner = pre + C.editor(g.render(h=327), mask_on, "color" if active == "blur" else "mask") + C.aside(head(line, bypass_on), cardshtml, active) + extra
    write(name, page(title, inner))


def callout(left, top, text, width=360):
    return f'<div class="sk-callout" style="left:{left}px;top:{top}px;max-width:{width}px">{text}</div>'


def centre_handle(fx, fy):
    x, y = fx * 1500, fy * 1000
    rays = "".join(f'<path d="M{x:.0f} {y:.0f}L{x + 900 * c:.0f} {y + 900 * s:.0f}" stroke="#fff" stroke-opacity=".22" stroke-width="2" stroke-dasharray="10 12"/>' for c, s in ((1, 0), (0.7071, 0.7071), (0, 1), (-0.7071, 0.7071), (-1, 0), (-0.7071, -0.7071), (0, -1), (0.7071, -0.7071)))
    return (
        '<svg class="sk-over" viewBox="0 0 1500 1000" preserveAspectRatio="xMidYMid meet">' + rays +
        f'<g transform="translate({x:.0f},{y:.0f})" fill="none" stroke-linecap="round"><circle r="26" stroke="#000" stroke-opacity=".5" stroke-width="7"/><circle r="26" stroke="#fff" stroke-width="3.5"/>'
        '<path d="M-44 0H-32M32 0H44M0 -44V-32M0 32V44" stroke="#000" stroke-opacity=".5" stroke-width="7"/><path d="M-44 0H-32M32 0H44M0 -44V-32M0 32V44" stroke="#fff" stroke-width="3.5"/>'
        '<circle r="4" fill="#ffc20e"/></g></svg>'
    )


# ---------------------------------------------------------------------------------------------------------------- pages
def page01():
    node_page("01.html", "Blur node, Soft", "soft-b.jpg", cards(kind="Soft", radius=50),
              extra=callout(640, 600, "<b>Soft, Radius 50.</b> The node's mask is the inverted Subject: the background blurs, the figure stays sharp, and there is no red glow around her. “Respect mask edges” is on (the default).", 380))


def page02():
    node_page("02.html", "Blur node, Lens", "lens-b.jpg", cards(kind="Lens", radius=50, blades=6, rotation=15, highlights=60), line="Lens",
              extra=callout(640, 600, "<b>Lens, 6 blades.</b> The string lights become hexagons, because the blur is in linear light and a light is far brighter than the sky. Highlights 60 weights the lights up so they read as discs. The figure is outside the mask.", 380))


def page03():
    node_page("03.html", "Blur node, Motion", "motion-b.jpg", cards(kind="Motion", length=45, angle=20), line="Motion",
              extra=callout(640, 600, "<b>Motion, Length 45, Angle 20°.</b> The background streaks along one line; the figure stays sharp.", 380))


def page04():
    node_page("04.html", "Blur node, Zoom", "zoom-b.jpg", cards(kind="Zoom", amount=80, centre_on=True, centre_set=True), line="Zoom", overlay=centre_handle(0.40, 0.40),
              extra=callout(640, 600, "<b>Zoom, Amount 80.</b> The centre is the handle on the photo (Pick centre, or drag it), as the Radial mask's centre is. Pixels farther from it blur more; the centre stays sharp.", 380))


def page05():
    node_page("05.html", "The Mask tab", "soft-b.jpg", mask_cards(), active="mask",
              extra=callout(640, 600, "<b>The mask is a corrector's.</b> One Subject row, inverted, so the node blurs the background. A Depth row (“far things”) works the same way: see 10-types.jpg. Every blur type reads the same mask.", 380))


def page06():
    node_page("06.html", "Show mask", "show-mask.jpg", cards(kind="Soft", radius=50), mask_on=True, after="soft-b.jpg",
              extra=callout(640, 600, "<b>Show mask</b> is the ordinary one: the node's mask, here the background. It does not change with the blur type or the radius.", 380))


def add_menu(highlight="Blur"):
    items = [
        ("Corrector", "corrector", "Grades all or part of the photo."),
        ("Film", "film", "Models a colour negative, a print and a scan."),
        ("Face Refinement", "face_refinement", "Refines each face: skin, under-eyes and teeth."),
        ("Colour Uniform", "color_uniform", "Pulls colours toward one colour."),
        ("Blur", "blur", "Soft, lens, motion or zoom blur."),
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


def page07():
    g = M.Graph()
    mid = 160
    xs = [120, 334]
    g.marker("input", "Input", xs[0] - 86, mid)
    g.corrector("main", "Main", xs[0], mid - 56, "01", AS + "scene.jpg")
    g.corrector("c2", "Warmth", xs[1], mid - 56, "02", AS + "scene.jpg", selected=True)
    g.marker("output", "Output", xs[1] + 150 + 86, mid)
    for a, b in (("input", "main"), ("main", "c2"), ("c2", "output")):
        g.link(a, b)
    pre = photo("scene.jpg")
    aside06 = M.BODY06[M.BODY06.find("<aside"): M.BODY06.find("</aside>") + 8].replace('src="../assets/', 'src="' + S.A).replace("Mountain", "Warmth").replace("Corrector · 2 masks", "Corrector · whole photo")
    tail = '<div aria-live="polite" class="sr-only"></div></div></div>'
    inner = pre + M.editor(g.render(h=327), M.BAR06, False, add_open=True, mask_off=False) + aside06 + tail + add_menu()
    inner += callout(400, 560, "<b>One new item, after Colour Uniform.</b> Adds a Blur node after the selected one and selects it, as Film and Face Refinement are added. A new node starts at Radius 20 (docs/33 §2.5).")
    write("07.html", page("Add node menu", inner))


def page08():
    s = open(os.path.join(M.NODES, "04.html"), encoding="utf-8").read()
    s = s.replace('src="../assets/', 'src="' + S.A)
    s = s.replace('src="' + S.A + 'photo-graded.jpg"', f'src="{AS}soft-b.jpg"')
    s = s.replace("This photo has 4 nodes.", "This photo has 2 nodes.").replace("Provence", "Street at dusk")
    s = s.replace(S.A + "strip-2-graded.jpg", AS + "soft-b.jpg")
    s = re.sub(r"Sony A1 · A1M\d+ · ISO \d+ · 1/\d+ · f/[\d.]+", "Sony A1 · A1M02214 · ISO 200 · 1/250 · f/2.0", s)
    s = re.sub(r'(<link[^>]*href=")\.\./(app\.css|nv\.css)', lambda m_: m_.group(1) + ("../../../mixers/html/app.css" if m_.group(2) == "app.css" else "../../../nodes/html/nv.css"), s)
    s = s.replace("</head>", '<link href="../../../mixers/html/mixers.css" rel="stylesheet"/><link href="../../../skin/html/skin.css" rel="stylesheet"/><link href="../../../colour-uniform/html/cc.css" rel="stylesheet"/><link href="../bl.css" rel="stylesheet"/></head>', 1)
    s = s.replace("<title>Yellowmoon</title>", "<title>Yellowmoon · Simple view</title>")
    s = s.replace("</body>", '<div class="sk-callout" style="left:24px;top:132px;max-width:380px"><b>Simple view.</b> It has no Blur controls. The photo shows the result; the notice counts the node, and Open nodes goes to the node view.</div></body>', 1)
    write("08.html", s)


SHEET_HEAD = '<body><div class="mx-sheet sk-sheet">'


def page09():
    nums = open(os.path.join(AS_DIR, "numbers.txt")).read().split()
    mean = float(nums[nums.index("mean_share_off") + 1])
    p95 = float(nums[nums.index("p95_off") + 1])
    inner = (
        SHEET_HEAD + '<h1 class="mx-sheet__title">Blur · Respect mask edges, off and on</h1>'
        '<p class="mx-sheet__sub">The same picture, the same mask (the inverted Subject, Feather 8), the same Soft blur at Radius 50. Only the switch differs. The picture is the maths of <code>docs/33</code> §3 applied by <code>blur_ref.py</code> to a synthetic scene.</p>'
        '<section><h2>The edge of the jacket, three times larger</h2><div class="bl-halo">'
        f'<figure><img class="crop" src="{AS}halo-scene.jpg" alt="The edge, before"><figcaption><b>Before</b>The red jacket against the dark street. Nothing blurred.</figcaption></figure>'
        f'<figure><img class="crop" src="{AS}halo-soft-a.jpg" alt="The edge, mask edges not respected"><figcaption><b>Off: blur everything, then blend by the mask</b>A red glow spreads from the jacket into the blurred street and shows as a pink rim. The figure\'s colour is in the background.</figcaption></figure>'
        f'<figure><img class="crop" src="{AS}halo-soft-b.jpg" alt="The edge, mask edges respected"><figcaption><b>On (the default): only the background blurs</b>The street is blurred from street pixels alone. There is no glow; the jacket keeps its sharp edge.</figcaption></figure>'
        '</div>'
        f'<p class="bl-num">Just outside the silhouette, <b>{mean * 100:.0f} %</b> of a blurred pixel\'s colour comes from the figure with the switch off (95th percentile {p95 * 100:.0f} %), and <b>0 %</b> with it on. Measured by the sketch on this scene, in the ring of 19,879 pixels out to 16 px from the figure.</p></section>'
        '<section><h2>The whole picture</h2><div class="bl-full">'
        f'<figure><img src="{AS}soft-a.jpg" alt="Off"><i class="bl-box"></i><figcaption style="margin-top:8px;font-size:13px;color:var(--glass-label-secondary)"><b style="color:var(--glass-label)">Off.</b> The yellow box is the crop above.</figcaption></figure>'
        f'<figure><img src="{AS}soft-b.jpg" alt="On"><i class="bl-box"></i><figcaption style="margin-top:8px;font-size:13px;color:var(--glass-label-secondary)"><b style="color:var(--glass-label)">On.</b> The same box.</figcaption></figure>'
        '</div></section></div>'
    )
    write("09.html", page("Respect mask edges", inner))


def tile(src, title, line):
    return f'<figure class="bl-tile"><img src="{AS}{src}" alt="{title}"><figcaption><b>{title}</b><br>{line}</figcaption></figure>'


def page10():
    tiles = [
        tile("scene.jpg", "Before", "The synthetic scene. Everything is sharp."),
        tile("soft-b.jpg", "Soft", "Radius 50 · a Gaussian, σ half the radius. Mask: Subject, inverted."),
        tile("lens-b.jpg", "Lens", "Radius 50 · 6 blades, rotation 15° · Highlights 60. Same mask."),
        tile("motion-b.jpg", "Motion", "Length 45 · Angle 20°. Same mask."),
        tile("zoom-b.jpg", "Zoom", "Amount 80 · centre at the figure's face. Same mask."),
        tile("depth-soft-b.jpg", "Soft, with a Depth mask", "Radius 40. Mask: a Depth row (far things), no Subject row: the street recedes upward, the figure is nearest. The “Blur the background” starting point."),
    ]
    inner = (
        SHEET_HEAD + '<h1 class="mx-sheet__title">Blur · the four types, through one mask</h1>'
        '<p class="mx-sheet__sub">One node, a Type control (Soft, Lens, Motion, Zoom). Every type reads the same mask the same way: with “Respect mask edges” on, nothing from outside the mask is blurred in. The pictures are the maths of <code>docs/33</code> by <code>blur_ref.py</code> on a synthetic scene (no photograph).</p>'
        f'<section><div class="bl-grid">{"".join(tiles)}</div></section></div>'
    )
    write("10.html", page("Blur types", inner))


def node_cell(caption, note="", **kw):
    g = M.Graph()
    bl_card(g, "s", 47, 90, "02", AS + "soft-b.jpg", **kw)
    return f'<figure class="mx-cell" style="width:250px"><div class="mx-cell__stage" style="height:230px">{g.render(250, 230)}</div><figcaption>{caption}{("<small>" + note + "</small>") if note else ""}</figcaption></figure>'


def sheet_card(caption, body, note=""):
    n = f"<small>{note}</small>" if note else ""
    return f'<figure class="sk-cell"><div class="sk-cell__stage sk-panel ym-tools" data-tab="color" style="width:400px">{body}</div><figcaption>{caption}{n}</figcaption></figure>'


def panel_fragment(headline, cardshtml, bypass=False):
    return head(headline, bypass) + f'<div class="ym-tools__scroll" data-tab="color" style="overflow:visible;height:auto">{cardshtml}</div>'


def states_page():
    cells = [
        node_cell("Normal", "numbered like Film; the node's glyph sits beside the mask glyph"),
        node_cell("Selected", selected=True),
        node_cell("Bypassed", "half-strength thumbnail and the slash", bypass=True),
        node_cell("Dead", "not on a path to the Output: 45 %", dead=True),
    ]
    panels = [
        ("A new node", LINE, cards(kind="Soft", radius=20, changed=False, masked=False), "Radius 20 on the whole photo: a new node blurs a little at once. “Respect mask edges” is on and says it changes nothing without a mask"),
        ("Lens, round", "Lens", cards(kind="Lens", radius=30, blades=0, highlights=0), "Round: Rotation is off. Highlights 0 is the plain disc"),
        ("Zoom, picking the centre", "Zoom", cards(kind="Zoom", amount=60, centre_on=True), "Pick centre is on: click the photo or drag the handle; Esc ends it"),
        ("The switch off", LINE, cards(kind="Soft", radius=50, respect=False), "The whole picture is blurred, then the mask blends: the behaviour of any corrector. The halo of page 09 returns"),
        ("Bypassed", "Bypassed", cards(kind="Soft", radius=50), "the head's Bypass is on; the node passes its picture through"),
    ]
    pcells = [sheet_card(cap, panel_fragment(line, ch, bypass=(line == "Bypassed")), note) for cap, line, ch, note in panels]
    tips = [
        ("Add node › Blur", "Soft, lens, motion or zoom blur."),
        ("Add node › Blur, at 64 nodes", "A photo can have up to 64 nodes."),
        ("Type", "Blur type"),
        ("Radius (tooltip)", "How far the blur reaches, as a share of the picture's size."),
        ("Shape, Round", "A round aperture. Choose a number of blades for a polygon."),
        ("Highlights (tooltip)", "Makes bright lights count for more, so they form clear discs."),
        ("Length (tooltip)", "How far the blur runs, as a share of the picture's size."),
        ("Angle (tooltip)", "The direction of the blur. 0° is along the picture's width."),
        ("Amount, Zoom (tooltip)", "How much blur grows toward the edges."),
        ("Pick centre", "Click the photo to set the centre. Esc stops."),
        ("Respect mask edges, on", "Colours from outside the mask are not blurred in, so there is no glow around the subject."),
        ("Respect mask edges, off", "The whole picture is blurred, then the mask decides where the blur shows."),
        ("Respect mask edges, no mask", "This node has no mask, so this changes nothing yet."),
        ("Blur › Reset", "Sets the type and its settings back to the defaults. Masks stay."),
        ("The simple view's notice", "This photo has 2 nodes. These controls change the input, the output and node 1, Main."),
    ]
    tip_html = "".join(f'<div class="mx-tiprow"><span>{a}</span>{M.tip(b)}</div>' for a, b in tips)
    inner = (
        SHEET_HEAD + '<h1 class="mx-sheet__title">Blur · states</h1>'
        '<p class="mx-sheet__sub">The panel is 400 px wide, as in the app. The picture is a synthetic scene with the maths of docs/33 §3 applied to it.</p>'
        f'<section><h2>Node card</h2><div class="mx-row">{"".join(cells)}</div></section>'
        f'<section><h2>Panel</h2><div class="mx-row sk-grid">{"".join(pcells)}</div></section>'
        f'<section><h2>Copy for tooltips and notes (drafts)</h2><div class="mx-tips">{tip_html}</div></section></div>'
    )
    write("states.html", page("Blur states", inner))


AS_DIR = os.path.join(HERE, "assets")

if __name__ == "__main__":
    page01(); page02(); page03(); page04(); page05(); page06(); page07(); page08(); page09(); page10(); states_page()
