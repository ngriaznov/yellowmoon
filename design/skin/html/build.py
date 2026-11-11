#!/usr/bin/env python3
"""Builds the pages of design/skin/html/pages/ (task S-11, design step; the owner reviews them at S-H1).

Same method as design/mixers/html/build.py: the app's DOM and CSS captured in design/nodes/html/pages/ (header, photo cell, toolbar, panel head,
tabs, cards, sliders, segmented controls), the graph and the panel rebuilt from the app's own classes. This file imports the mixers builder for
the shared pieces (it only reads files at import) and adds the Skin node. The pictures are drawings made by faces.py (no photograph).
Run: python3 build.py   (needs bs4).
"""
import copy
import importlib.util
import os
import re

from bs4 import BeautifulSoup

import faces as F

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("mxb", os.path.join(HERE, "..", "..", "mixers", "html", "build.py"))
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

A = M.A  # ../../../nodes/html/assets/
AS = "../assets/"  # ours
svg = M.svg

FACE_ICON = '<path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/><path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/><path d="M8 14s1.5 2 4 2 4-2 4-2"/><path d="M9 9h.01"/><path d="M15 9h.01"/>'
EYE = '<path d="M2.062 12.348a1 1 0 0 1 0-.696 10.75 10.75 0 0 1 19.876 0 1 1 0 0 1 0 .696 10.75 10.75 0 0 1-19.876 0"/><circle cx="12" cy="12" r="3"/>'
M.ICON["skin"] = FACE_ICON

# ------------------------------------------------------------------------------------------------------------------------------ templates
S07 = open(os.path.join(M.NODES, "07.html"), encoding="utf-8").read()
SOUP07 = BeautifulSoup(S07[S07.find("<body"):], "html.parser")
SEG = SOUP07.select_one('.nv-maskrow [data-slot="segmented-control"]')
EYE_BTN = SOUP07.select('.nv-maskrow > span[data-slot="tooltip-trigger"]')[1]
BLEND = BeautifulSoup(M.BLEND_CARD, "html.parser")
SLIDER = BLEND.select_one("[data-slider]")
RESET_LINK = BLEND.select_one(".ym-card__header button")

_uid = [0]


def uid():
    _uid[0] += 1
    return f"sk{_uid[0]}"


def slider_row(label, value, lo, hi, default, unit="", sub=False, disabled=False, bipolar=False):
    """The app's slider row (label, slider, number field, Reset when changed)."""
    r = copy.copy(SLIDER)
    r = BeautifulSoup(str(SLIDER), "html.parser").select_one("[data-slider]")
    i = uid()
    lab = r.select_one("span.text-\\[0\\.8125rem\\]")
    lab["id"] = i
    lab.string = label
    f = (value - lo) / (hi - lo)
    rng = r.select_one('[data-slot="slider-range"]')
    if bipolar:
        z = (0 - lo) / (hi - lo)
        a, b = min(f, z), max(f, z)
        rng["style"] = f"inset-inline-start: {a * 100:.2f}%; width: {(b - a) * 100:.2f}%;"
    else:
        rng["style"] = f"inset-inline-start: 0%; width: {f * 100:.2f}%;"
    thumb = r.select_one('[data-slot="slider-thumb"]')
    thumb.parent["style"] = f"transform: var(--radix-slider-thumb-transform); position: absolute; left: calc({f * 100:.2f}% + {8 - 16 * f:.2f}px);"
    thumb["aria-labelledby"] = i
    thumb["aria-valuemin"], thumb["aria-valuemax"], thumb["aria-valuenow"] = str(lo), str(hi), str(value)
    thumb["aria-valuetext"] = f"{value}{unit}"
    inp = r.select_one("input")
    inp["value"] = f"{value:+d}" if (bipolar and value > 0) else f"{value}{unit}"
    inp["aria-label"] = f"{label} value"
    changed = value != default
    if changed:
        inp["data-changed"] = ""
        last = r.select_one("div.flex.justify-end")
        btn = BeautifulSoup(str(RESET_LINK), "html.parser").find("button")
        btn["aria-label"] = f"Reset {label.lower()}"
        last.append(btn)
    html = str(r)
    cls = "sk-sub" if sub else ""
    dis = ' aria-disabled="true"' if disabled else ""
    return f'<div class="sk-row {cls}"{dis}>{html}</div>'


def switch(on, label, hint=None, small=False, disabled=False):
    h = f'<span class="sk-switch__hint">{hint}</span>' if hint else ""
    dis = ' aria-disabled="true"' if disabled else ""
    return (
        f'<div class="ym-switch-row sk-switchrow"{dis}><span class="ym-switch-row__label">{label}{h}</span>'
        f'<button class="sk-switch" role="switch" aria-checked="{"true" if on else "false"}" aria-label="{label}" type="button"><span class="sk-switch__thumb"></span></button></div>'
    )


def segmented(labels, active, width, label):
    s = BeautifulSoup(str(SEG), "html.parser").select_one("[data-slot=segmented-control]")
    items = s.select('[data-slot="segmented-control-item"]')
    proto = str(items[0])
    for it in items:
        it.decompose()
    for k, t in enumerate(labels):
        b = BeautifulSoup(proto, "html.parser").find("button")
        b.string = t
        b["data-state"] = "on" if k == active else "off"
        b["aria-checked"] = "true" if k == active else "false"
        b["data-value"] = t.lower()
        s.append(b)
    s["aria-label"] = label
    iw = (width - 4) / len(labels)
    s["style"] = f"outline:none; width:{width}px; --indicator-x:{2 + active * iw:.1f}px; --indicator-y:2px; --indicator-w:{iw:.1f}px; --indicator-h:{'32' if True else ''}px; --indicator-r:9997px; --indicator-opacity:1;"
    return str(s)


def eye_toggle(on, label="Show masks"):
    e = BeautifulSoup(str(EYE_BTN), "html.parser")
    b = e.find("button")
    b["aria-label"] = label
    b["aria-pressed"] = "true" if on else "false"
    b["data-state"] = "on" if on else "off"
    e.select_one(".sr-only").string = "Show where the tools work on this face" if not on else "Hide the masks"
    if on:
        b["class"] = b.get("class", []) + ["sk-eye-on"]
    return str(e)


def link_button(text, disabled=False):
    b = BeautifulSoup(str(RESET_LINK), "html.parser").find("button")
    b.string = text
    if disabled:
        b["aria-disabled"] = "true"
    return str(b)


def sec_button(text, icon=None, disabled=False):
    """A small glass button, as the header's Copy settings (variant header, size md)."""
    b = BeautifulSoup(M.FOOT, "html.parser").find("button")
    b.string = text
    b["aria-describedby"] = ""
    if disabled:
        b["aria-disabled"] = "true"
    elif b.has_attr("aria-disabled"):
        del b["aria-disabled"]
    return str(b)


# ------------------------------------------------------------------------------------------------------------------------------ faces
FAMILY_N = len(F.FAMILY)
SMALL = {4}  # in the family scene, face 5 is too small


def crop(scene, i, size=52, small=False, selected=False, num=None, dim=False):
    cx, cy, h = F.geom(scene, i)
    side = 1.55 * h
    k = size / side
    x = -(cx - side / 2) * k
    y = -(cy + 0.02 * h - side / 2) * k
    st = f"background-image:url({AS}scene-{scene}.svg);background-size:{F.W * k:.1f}px {F.H * k:.1f}px;background-position:{x:.1f}px {y:.1f}px;width:{size}px;height:{size}px"
    flags = (" data-selected" if selected else "") + (" data-small" if small else "")
    n = i + 1 if num is None else num
    return (
        f'<button class="sk-face" type="button" role="radio" aria-checked="{"true" if selected else "false"}" aria-label="Face {n}{", too small" if small else ""}"{flags}>'
        f'<span class="sk-face__pic" style="{st}"></span><span class="sk-face__num">{n}</span></button>'
    )


def faces_card(scene, n, selected, same, found_state="ok", small=(), many=False, overlay_hint=None):
    """The Faces card. found_state: ok | none | finding."""
    head = f'<div class="ym-card__header"><h3 class="ym-card__title">Faces</h3>'
    if found_state == "ok":
        head += f'<span class="sk-count">{n} found</span>'
    elif found_state == "finding":
        head += '<span class="sk-count">Finding…</span>'
    head += "</div>"
    if found_state == "none":
        body = (
            '<div class="sk-empty"><span class="sk-empty__ico">' + svg(FACE_ICON + '<path d="m4 4 16 16" opacity=".0"/>', 22) + '</span>'
            '<p class="sk-empty__title">No faces found in this photo.</p>'
            '<p class="ym-row-note" style="margin:6px 0 0">Skin works on faces that are at least 200 px across and turned towards the camera. Try another photo, or delete this node. It changes nothing until it finds a face.</p></div>'
        )
        return f'<section class="ym-card" data-last="true">{head}{body}</section>'
    if found_state == "finding":
        body = (
            '<div class="sk-finding"><span class="sk-spin"></span><div class="sk-finding__txt"><p class="sk-empty__title" style="margin:0">Finding faces…</p>'
            '<p class="ym-row-note" style="margin:2px 0 0">The first time takes a few seconds. The photo stays as it is until then.</p></div></div>'
            '<div class="sk-prog" role="progressbar" aria-label="Finding faces"><span class="sk-prog__bar"></span></div>'
        )
        return f'<section class="ym-card" data-last="true">{head}{body}</section>'
    pics = "".join(crop(scene, i if i < len(F.SCENES[scene][0]) else i % len(F.SCENES[scene][0]), 52, small=i in small, selected=(i == selected and not same), num=i + 1) for i in range(n))
    cls = "sk-faces" + (" sk-faces--scroll" if n > 10 else "")
    grid = f'<div class="{cls}" role="radiogroup" aria-label="Faces" data-all="{"true" if same else "false"}">{pics}</div>'
    extra = ""
    if n > 1:
        extra = switch(same, "Same for all faces")
    return f'<section class="ym-card" data-last="true">{head}{grid}{extra}</section>'


def face_card(scene, n, selected, same, values, neck=(False, False), masks_on=False, small=False, disabled=False, copy_btn=True):
    """The controls of the selected face (or of all faces)."""
    title = "All faces" if same else f"Face {selected + 1}"
    if n == 1:
        title = "Face"
    bl, dc, es, tx = values
    dis = disabled or small
    reset = link_button("Reset", disabled=dis)
    head = f'<div class="ym-card__header"><h3 class="ym-card__title">{title}</h3>{eye_toggle(masks_on) if not dis else ""}{reset}</div>'
    note = ""
    if small:
        note = (
            '<div class="sk-note" role="note">' + svg('<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>', 16) +
            '<p><b>This face is too small to retouch.</b> Faces need to be at least 200 px across in the full photo.</p></div>'
        )
    rows = (
        slider_row("Blemish", bl, 0, 100, 30, disabled=dis)
        + switch(neck[0], "Include neck", disabled=dis).replace("sk-switchrow", "sk-switchrow sk-sub")
        + slider_row("Dark circles", dc, 0, 100, 30, disabled=dis)
        + slider_row("Even skin", es, 0, 100, 25, disabled=dis)
        + slider_row("Texture", tx, -100, 100, 0, sub=True, disabled=dis, bipolar=True)
        + switch(neck[1], "Include neck", disabled=dis).replace("sk-switchrow", "sk-switchrow sk-sub")
    )
    legend = ""
    if masks_on:
        legend = (
            '<div class="sk-legend" aria-label="Mask colours"><span><i style="background:#ff6fd2"></i>Skin</span><span><i style="background:#58d4ff"></i>Under the eyes</span>'
            '<span><i style="background:none;box-shadow:inset 0 0 0 2px #9be56a"></i>Even skin</span></div>'
        )
    foot = ""
    if n > 1 and not same and not dis and copy_btn:
        foot = f'<div class="sk-foot">{sec_button("Copy to other faces")}</div>'
    return f'<section class="ym-card" data-last="{"false" if True else "true"}">{head}{note}<div class="sk-rows"{" data-disabled" if dis else ""}>{rows}</div>{legend}{foot}</section>'


def protect_card(active=False, strokes=0, disabled=False):
    sub = "Paint over moles, freckles and scars you want to keep." if strokes == 0 else f"{strokes} stroke{'s' if strokes != 1 else ''}. Blemish leaves painted areas alone."
    if active:
        sub = "Paint on the photo. Hold ⌥ to erase."
    seg = segmented(["Paint", "Erase"], 0, 168, "Protect tool")
    clear = link_button("Clear", disabled=strokes == 0 or disabled)
    tool = f'<div class="sk-tool"><span class="ym-field__label" style="font-size:13px;font-weight:700">Brush</span>{seg}</div>'
    return (
        f'<section class="ym-card" data-last="true"{" aria-disabled=\"true\"" if disabled else ""}><div class="ym-card__header"><h3 class="ym-card__title">Protect</h3>{clear}</div>'
        f'<p class="ym-row-note" style="margin:0 0 12px">{sub}</p>{tool}'
        f'{slider_row("Size", 40, 1, 100, 40)}'
        f'</section>'
    )


def cards(scene, n, selected, same, values, neck=(False, False), masks_on=False, found="ok", small=(), protect=0, protect_active=False):
    fc = faces_card(scene, n, selected, same, found, small=small)
    if found != "ok":
        return fc
    isl = selected in small and not same
    return fc + face_card(scene, n, selected, same, values, neck, masks_on, small=isl) + protect_card(protect_active, protect, disabled=isl)


# ------------------------------------------------------------------------------------------------------------------------------ panel
def tabs2(active=0):
    t = M.TABLIST.replace('data-tab="raw"', 'data-tab="look"').replace(">Blend</button>", ">Skin</button>").replace("ym-tab-raw-blend", "ym-tab-look-skin").replace("ym-tabpanel-raw", "ym-tabpanel-look")
    return t


def aside2(headhtml, cardshtml, inert=False):
    return (
        f'<aside aria-label="Tools" class="ym-tools" data-tab="look">{headhtml}{tabs2()}'
        f'<div aria-labelledby="ym-tab-look-skin" class="ym-tools__scroll" data-tab="look" id="ym-tabpanel-look" role="tabpanel">{cardshtml}</div>{M.FOOT}</aside>'
        '<div aria-live="polite" class="sr-only"></div></div></div>'
    )


def head(line, bypass_on=False):
    h = M.head("skin", "Skin", line, pressed_on=bypass_on)
    return re.sub(r'<span class="ym-nodehead__chip">.*?</span>', '<span class="ym-nodehead__chip" style="font-size:12px;font-weight:800">02</span>', h, count=1, flags=re.S)


# ------------------------------------------------------------------------------------------------------------------------------ the page
def page(title, inner, extra_css=""):
    return (
        f'<!doctype html>\n{M.HTML_TAG}<head>\n<meta charset="utf-8"/>\n<meta content="width=device-width, initial-scale=1.0" name="viewport"/>\n<meta content="dark" name="color-scheme"/>\n'
        f'<title>Yellowmoon · {title}</title>\n<link href="../../../mixers/html/app.css" rel="stylesheet"/>\n<link href="../../../nodes/html/nv.css" rel="stylesheet"/>\n'
        f'<link href="../../../mixers/html/mixers.css" rel="stylesheet"/>\n<link href="../skin.css" rel="stylesheet"/>\n</head>\n{inner}\n{M.SCRIPT}</body></html>\n'
    )


def write(name, html):
    with open(os.path.join(HERE, "pages", name), "w", encoding="utf-8") as f:
        f.write(html)
    print("wrote", name, len(html))


def skin_card(g, key, x, y, num, thumb, selected=False, bypass=False, dead=False, name="Skin"):
    """A Skin node card: the corrector's card with the face glyph in the footer where a mask glyph would be."""
    g.corrector(key, name, x, y, num, thumb, masks=("mask",), selected=selected, bypass=bypass, dead=dead)
    h = g.html[-1]
    h = h.replace(f'data-glyph="mask">{M.HALF}', f'data-glyph="skin">{svg(FACE_ICON, 14)}')
    g.html[-1] = h


def photo_prefix(scene, narrow=False, fit=28, name="Portraits", sub="2026-10-04 · 7 photos", overlay="", camera="Sony A1 · A1M02214 · ISO 200 · 1/250 · f/2.0", body=None):
    pre = M.prefix_of(body or M.BODY06, narrow, fit)
    pre = re.sub(r'(class="ym-photo-cell__image"[^>]*?src=")[^"]*(")', rf"\g<1>{AS}scene-{scene}.svg\g<2>", pre)
    pre = pre.replace("Provence", name).replace("2026-10-04 · 7 photos", sub)
    pre = re.sub(r"Sony A1 · A1M\d+ · ISO \d+ · 1/\d+ · f/[\d.]+", camera, pre)
    pre = pre.replace('aria-label="Photo A1M01739, Signature · 100%"', 'aria-label="Photo A1M02214, Signature · 100%"')
    # the overlay goes inside the photo's view, after the image
    pre = re.sub(r'(<img alt="" class="ym-photo-cell__image"[^>]*/>)', lambda m_: m_.group(1) + overlay, pre, count=1)
    return pre


def thumb(scene):
    return AS + f"scene-{scene}.svg"


def graph(scene, selected=True, bypass=False, dead=False, n_nodes=2):
    g = M.Graph()
    mid = 160
    xs = [120, 334]
    g.marker("input", "Input", xs[0] - 86, mid)
    g.corrector("main", "Main", xs[0], mid - 56, "01", thumb(scene))
    skin_card(g, "skin", xs[1], mid - 56, "02", thumb(scene), selected=selected, bypass=bypass, dead=dead)
    g.marker("output", "Output", xs[1] + 150 + 86, mid)
    for a, b in (("input", "main"), ("main", "skin"), ("skin", "output")):
        g.link(a, b)
    return g


def node_page(name, title, scene, headline, cardshtml, sel_face=1, show_faces=True, overlay_extra="", bar_mask_on=False, fit=28, bypass_on=False, tag=None):
    n = len(F.SCENES[scene][0])
    over = (F.outline(scene, None, sel_face) if show_faces else "") + overlay_extra
    pre = photo_prefix(scene, False, fit, overlay=over)
    if tag:
        pre = pre.replace('<span class="', f'<span class="sk-tag">{tag}</span><span class="', 1) if False else pre
    g = graph(scene, bypass=bypass_on)
    ed = M.editor(g.render(h=327), M.BAR06, False, mask_off=False)
    inner = pre + ed + aside2(head(headline, bypass_on), cardshtml)
    write(name, page(title, inner))


# ---- 01 the Add node menu -----------------------------------------------------------------------------------------------------
def add_menu(highlight="Skin"):
    items = [
        ("Corrector", "corrector", "Grades all or part of the photo."),
        ("Film", "film", "Models a colour negative, a print and a scan."),
        ("Skin", "skin", "Retouches the skin of each face."),
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


def page01():
    scene = "portrait"
    g = M.Graph()
    mid = 160
    xs = [120, 334, 548]
    g.marker("input", "Input", xs[0] - 86, mid)
    g.corrector("main", "Main", xs[0], mid - 56, "01", thumb(scene))
    g.corrector("c2", "Warmth", xs[1], mid - 56, "02", thumb(scene), selected=True)
    g.marker("output", "Output", xs[1] + 150 + 86, mid)
    for a, b in (("input", "main"), ("main", "c2"), ("c2", "output")):
        g.link(a, b)
    pre = photo_prefix(scene, False, 28, name="Portraits")
    aside06 = M.BODY06[M.BODY06.find("<aside"): M.BODY06.find("</aside>") + 8].replace('src="../assets/', 'src="' + A).replace("Mountain", "Warmth").replace("Corrector · 2 masks", "Corrector · whole photo")
    tail = '<div aria-live="polite" class="sr-only"></div></div></div>'
    inner = pre + M.editor(g.render(h=327), M.BAR06, False, add_open=True, mask_off=False) + aside06 + tail + add_menu("Skin")
    inner += '<div class="sk-callout" style="left:640px;top:610px;max-width:360px"><b>Node view only.</b> Skin is not in the simple view (the owner, 2026-11-11).</div>'
    write("01.html", page("Add node menu", inner))


# ---- 02 the simple view of a photo that has a Skin node ------------------------------------------------------------------------
def page02():
    s = open(os.path.join(M.NODES, "04.html"), encoding="utf-8").read()
    s = s.replace('src="../assets/', 'src="' + A)
    s = s.replace('src="' + A + 'photo-graded.jpg"', f'src="{AS}scene-family.svg"')
    s = s.replace("This photo has 4 nodes.", "This photo has 2 nodes.")
    s = s.replace("Provence", "Portraits")
    s = s.replace(A + "strip-2-graded.jpg", AS + "scene-family.svg")
    s = re.sub(r"Sony A1 · A1M\d+ · ISO \d+ · 1/\d+ · f/[\d.]+", "Sony A1 · A1M02214 · ISO 200 · 1/250 · f/2.0", s)
    s = re.sub(r'(<link[^>]*href=")\.\./(app\.css|nv\.css)', lambda m_: m_.group(1) + ("../../../mixers/html/app.css" if m_.group(2) == "app.css" else "../../../nodes/html/nv.css"), s)
    # the same extra css as every page
    s = s.replace("</head>", '<link href="../../../mixers/html/mixers.css" rel="stylesheet"/><link href="../skin.css" rel="stylesheet"/></head>', 1)
    s = s.replace("<title>Yellowmoon</title>", "<title>Yellowmoon · Simple view</title>")
    # a design note over the photo (not part of the product)
    s = s.replace("</body>", '<div class="sk-callout" style="left:24px;top:132px;max-width:380px"><b>Simple view.</b> It has no Skin controls. The photo still shows the retouching; to change it, open the nodes.</div></body>', 1)
    # keep the link tags of nv.css only once if the page had them
    write("02.html", s)


# ---- 03 / 04 / 05 the Skin node ------------------------------------------------------------------------------------------------
def page03():
    # four faces and a small one; Face 2 selected, per face; Blemish changed on this face
    n = len(F.FAMILY)
    ch = cards("family", n, 1, False, (45, 30, 25, 0), small=SMALL, protect=0)
    node_page("03.html", "Skin node, four faces", "family", "Skin · 5 faces", ch, sel_face=1)


def page04():
    # one face, the masks shown
    ch = cards("portrait", 1, 0, False, (30, 30, 25, 0), neck=(False, False), masks_on=True)
    over = F.mask_overlay("portrait", 0)
    node_page("04.html", "Skin node, one face, masks shown", "portrait", "Skin · 1 face", ch, sel_face=0, overlay_extra=over, show_faces=False)


def page05():
    # the protect brush active
    ch = cards("portrait", 1, 0, False, (30, 30, 25, 0), protect=2, protect_active=True)
    over = F.protect_overlay("portrait", 0)
    node_page("05.html", "Skin node, protect brush", "portrait", "Skin · 1 face", ch, sel_face=0, overlay_extra=over, show_faces=False)


def page06():
    # many faces with Same for all faces on
    ch = cards("group12", 12, 0, True, (30, 30, 25, 0))
    node_page("06.html", "Skin node, twelve faces", "group12", "Skin · 12 faces", ch, sel_face="all")


# ---- states ------------------------------------------------------------------------------------------------------------------
def sheet_card(caption, body, note=""):
    n = f"<small>{note}</small>" if note else ""
    return f'<figure class="sk-cell"><div class="sk-cell__stage sk-panel ym-tools" data-tab="look" style="width:400px">{body}</div><figcaption>{caption}{n}</figcaption></figure>'


def panel_fragment(headline, cardshtml, bypass=False):
    return head(headline, bypass) + f'<div class="ym-tools__scroll" data-tab="look" style="overflow:visible;height:auto">{cardshtml}</div>'


def node_cell(caption, note="", **kw):
    g = M.Graph()
    skin_card(g, "s", 47, 90, "02", thumb("family"), **kw)
    return f'<figure class="mx-cell" style="width:250px"><div class="mx-cell__stage" style="height:230px">{g.render(250, 230)}</div><figcaption>{caption}{("<small>" + note + "</small>") if note else ""}</figcaption></figure>'


def states_page():
    cells = [
        node_cell("Normal", "numbered like Film; the face glyph sits where mask glyphs do"),
        node_cell("Selected", selected=True),
        node_cell("Bypassed", "half-strength thumbnail and the slash", bypass=True),
        node_cell("Dead", "not on a path to the Output: 45 %", dead=True),
    ]
    many = faces_card("group12", 24, 0, False)
    panels = [
        ("1 face", "Skin · 1 face", cards("portrait", 1, 0, False, (30, 30, 25, 0)), "no list of faces, no Same for all faces"),
        ("4 faces, per face (Face 2 selected)", "Skin · 4 faces", cards("family", 4, 1, False, (45, 30, 25, 0)), "Blemish was changed on this face: its field is yellow and has a Reset"),
        ("4 faces, Same for all faces on", "Skin · 4 faces", cards("family", 4, 0, True, (30, 30, 25, 0)), "no face is highlighted: the controls act on every face"),
        ("12 faces", "Skin · 12 faces", cards("group12", 12, 3, False, (30, 30, 25, 0)), "the list shows about two and a half rows and scrolls"),
        ("No faces found", "Skin · no faces found", cards("family", 0, 0, False, (30, 30, 25, 0), found="none"), "no controls: nothing to set"),
        ("Finding faces", "Skin · finding faces", cards("family", 0, 0, False, (30, 30, 25, 0), found="finding"), "controls wait; the photo is unchanged"),
        ("A face too small", "Skin · 5 faces", cards("family", 5, 4, False, (30, 30, 25, 0), small=SMALL), "face 5 is dimmed in the list; its controls are off"),
        ("Masks shown, neck on", "Skin · 1 face", cards("portrait", 1, 0, False, (60, 30, 40, 0), neck=(True, True), masks_on=True), "the legend names the three tints"),
    ]
    pcells = []
    for cap, line, ch, note in panels:
        pcells.append(sheet_card(cap, panel_fragment(line, ch, False), note))
    pcells.append(sheet_card("24 faces, the list on its own", f'<div class="ym-tools__scroll" style="overflow:visible;height:auto">{many}</div>', "up to 32 faces; the list scrolls"))
    tips = [
        ("Add node › Skin", "Retouches the skin of each face."),
        ("Add node › Skin, at 64 nodes", "A photo can have up to 64 nodes."),
        ("A face in the list", "Face 2. Click to retouch it on its own."),
        ("A face in the list that is too small", "Face 5 is too small to retouch."),
        ("Show masks (the eye)", "Show where the tools work on this face"),
        ("Show masks, while on", "Hide the masks"),
        ("Include neck (Blemish, Even skin)", "Also retouch the neck. Turn it off for a scarf, a collar or a beard."),
        ("Same for all faces", "One set of controls for every face. Turn it off to set each face on its own."),
        ("Copy to other faces", "Gives every other face these settings."),
        ("Protect › Brush", "Paint over moles, freckles and scars you want to keep."),
        ("Reset (the head)", "Reset this node. Masks and protection stay."),
        ("Show mask on a Skin node (the toolbar)", "Shows where the node acts, as on a corrector."),
        ("Skin in the simple view", "This photo has 2 nodes. These controls change the input, the output and node 1, Main."),
    ]
    tip_html = "".join(f'<div class="mx-tiprow"><span>{a}</span>{M.tip(b)}</div>' for a, b in tips)
    inner = (
        '<body><div class="mx-sheet sk-sheet">'
        '<h1 class="mx-sheet__title">Skin node · states</h1>'
        '<p class="mx-sheet__sub">The panel is 400 px wide, as in the app. Faces are drawings made for this folder, not photographs.</p>'
        f'<section><h2>Node card</h2><div class="mx-row">{"".join(cells)}</div></section>'
        f'<section><h2>Panel</h2><div class="mx-row sk-grid">{"".join(pcells)}</div></section>'
        f'<section><h2>Copy for tooltips and notes (drafts)</h2><div class="mx-tips">{tip_html}</div></section>'
        "</div>"
    )
    write("states.html", page("Skin states", inner))


if __name__ == "__main__":
    F.write_assets()
    page01()
    page02()
    page03()
    page04()
    page05()
    page06()
    states_page()
