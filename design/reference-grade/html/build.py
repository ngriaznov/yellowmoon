#!/usr/bin/env python3
"""Builds the pages of design/reference-grade/html/pages/ (task RG-07). Same method as design/blur/html/build.py and design/colour-uniform/html/build.py, which this
file imports for the shared pieces (the app's captured DOM and CSS: header, photo cell, toolbar, panel head, tabs, cards, sliders, the graph). It adds the Reference
Grade node (kind `reference_grade`, docs/34 section 9): the Reference, Hue and Elements tabs. The pictures are the synthetic scene of rg_demo.py with the maths of
docs/34 applied by rg_ref.py with ILLUSTRATIVE values (no fit was run). Run: python3 build.py (needs bs4). Not product code.
"""
import importlib.util
import math
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

# ---------------------------------------------------------------------------------------------------------------- glyph and icons
# The card's glyph (proposal): a picture beside a picture, with a grade curve in the front one.
RG_ICON = '<path d="M8 3h11a2 2 0 0 1 2 2v11"/><rect x="3" y="8" width="14" height="13" rx="2"/><path d="M6.2 18c2.4 0 2.4-6 4.8-6s2.4 3.6 3.6 3"/>'
M.ICON["reference_grade"] = RG_ICON
IMAGES = '<rect x="3" y="3" width="14" height="14" rx="2"/><path d="M7 21h12a2 2 0 0 0 2-2V7"/><circle cx="8" cy="8" r="1.4"/><path d="m17 13-3.3-3.3a1 1 0 0 0-1.4 0L4 17.5"/>'
FILE = '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5"/>'
ALERT = '<circle cx="12" cy="12" r="10"/><path d="M12 8v5"/><path d="M12 16.5h.01"/>'
INFO = C.INFO
CHEV = '<path d="m6 9 6 6 6-6"/>'
EYE = S.EYE

NAME = "Match"
REF_NAME = "harbour-dusk.jpg"

# illustrative values (rg_demo.SETTINGS); the panel shows the stored numbers
EL = {
    "sky": dict(share=26, cap=40, exposure_ev=-0.45, hue_deg=-14, chroma=125),
    "vegetation": dict(share=29, cap=35, exposure_ev=-0.25, hue_deg=20, chroma=75),
    "water": dict(share=0.6, cap=35, exposure_ev=0, hue_deg=0, chroma=100),
    "ground": dict(share=19, cap=30, exposure_ev=0.10, hue_deg=-6, chroma=130),
    "built": dict(share=8, cap=25, exposure_ev=0, hue_deg=-8, chroma=125),
    "skin": dict(share=3.3, cap=15, exposure_ev=0, hue_deg=-3, chroma=95),
    "subject": dict(share=18, exposure_ev=0.25),
}
TITLES = {"sky": "Sky", "vegetation": "Vegetation", "water": "Water", "ground": "Ground", "built": "Built", "skin": "Skin", "subject": "Subject"}


# ---------------------------------------------------------------------------------------------------------------- rows and buttons
def srow(label, value, lo, hi, default, text, disabled=False, origin=None, track=None):
    """C.srow with an origin for the filled part: the fill runs from the default's place to the value (a bipolar slider)."""
    r = BeautifulSoup(str(S.SLIDER), "html.parser").select_one("[data-slider]")
    i = C.uid()
    lab = r.select_one("span.text-\\[0\\.8125rem\\]")
    lab["id"] = i
    lab.string = label
    f = (value - lo) / (hi - lo)
    rng = r.select_one('[data-slot="slider-range"]')
    if origin is None:
        rng["style"] = f"inset-inline-start: 0%; width: {f * 100:.2f}%;"
    else:
        z = (origin - lo) / (hi - lo)
        a, b = min(f, z), max(f, z)
        rng["style"] = f"inset-inline-start: {a * 100:.2f}%; width: {(b - a) * 100:.2f}%;"
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


def glass_button(text, icon=None, disabled=False, pressed=None, chevron=False, cls=""):
    b = BeautifulSoup(M.FOOT, "html.parser").find("button")
    b.clear()
    if icon:
        b.append(BeautifulSoup(svg(icon, 16), "html.parser"))
    b.append(BeautifulSoup(f"<span>{text}</span>", "html.parser"))
    if chevron:
        b.append(BeautifulSoup(svg(CHEV, 14), "html.parser"))
    b["aria-describedby"] = ""
    if disabled:
        b["aria-disabled"] = "true"
    elif b.has_attr("aria-disabled"):
        del b["aria-disabled"]
    if pressed is not None:
        b["aria-pressed"] = "true" if pressed else "false"
        b["data-state"] = "on" if pressed else "off"
    b["class"] = b.get("class", []) + ["rg-btn"] + ([cls] if cls else [])
    return str(b)


_EXPORT = BeautifulSoup(open(os.path.join(M.NODES, "04.html"), encoding="utf-8").read(), "html.parser")
_EXPORT_BTN = [b for b in _EXPORT.select("button") if b.get_text(strip=True).startswith("Export")][0]


def prominent(text, disabled=False, accent=False):
    """glasscn's prominent button (the Export button's markup); `accent` adds a ring for the stale fit."""
    b = BeautifulSoup(str(_EXPORT_BTN), "html.parser").find("button")
    b.clear()
    b.append(BeautifulSoup(f"<span>{text}</span>", "html.parser"))
    for a in ("aria-describedby", "title"):
        if b.has_attr(a):
            del b[a]
    if disabled:
        b["aria-disabled"] = "true"
    b["class"] = b.get("class", []) + ["rg-fit"] + (["rg-fit--accent"] if accent else [])
    return str(b)


def progress(pct, label="Fit progress"):
    return f'<div class="sk-prog rg-prog" role="progressbar" aria-label="{label}" aria-valuemin="0" aria-valuemax="100" aria-valuenow="{pct}"><span class="sk-prog__bar" style="width:{pct}%;margin-left:0"></span></div>'


def tooltip(text, style=""):
    return f'<span class="rg-tooltip glass-liquid" role="tooltip" style="{style}">{text}</span>'


# ---------------------------------------------------------------------------------------------------------------- the Reference card
STATUS = {
    "none": "Choose a picture to match.",
    "ready": "Press Fit to match the photo to this picture.",
    "scene": "Finding what is in the pictures…",
    "fit1": "Fitting… 1 of 3",
    "fit2": "Fitting… 2 of 3",
    "fit3": "Fitting… 3 of 3",
    "fitted": "Fitted to the reference.",
    "nochange": "The photo is already close to the reference. Nothing was changed.",
    "edited": "Edited by hand since the fit. Fit again to start over.",
    "look": "The look changed since the fit. Fit again to match it with the new look.",
    "kept": "The values are kept. Choose a reference to fit again.",
}
NOTES = {
    "spatial": "Nodes after this one that blur or add film aren't part of the fit.",
    "model": "The scene model is missing from this install, so element grades aren't applied. Reinstall Yellowmoon.",
    "bars": "Black bars were removed from the reference.",
    "profile": "The picture's colour profile couldn't be read, so it is taken as sRGB.",
    "grey": "This picture has no colour. The photo will be matched in black and white.",
    "noskin": "No skin was found, so Skin protection did nothing.",
}
ERRORS = {
    "unreadable": "This file can't be read. It may be damaged. Choose another picture.",
    "format": "Yellowmoon reads JPEG, PNG and TIFF pictures. Save this one as one of those, then choose it again.",
    "small": "This picture is too small to match. Use one that is at least <i>N</i> pixels on its short side.",
    "bars": "There's no picture inside the black bars. Choose another picture.",
    "same": "That's the photo you're grading. Choose another.",
    "failed": "The grade couldn't be fitted. Try another reference.",
    "model": "The scene model is missing from this install. Reinstall Yellowmoon.",
}


def thumb_box(src=None, name=None, tall=128):
    if src:
        pic = f'<img alt="The reference picture" src="{AS}{src}" style="height:{tall}px"/>'
        cap = f'<p class="rg-thumb__name">{name}</p>'
    else:
        pic = f'<span class="rg-thumb__empty">{svg(IMAGES, 22)}</span>'
        cap = '<p class="rg-thumb__name rg-thumb__name--empty">No reference</p>'
    return f'<div class="rg-thumb{"" if src else " rg-thumb--empty"}" style="height:{tall}px">{pic}</div>{cap}'


def status_block(key, notes=(), error=None):
    n = "".join(f'<p class="ym-row-note rg-note">{svg(INFO, 14)}<span>{NOTES[k]}</span></p>' for k in notes)
    e = f'<div class="sk-note rg-error" role="alert">{svg(ALERT, 16)}<p>{ERRORS[error]}</p></div>' if error else ""
    return f'<div class="rg-statusgrp" role="status">{e}<p class="ym-row-note rg-status">{STATUS[key]}</p>{n}</div>'


def ref_card(state="none", strength=100, skin=True, notes=(), error=None, menu_open=False, chooser=False, name=REF_NAME, thumb="reference-thumb.jpg", tip=None, status=None, pct=0, compact=False, show_strength=True):
    """state: none | ready | scene | fit | fitted | edited | look | nochange | kept (the status key is the same unless `status` is given)."""
    has_ref = state in ("ready", "scene", "fit", "fitted", "edited", "look", "nochange")
    fitting = state in ("scene", "fit")
    fitted = state in ("fitted", "edited", "look", "nochange")
    key = status or {"none": "none", "ready": "ready", "scene": "scene", "fit": "fit2", "fitted": "fitted", "edited": "edited", "look": "look", "nochange": "nochange", "kept": "kept"}[state]
    top = thumb_box(thumb if has_ref else None, name if has_ref else None, tall=96 if compact else 128)
    choose = glass_button("Choose reference…", icon=IMAGES, disabled=fitting, pressed=(True if (menu_open or chooser) else None), chevron=True)
    if fitting:
        fitrow = f'<div class="rg-fitrow rg-fitrow--busy">{progress(pct or (8 if state == "scene" else 55))}{glass_button("Cancel")}</div>'
    else:
        label = "Fit again" if fitted else "Fit"
        dis = not has_ref
        fb = prominent(label, disabled=dis, accent=(state == "look")) if (not fitted or state == "look") else glass_button(label, cls="rg-fit")
        fitrow = f'<div class="rg-fitrow">{choose}{fb}</div>'
    if fitting:
        fitrow = f'<div class="rg-fitrow">{choose}</div>' + fitrow
    tp = tooltip("Choose a reference first.", "left:206px;top:178px") if tip == "fit" else ""
    strength_row = ""
    if show_strength:
        strength_row = (
            f'<div class="sk-rows rg-strength">{srow("Strength", strength, 0, 100, 100, f"{strength}%")}</div>'
            '<p class="ym-row-note rg-line">Scales the fitted values. The tabs show them before Strength.</p>'
        )
    sw = S.switch(skin, "Skin protection", "Keeps skin tones from drifting. Applies at the next fit.")
    return (
        f'<section class="ym-card rg-card" data-last="true">{C.head_of("Reference", None, "", "")}{top}{fitrow}{tp}{strength_row}{sw}{status_block(key, notes, error)}</section>'
    )


# ---------------------------------------------------------------------------------------------------------------- the Hue tab
RAINBOW = "linear-gradient(90deg, #ff4d4d, #ffb23a 17%, #e9e24a 25%, #66e060 33%, #2fe0c8 50%, #2fa4ff 62%, #8a63ff 75%, #ff5fd0 88%, #ff4d4d)"


def hue_plot(kind, pts, sel=1, read=None):
    """A periodic plot: x is the hue 0 to 360 with the rainbow track under it; y is -180 to +180 degrees, or a log scale 1/8 to 8 for saturation (detent at the middle)."""
    Wp, Hp, pl, pr, pt, pb = 352, 176, 34, 12, 12, 14
    px = lambda h: pl + h / 360.0 * (Wp - pl - pr)
    if kind == "hue":
        py = lambda v: pt + (180 - v) / 360.0 * (Hp - pt - pb)
        det = 0.0
        ticks = [(-180, "−180°"), (-90, "−90°"), (0, "0"), (90, "+90°"), (180, "+180°")]
        tx = lambda v: py(v)
    else:
        py = lambda v: pt + (3 - math.log2(max(v, 0.125))) / 6.0 * (Hp - pt - pb)
        det = 1.0
        ticks = [(0.125, "0"), (0.25, "¼"), (0.5, "½"), (1, "1"), (2, "2"), (4, "4"), (8, "8")]
        tx = py
    # a periodic, smooth curve through the points (cosine easing between the points, as rg_ref does)
    pts = sorted(pts)
    path = []
    N = 360
    for k in range(N + 1):
        h = k
        j = max([i for i, p in enumerate(pts) if p[0] <= h] or [-1])
        if j < 0:
            x0, y0 = pts[-1][0] - 360, pts[-1][1]
            x1, y1 = pts[0]
        else:
            x0, y0 = pts[j]
            x1, y1 = pts[j + 1] if j + 1 < len(pts) else (pts[0][0] + 360, pts[0][1])
        t = 0.5 - 0.5 * math.cos(math.pi * (h - x0) / (x1 - x0))
        path.append((h, y0 + (y1 - y0) * t))
    d = "M" + "L".join(f"{px(h):.1f} {py(v):.1f}" for h, v in path)
    grid = "".join(f'<path d="M{px(h):.1f} {pt}V{Hp - pb}" stroke="#fff" stroke-opacity=".10"/>' for h in range(60, 360, 60))
    grid += "".join(f'<path d="M{pl} {tx(v):.1f}H{Wp - pr}" stroke="#fff" stroke-opacity="{0.34 if v == det else 0.10}" {"stroke-dasharray=\"4 5\"" if v == det else ""}/>' for v, _ in ticks)
    labels = "".join(f'<text x="{pl - 5}" y="{tx(v) + 3.5:.1f}" fill="#fff" fill-opacity=".55" font-size="10" text-anchor="end">{t}</text>' for v, t in ticks)
    dots = ""
    for i, (h, v) in enumerate(pts):
        on = i == sel
        dots += f'<circle cx="{px(h):.1f}" cy="{py(v):.1f}" r="{6.2 if on else 4.4}" fill="{"var(--tab-curve)" if on else "#1c1c1e"}" stroke="var(--tab-curve)" stroke-width="2.2"/>'
    axis = "".join(f'<text x="{px(h):.1f}" y="{Hp + 26}" fill="#fff" fill-opacity=".6" font-size="10" text-anchor="{"start" if h == 0 else "end" if h == 360 else "middle"}">{h}°</text>' for h in (0, 60, 120, 180, 240, 300, 360))
    svg_ = (
        f'<svg class="rg-plot__svg" viewBox="0 0 {Wp} {Hp + 32}" width="{Wp}" height="{Hp + 32}" role="img" aria-label="{"Hue against hue" if kind == "hue" else "Hue against saturation"} curve, {len(pts)} points">'
        f'<rect x="{pl}" y="{pt}" width="{Wp - pl - pr}" height="{Hp - pt - pb}" rx="10" fill="#1d1d1f" stroke="#fff" stroke-opacity=".14"/>'
        f'{grid}{labels}<path d="{d}" fill="none" stroke="var(--tab-curve)" stroke-width="2.6" stroke-linejoin="round"/>{dots}'
        f'<foreignObject x="{pl}" y="{Hp - pb + 4}" width="{Wp - pl - pr}" height="7"><div xmlns="http://www.w3.org/1999/xhtml" style="width:100%;height:7px;border-radius:4px;background:{RAINBOW};opacity:.9"></div></foreignObject>'
        f'{axis}</svg>'
    )
    return svg_


def hue_card(kind, pts, sel=1, reset=True):
    title = "Hue vs hue" if kind == "hue" else "Hue vs saturation"
    h, v = pts[sel]
    if kind == "hue":
        read = f'Hue <b>{h}°</b> <span class="rg-read__sp"></span> Turn <b>{v:+g}°</b>'.replace("-", "−")
    else:
        read = f'Hue <b>{h}°</b> <span class="rg-read__sp"></span> Saturation <b>{round(v * 100)}%</b>'
    return (
        f'<section class="ym-card" data-last="false">{C.head_of(title, None, "", S.link_button("Reset") if reset else "")}'
        f'<div class="rg-plot">{hue_plot(kind, pts, sel)}</div><p class="rg-read">{read}</p></section>'
    )


HUE_PTS = [(0, 10.0), (60, -8.0), (120, 22.0), (180, 10.0), (240, -14.0), (300, 0.0)]
SAT_PTS = [(0, 1.15), (60, 1.25), (120, 0.80), (180, 1.30), (240, 1.20), (300, 0.80)]


def hue_cards(hue=HUE_PTS, sat=SAT_PTS, reset=True):
    return hue_card("hue", hue, 2, reset) + hue_card("sat", sat, 2, reset)


# ---------------------------------------------------------------------------------------------------------------- the Elements tab
def element_card(e, found=True, looking=False, eye_on=False, v=None, last=False):
    d = dict(EL[e])
    if v:
        d.update(v)
    share = d["share"]
    changed = (d["exposure_ev"] != 0) or (e != "subject" and (d["hue_deg"] != 0 or d["chroma"] != 100))
    if looking:
        tag = f'<span class="sk-count rg-looking"><i class="sk-spin rg-spin"></i>Looking at the photo…</span>'
    elif not found:
        tag = ""
    else:
        tag = f'<span class="sk-count">{share:g} % of the photo</span>'
    eye = ""
    if True:
        eye = S.eye_toggle(eye_on, f"Show the {TITLES[e].lower()} mask")
        eye = eye.replace("Show where the tools work on this face", f"Show where the {TITLES[e].lower()} grade acts").replace("Hide the skin areas", f"Hide the {TITLES[e].lower()} mask")
        if not found or looking:
            eye = eye.replace("<button", '<button aria-disabled="true"', 1)
    reset = S.link_button("Reset") if changed and not looking else ""
    rows = srow("Exposure (EV)", d["exposure_ev"], -1, 1, 0, f"{d['exposure_ev']:+.2f}".replace("-", "−") if d["exposure_ev"] else "0.00", origin=0)
    if e != "subject":
        cap = d["cap"]
        rows += srow("Hue", d["hue_deg"], -cap, cap, 0, (f"{d['hue_deg']:+d}°".replace("-", "−") if d["hue_deg"] else "0°"), origin=0)
        rows += srow("Chroma", d["chroma"], 50, 160, 100, f"{d['chroma']}%", origin=100)
    note = ""
    if not found and not looking:
        note = '<p class="ym-row-note rg-absent-note">Not found in this photo.</p>'
    flag = ' data-absent="true"' if (not found or looking) else ""
    return f'<section class="ym-card rg-el" data-last="{"true" if last else "false"}"{flag}>{C.head_of(TITLES[e], None, tag, "")[:-6]}{eye}{reset}</div>{note}<div class="sk-rows">{rows}</div></section>'


def element_cards(found=("sky", "vegetation", "ground", "built", "skin", "subject"), eye=None, looking=False, only=None):
    out = []
    order = ["sky", "vegetation", "water", "ground", "built", "skin", "subject"]
    for i, e in enumerate(order):
        if only and e not in only:
            continue
        out.append(element_card(e, found=(e in found), looking=looking, eye_on=(e == eye), last=(i == 6)))
    return "".join(out)


# ---------------------------------------------------------------------------------------------------------------- panel
TABS = [("reference", "Reference", "look"), ("light", "Light", "light"), ("color", "Colour", "color"), ("curve", "Curve", "curve"), ("hue", "Hue", "color"), ("elements", "Elements", "color"), ("mask", "Mask", "mask")]


def tablist(active="reference"):
    soup = BeautifulSoup(M.TABLIST, "html.parser")
    tl = soup.select_one('[data-slot="tabs-list"]')
    btns = tl.select('button[role="tab"]')
    proto = BeautifulSoup(str(btns[0]), "html.parser").find("button")
    for b in btns:
        b.decompose()
    tl["class"] = tl.get("class", []) + ["rg-tabs"]
    tl["data-tab"] = dict((k, c) for k, _, c in TABS)[active]
    for k, label, col in TABS:
        b = BeautifulSoup(str(proto), "html.parser").find("button")
        on = k == active
        b.string = label
        b["id"] = f"ym-tab-{k}-{label.lower()}"
        b["aria-controls"] = f"ym-tabpanel-{k}"
        b["aria-selected"] = "true" if on else "false"
        b["data-state"] = "active" if on else "inactive"
        b["data-tab"] = col
        b["tabindex"] = "0" if on else "-1"
        tl.append(b)
    return str(soup)


def aside(headhtml, cardshtml, active="reference", scroll_style=""):
    col = dict((k, c) for k, _, c in TABS)[active]
    return (
        f'<aside aria-label="Tools" class="ym-tools" data-tab="{col}">{headhtml}{tablist(active)}'
        f'<div aria-labelledby="ym-tab-{active}-{active}" class="ym-tools__scroll" data-tab="{col}" id="ym-tabpanel-{active}" role="tabpanel" style="{scroll_style}">{cardshtml}</div>{M.FOOT}</aside>'
        '<div aria-live="polite" class="sr-only"></div></div></div>'
    )


def head(line="Reference Grade", bypass_on=False, num="02"):
    h = M.head("reference_grade", NAME, line, pressed_on=bypass_on)
    return re.sub(r'<span class="ym-nodehead__chip">.*?</span>', f'<span class="ym-nodehead__chip" style="font-size:12px;font-weight:800">{num}</span>', h, count=1, flags=re.S)


def page(title, inner):
    return (
        f'<!doctype html>\n{M.HTML_TAG}<head>\n<meta charset="utf-8"/>\n<meta content="width=device-width, initial-scale=1.0" name="viewport"/>\n<meta content="dark" name="color-scheme"/>\n'
        f'<title>Yellowmoon · {title}</title>\n<link href="../../../mixers/html/app.css" rel="stylesheet"/>\n<link href="../../../nodes/html/nv.css" rel="stylesheet"/>\n'
        f'<link href="../../../mixers/html/mixers.css" rel="stylesheet"/>\n<link href="../../../skin/html/skin.css" rel="stylesheet"/>\n<link href="../../../colour-uniform/html/cc.css" rel="stylesheet"/>\n<link href="../rg.css" rel="stylesheet"/>\n</head>\n{inner}\n{M.SCRIPT}</body></html>\n'
    )


def write(name, html):
    os.makedirs(os.path.join(HERE, "pages"), exist_ok=True)
    with open(os.path.join(HERE, "pages", name), "w", encoding="utf-8") as f:
        f.write(html)
    print("wrote", name, len(html))


def rg_card(g, key, x, y, num, thumb_, selected=False, bypass=False, dead=False, name=NAME):
    g.corrector(key, name, x, y, num, thumb_, masks=("cu", "mask"), selected=selected, bypass=bypass, dead=dead)
    g.html[-1] = g.html[-1].replace(f'data-glyph="cu">{M.HALF}', f'data-glyph="reference_grade">{svg(RG_ICON, 14)}')


def photo(src, overlay="", look="SIGNATURE"):
    pre = S.photo_prefix("portrait", False, 28, name="Lakeside", sub="2026-11-25 · synthetic scene", overlay=overlay)
    pre = re.sub(r'(class="ym-photo-cell__image"[^>]*?src=")[^"]*(")', rf"\g<1>{AS}{src}\g<2>", pre)
    if look != "SIGNATURE":
        pre = pre.replace("Signature", look.title()).replace("SIGNATURE", look)
    return pre


def graph(after, selected=True, bypass=False, dead=False):
    g = M.Graph()
    mid = 160
    xs = [120, 334]
    g.marker("input", "Input", xs[0] - 86, mid)
    g.corrector("main", "Main", xs[0], mid - 56, "01", AS + "scene-thumb.jpg")
    rg_card(g, "rg", xs[1], mid - 56, "02", AS + after, selected=selected, bypass=bypass, dead=dead)
    g.marker("output", "Output", xs[1] + 150 + 86, mid)
    for a, b in (("input", "main"), ("main", "rg"), ("rg", "output")):
        g.link(a, b)
    return g


def callout(left, top, text, width=360):
    return f'<div class="sk-callout" style="left:{left}px;top:{top}px;max-width:{width}px">{text}</div>'


def node_page(name, title, src, cards_html, active="reference", thumb=None, mask_on=False, line="Reference Grade", extra="", look="SIGNATURE", editor_html=None, overlay="", bypass_on=False):
    pre = photo(src, overlay, look)
    thumb = thumb or src
    g = graph(thumb.replace("show-mask-", "scene-fitted-") if thumb.startswith("show-mask-") else thumb, bypass=bypass_on)
    ed = editor_html or C.editor(g.render(h=327), mask_on, "look" if active == "reference" else "color")
    inner = pre + ed + aside(head(line, bypass_on), cards_html, active) + extra
    write(name, page(title, inner))


# ---------------------------------------------------------------------------------------------------------------- pages
def page01():
    node_page("01.html", "Reference tab, empty", "scene.jpg", ref_card("none", tip="fit"), thumb="scene-thumb.jpg",
              extra=callout(630, 600, "<b>A new node changes nothing.</b> Main, Match, Output. The node is selected and opens on its Reference tab. Fit is off until a reference is chosen; the line says what to do.", 380))


def menu(open_items=True, top=430, left=1062):
    items = [("Image file…", FILE, "JPEG, PNG or TIFF"), ("Photo in the strip", IMAGES, "")]
    out = []
    for i, (nm, ic, sub) in enumerate(items):
        hl = ' style="background:color-mix(in oklch, var(--ink) 12%, transparent)"' if i == 0 else ""
        out.append(f'<div class="{M.ITEM_CLASS}" data-slot="dropdown-menu-item" role="menuitem" tabindex="-1"{hl}>{svg(ic, 20)}<span class="ym-menu-line"><span class="ym-menu-line__name">{nm}</span></span></div>')
    return (
        f'<div data-radix-popper-content-wrapper="" style="position:fixed;left:{left}px;top:{top}px;z-index:50;">'
        f'<div aria-label="Choose reference" class="{M.MENU_CLASS}" data-slot="dropdown-menu-content" data-state="open" role="menu" '
        f'style="width:216px;outline:none;--glass-liquid-opacity:90%;--glass-liquid-blur:22px">{"".join(out)}</div></div>'
    )


def page02():
    extra = menu() + callout(620, 640, "<b>Image file…</b> opens the system open dialog (JPEG, PNG and TIFF). <b>Photo in the strip</b> turns the strip into a chooser (next picture). Nothing is chosen yet, so Fit stays off.", 400)
    node_page("02.html", "Choosing a reference: file", "scene.jpg", ref_card("none", menu_open=True), thumb="scene-thumb.jpg", extra=extra)


STRIP = [
    ("strip-1.jpg", "A1M02210", False, False), ("scene-thumb.jpg", "A1M02214", True, False), ("strip-2.jpg", "A1M02215", False, False), ("strip-3.jpg", "A1M02219", False, True),
    ("strip-4.jpg", "A1M02226", False, False), ("strip-5.jpg", "A1M02227", False, False), ("strip-6.jpg", "A1M02231", False, False),
]


def strip_html(hover=3):
    items = []
    for i, (src, nm, cur, unready) in enumerate(STRIP):
        flags = ""
        label = f"Show {nm}"
        extra = ""
        cls = "ym-filmstrip__thumb rg-sthumb"
        if cur:
            flags = ' aria-current="true" aria-disabled="true" data-rg="self"'
            extra = '<span class="rg-sthumb__badge">This photo</span>'
        if unready:
            flags = ' aria-disabled="true" data-rg="unready"'
            extra = f'<span class="rg-sthumb__badge rg-sthumb__badge--busy"><i class="sk-spin rg-spin"></i>Developing</span>'
        items.append(f'<button aria-label="{label}" class="{cls}"{flags} tabindex="-1" type="button"><img alt="" draggable="false" src="{AS}{src}"/>{extra}</button>')
    return f'<nav aria-label="Choose a photo as the reference" class="ym-filmstrip rg-strip" data-tab="look"><div class="ym-filmstrip__scroll rg-strip__scroll">{"".join(items)}</div><span class="ym-filmstrip__caption">Choose a reference · 2 of 7</span></nav>'


def page02b():
    g = graph("scene-thumb.jpg")
    bar = M.BAR06
    ed = (
        f'<section aria-label="Node editor" class="ym-filmstrip nv2-editor rg-chooser" data-tab="look"><div class="nv2-bar" inert style="opacity:.45">{bar["add"]}{M.pressed(str(bar["bypass"]), False)}{bar["delete"]}'
        f'<span class="nv2-bar__sp"></span>{bar["switcher"]}{bar["mask"]}{M.pressed(str(bar["expand"]), False)}</div>'
        f'<div class="nv2-canvas rg-chooser__body" style="--cw:1040px;">{strip_html()}'
        f'{tooltip("This photo isn’t ready yet.", "left:560px;top:128px")}</div></section>'
    )
    pill = f'<div class="cc-pill" style="left:480px;top:126px">{svg(IMAGES, 14)}<span>Click a photo in the strip. Esc cancels.</span></div>'
    extra = pill + callout(24, 770, "<b>The strip is a chooser,</b> like Pick colour. The photo being graded (“This photo”) and a photo still developing are off, with the tooltip “This photo isn’t ready yet.” The graph steps aside for the strip while the mode is on.", 520)
    pre = photo("scene.jpg")
    inner = pre + ed + aside(head(), ref_card("none", chooser=True), "reference") + extra
    write("02b.html", page("Choosing a reference: the strip", inner))


def page03():
    node_page("03.html", "Fitting", "scene.jpg", ref_card("fit", pct=62), thumb="scene-thumb.jpg",
              extra=callout(630, 600, "<b>Fitting… 2 of 3.</b> A progress bar and Cancel replace the Fit button; the step is the status line, announced politely. The photo does not change until the fit is done, and a cancel changes nothing.", 380))


def page04():
    node_page("04.html", "Fitted", "scene-fitted.jpg", ref_card("fitted"), thumb="scene-fitted.jpg",
              extra=callout(630, 600, "<b>Fitted to the reference.</b> The values here are illustrative; no fit was run. The reference's thumbnail and name stay for the session only. Fit is now “Fit again”, without an accent.", 380))


def page05():
    node_page("05.html", "The Hue tab", "scene-fitted.jpg", hue_cards(), active="hue", thumb="scene-fitted.jpg",
              extra=callout(630, 600, "<b>Two periodic curves.</b> The x axis is the hue, 0° to 360°, on the rainbow track. Hue vs hue is in degrees (−180° to +180°, a detent at 0); Hue vs saturation is a log scale from 0 to 8 with its detent at 1. The fit's six knots are drawn; the person's own points are free.", 380))


def page06():
    node_page("06.html", "The Elements tab", "scene-fitted.jpg", element_cards(), active="elements", thumb="scene-fitted.jpg",
              extra=callout(630, 600, "<b>One card per element, in the order the grades run.</b> Exposure, Hue and Chroma, with the element's share of the photo. <b>Water is under 2 %</b>, so its card says “Not found in this photo.”, greys, and keeps its sliders. The eye in a card shows that element's mask.", 380))


def page06c():
    node_page("06c.html", "Show an element's mask", "show-mask-sky.jpg", element_cards(eye="sky"), active="elements", thumb="scene-fitted.jpg",
              extra=callout(630, 600, "<b>The eye in a card header</b> shows that element's soft mask in the mask accent, one at a time (the Mask tab's rows do the same). The node's Show mask button is not used: it shows the node's own masks.", 380))


def page07():
    node_page("07.html", "The look changed", "scene-newlook.jpg", ref_card("look"), thumb="scene-newlook.jpg", look="NEUTRAL",
              extra=callout(630, 600, "<b>The Output's look changed after the fit.</b> No number moved. The hint is only the line and the accent on Fit again. A re-fit would replace hand edits and takes seconds, so nothing is re-fitted by itself.", 380))


def page08():
    node_page("08.html", "Edited by hand", "scene-edited.jpg", ref_card("edited"), thumb="scene-edited.jpg",
              extra=callout(630, 600, "<b>Edited by hand since the fit.</b> Here Vegetation's Hue and Chroma were moved on the Elements tab. A fit starts again from the identity, so Fit again replaces them. The flag is a session flag; nothing is saved.", 380))


def add_menu(highlight="Reference Grade"):
    items = [
        ("Corrector", "corrector", "Grades all or part of the photo."),
        ("Film", "film", "Models a colour negative, a print and a scan."),
        ("Face Refinement", "face_refinement", "Refines each face: skin, under-eyes and teeth."),
        ("Colour Uniform", "color_uniform", "Pulls colours toward one colour."),
        ("Blur", "blur", "Soft, lens, motion or zoom blur."),
        ("Reference Grade", "reference_grade", "Matches the grade of a reference picture."),
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
        if ic not in M.ICON:
            M.ICON[ic] = '<circle cx="12" cy="12" r="3"/><circle cx="12" cy="12" r="6.5" stroke-dasharray="2 2.6"/><circle cx="12" cy="12" r="10" stroke-dasharray="1.2 3.6"/>'
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


def page09():
    g = M.Graph()
    mid = 160
    xs = [120, 334]
    g.marker("input", "Input", xs[0] - 86, mid)
    g.corrector("main", "Main", xs[0], mid - 56, "01", AS + "scene-thumb.jpg")
    g.corrector("c2", "Warmth", xs[1], mid - 56, "02", AS + "scene-thumb.jpg", selected=True)
    g.marker("output", "Output", xs[1] + 150 + 86, mid)
    for a, b in (("input", "main"), ("main", "c2"), ("c2", "output")):
        g.link(a, b)
    pre = photo("scene.jpg")
    aside06 = M.BODY06[M.BODY06.find("<aside"): M.BODY06.find("</aside>") + 8].replace('src="../assets/', 'src="' + S.A).replace("Mountain", "Warmth").replace("Corrector · 2 masks", "Corrector · whole photo")
    tail = '<div aria-live="polite" class="sr-only"></div></div></div>'
    inner = pre + M.editor(g.render(h=327), M.BAR06, False, add_open=True, mask_off=False) + aside06 + tail + add_menu()
    inner += callout(600, 600, "<b>One new item, last of the nodes.</b> “Matches the grade of a reference picture.” It adds a Reference Grade node after the selected one, selects it and opens its Reference tab. It does not open a file dialog. Off at 64 nodes: “A photo can have up to 64 nodes.”", 380)
    write("09.html", page("Add node menu", inner))


def page10():
    s = open(os.path.join(M.NODES, "04.html"), encoding="utf-8").read()
    s = s.replace('src="../assets/', 'src="' + S.A)
    s = s.replace('src="' + S.A + 'photo-graded.jpg"', f'src="{AS}scene-fitted.jpg"')
    s = s.replace("url(\"../assets/fonts/", "url(\"" + S.A + "fonts/").replace("url(../assets/fonts/", "url(" + S.A + "fonts/")
    s = s.replace("This photo has 4 nodes.", "This photo has 2 nodes.").replace("Provence", "Lakeside")
    s = s.replace(S.A + "strip-2-graded.jpg", AS + "scene-fitted.jpg")
    s = re.sub(r"Sony A1 · A1M\d+ · ISO \d+ · 1/\d+ · f/[\d.]+", "Sony A1 · A1M02214 · ISO 200 · 1/250 · f/2.0", s)
    s = re.sub(r'(<link[^>]*href=")\.\./(app\.css|nv\.css)', lambda m_: m_.group(1) + ("../../../mixers/html/app.css" if m_.group(2) == "app.css" else "../../../nodes/html/nv.css"), s)
    s = s.replace("</head>", '<link href="../../../mixers/html/mixers.css" rel="stylesheet"/><link href="../../../skin/html/skin.css" rel="stylesheet"/><link href="../../../colour-uniform/html/cc.css" rel="stylesheet"/><link href="../rg.css" rel="stylesheet"/></head>', 1)
    s = s.replace("<title>Yellowmoon</title>", "<title>Yellowmoon · Simple view</title>")
    s = s.replace("</body>", '<div class="sk-callout" style="left:24px;top:132px;max-width:380px"><b>Simple view.</b> It has no Reference Grade controls. The photo shows the result; the notice counts the node, and Open nodes goes to the node view.</div></body>', 1)
    write("10.html", s)


# ---------------------------------------------------------------------------------------------------------------- sheets
SHEET_HEAD = '<body><div class="mx-sheet sk-sheet">'


def cell(caption, body, note="", w=400, tab="look"):
    n = f"<small>{note}</small>" if note else ""
    return f'<figure class="sk-cell"><div class="sk-cell__stage sk-panel ym-tools rg-stage" data-tab="{tab}" style="width:{w}px;padding:8px 0">{body}</div><figcaption style="max-width:{w}px">{caption}{n}</figcaption></figure>'


def mini_card(state, **kw):
    """The Reference card cut down to what a status needs: the thumbnail, the buttons and the status block (no Strength or switch)."""
    h = ref_card(state, compact=True, show_strength=False, **kw)
    h = re.sub(r'<div class="ym-switch-row.*?</div>(?=<div class="rg-statusgrp")', "", h, flags=re.S)
    return h


def status_sheet():
    order = [
        ("No reference", "none", {}), ("A reference, not fitted", "ready", {}), ("Making the scene", "scene", {}), ("Fitting", "fit", {"status": "fit2"}), ("Fitted", "fitted", {}),
        ("Nothing to change", "nochange", {}), ("Edited by hand", "edited", {}), ("Look changed", "look", {}), ("Values, no reference", "kept", {}),
    ]
    cells = [cell(a, mini_card(s, **kw), note=STATUS[kw.get("status", s if s in STATUS else {"fit": "fit2", "fitted": "fitted"}.get(s, s))]) for a, s, kw in order]
    notes = [("spatial", "A note under the status, when a node after this one blurs or adds film"), ("model", "A note, when the scene model is missing"), ("bars", "Under the status"), ("profile", "Under the status"), ("grey", "Under the status"), ("noskin", "Under the status")]
    note_cell = cell("The notes", mini_card("fitted", notes=("bars", "profile", "grey", "noskin", "spatial", "model")), note="Only the ones that apply show, each on its own line under the status. All six are drawn here to show their text.")
    errs = [
        ("Unreadable file", "unreadable"), ("Unsupported format", "format"), ("Reference too small", "small"), ("Nothing inside the bars", "bars"), ("The same photo", "same"), ("Fit failed", "failed"), ("The scene model is missing (Fit)", "model"),
    ]
    ecells = [cell(a, mini_card("none", error=k), note="In `ym`, the same text follows “The reference can’t be used:”." if k == "unreadable" else "") for a, k in errs]
    ecells[2] = cell("Reference too small", mini_card("none", error="small"), note="<i>N</i> is not set yet (docs/34 §16.6). It is drawn as N.")
    ecells[0] = cell("Unreadable file", mini_card("none", error="unreadable"), note="In ym, the same text follows “The reference can’t be used:”.")
    unready = (
        '<div class="rg-sthumb-demo"><div class="rg-sthumb-demo__t">This photo isn’t ready yet.</div><div class="rg-sthumb-demo__d">An unready photo’s tile in the chooser strip is off, with this tooltip. It is not a notice in the card.</div></div>'
    )
    inner = (
        SHEET_HEAD + '<h1 class="mx-sheet__title">Reference Grade · every status line, note and error</h1>'
        '<p class="mx-sheet__sub">The copy is that of <code>docs/34</code> §9.3, character for character. The cards are cut down to the thumbnail, the buttons and the status block (the full card is in 01 to 08). One status line shows at a time; notes and errors sit with it.</p>'
        f'<section><h2>Status line, one at a time</h2><div class="mx-row sk-grid rg-grid">{"".join(cells)}</div></section>'
        f'<section><h2>Notes</h2><div class="mx-row sk-grid rg-grid">{note_cell}</div></section>'
        f'<section><h2>Errors, as a notice in the card with the text and no code</h2><div class="mx-row sk-grid rg-grid">{"".join(ecells)}</div></section>'
        f'<section><h2>An unready photo (a tile tooltip)</h2>{unready}</section></div>'
    )
    write("11.html", page("Status lines and errors", inner))


def tile(src, title, line, cls=""):
    return f'<figure class="rg-tile {cls}"><img src="{AS}{src}" alt="{title}"><figcaption><b>{title}</b>{line}</figcaption></figure>'


def before_after():
    vals = (
        '<div class="rg-vals"><b>Values used (illustrative)</b><table>'
        '<tr><th>Luma curve</th><td>toe −0.9 stop at −6, −0.8 at −4, −0.5 at −2; shoulder +0.2 at +1</td></tr>'
        '<tr><th>Red, blue curves</th><td>red −0.75 stop in the shadows, +0.75 in the highlights; blue +0.7 in the shadows, −0.8 in the highlights</td></tr>'
        '<tr><th>Saturation</th><td>−4</td></tr>'
        '<tr><th>Hue vs hue</th><td>0° +10°, 60° −8°, 120° +22°, 180° +10°, 240° −14°, 300° 0°</td></tr>'
        '<tr><th>Hue vs saturation</th><td>0° 115 %, 60° 125 %, 120° 80 %, 180° 130 %, 240° 120 %, 300° 80 %</td></tr>'
        '<tr><th>Sky</th><td>−0.45 EV · −14° · 125 %</td></tr><tr><th>Vegetation</th><td>−0.25 EV · +20° · 75 %</td></tr>'
        '<tr><th>Ground</th><td>+0.10 EV · −6° · 130 %</td></tr><tr><th>Built</th><td>0 EV · −8° · 125 %</td></tr>'
        '<tr><th>Skin</th><td>0 EV · −3° · 95 %</td></tr><tr><th>Subject</th><td>+0.25 EV</td></tr>'
        '<tr><th>Water</th><td>left at the identity: under 2 % of the photo</td></tr></table></div>'
    )
    tiles = [
        tile("scene.jpg", "The photo", "A drawn lakeside village. Nothing is graded."),
        tile("reference.jpg", "The reference", "A drawn harbour at dusk with a teal-and-amber grade. Different content on purpose: a reference is matched in its distributions, not pixel by pixel."),
        tile("scene-fitted.jpg", "Fitted, Strength 100 %", "The global grade, the hue curves and the element grades. Luma is restored by the hue step."),
        tile("scene-s50.jpg", "The same fit at Strength 50 %", "Every fitted value is halfway to the identity, with no re-fit (a ratio such as Chroma goes by a power)."),
        tile("scene-global.jpg", "Without the element grades", "The same fit with the Elements tab at the identity: the buildings, the sky and the skin keep the photo's own colour."),
        vals,
    ]
    inner = (
        SHEET_HEAD + '<h1 class="mx-sheet__title">Reference Grade · before and after</h1>'
        '<p class="mx-sheet__sub"><b>The values are illustrative; no fit was run.</b> The photo and the reference are drawings (no photograph, no film still, nobody’s likeness). The result is the maths of <code>docs/34</code> §3, §4 and §4a applied by <code>rg_ref.py</code> with values chosen by hand, through soft masks taken from the drawing’s own element layers. The display transform is a simple filmic curve and the sRGB encoding, not the product’s.</p>'
        f'<section><div class="rg-tiles">{"".join(tiles)}</div></section></div>'
    )
    write("12.html", page("Before and after", inner))


def node_cell(caption, note="", **kw):
    g = M.Graph()
    rg_card(g, "s", 47, 90, "02", AS + "scene-fitted.jpg", **kw)
    return f'<figure class="mx-cell" style="width:250px"><div class="mx-cell__stage" style="height:230px">{g.render(250, 230)}</div><figcaption>{caption}{("<small>" + note + "</small>") if note else ""}</figcaption></figure>'


def states_page():
    cells = [
        node_cell("Normal", "numbered like Film; the node's glyph sits beside the mask glyph"),
        node_cell("Selected", selected=True),
        node_cell("Bypassed", "half-strength thumbnail and the slash", bypass=True),
        node_cell("Dead", "not on a path to the Output: 45 %", dead=True),
    ]
    glyph = (
        '<div class="rg-glyphs">'
        + "".join(f'<span class="rg-glyph" style="--s:{s}px">{svg(RG_ICON, s)}<small>{s} px</small></span>' for s in (14, 24, 48, 96))
        + '<p class="rg-glyphs__txt"><b>The glyph (proposal).</b> A picture beside a picture, with a grade curve in the front one: “this photo, graded like that one”. It sits beside the mask glyph in the card’s footer at 14 px, and in the Add node menu at 24 px. It is drawn with the same 2 px stroke and round caps as Blur’s and Colour Uniform’s, so it is not a filled shape and does not read as the Output look’s half-disc.</p></div>'
    )
    panels = [
        cell("Panel head, normal", head() + '<div style="height:6px"></div>', "“Match” is the node's name (default); the line is “Reference Grade”"),
        cell("Panel head, bypassed", head("Bypassed", True), "the head’s Bypass is on; the node passes its picture through"),
        cell("Panel head, not connected", head("Not connected"), "the node is on no path to the Output"),
    ]
    tips = [
        ("Add node › Reference Grade", "Matches the grade of a reference picture."),
        ("Add node › Reference Grade, at 64 nodes", "A photo can have up to 64 nodes."),
        ("Fit, no reference", "Choose a reference first."),
        ("Choose reference… › Image file…", "Choose a JPEG, PNG or TIFF picture."),
        ("A strip tile, the photo being graded", "That’s the photo you’re grading. Choose another."),
        ("A strip tile, still developing", "This photo isn’t ready yet."),
        ("The strip chooser’s hint", "Click a photo in the strip. Esc cancels."),
        ("Strength, under the slider", "Scales the fitted values. The tabs show them before Strength."),
        ("Skin protection", "Keeps skin tones from drifting. Applies at the next fit."),
        ("An element’s eye", "Show where the sky grade acts"),
        ("An element not in the photo", "Not found in this photo."),
        ("Elements, while the scene is made", "Looking at the photo…"),
        ("Hue curves, at 16 points", "A curve can have up to 16 points."),
        ("The simple view’s notice", "This photo has 2 nodes. These controls change the input, the output and node 1, Main."),
    ]
    tip_html = "".join(f'<div class="mx-tiprow"><span>{a}</span>{M.tip(b)}</div>' for a, b in tips)
    inner = (
        SHEET_HEAD + '<h1 class="mx-sheet__title">Reference Grade · states</h1>'
        '<p class="mx-sheet__sub">The panel is 400 px wide, as in the app. The picture is a synthetic scene; the values are illustrative.</p>'
        f'<section><h2>Node card</h2><div class="mx-row">{"".join(cells)}</div></section>'
        f'<section><h2>The glyph</h2>{glyph}</section>'
        f'<section><h2>Panel head</h2><div class="mx-row sk-grid">{"".join(panels)}</div></section>'
        f'<section><h2>Copy for tooltips (docs/34 §9.3 where it has the text; the rest are drafts)</h2><div class="mx-tips">{tip_html}</div></section></div>'
    )
    write("states.html", page("Reference Grade states", inner))


def elements_sheet():
    order = ["sky", "vegetation", "water", "ground", "built", "skin", "subject"]
    found = ("sky", "vegetation", "ground", "built", "skin", "subject")
    cells = [cell(TITLES[e] if e != "water" else "Water (not in the photo)", element_card(e, found=(e in found), eye_on=False, last=True), tab="color") for e in order]
    cells.append(cell("An element with its mask shown", element_card("sky", eye_on=True, last=True), "the eye is on: the photo shows the sky’s soft mask (06c)", tab="color"))
    cells.append(cell("While the scene is being made", element_card("vegetation", looking=True, last=True), "every card says “Looking at the photo…”; the eye is off and the sliders still work", tab="color"))
    inner = (
        SHEET_HEAD + '<h1 class="mx-sheet__title">Reference Grade · the Elements tab, all seven cards</h1>'
        '<p class="mx-sheet__sub">In the order the grades run (<code>docs/34</code> §2.3): Sky, Vegetation, Water, Ground, Built, Skin, Subject. Exposure, Hue and Chroma are the fit’s own ranges (Hue is ± the element’s cap: 40°, 35°, 35°, 30°, 25°, 15°). Chroma is a percentage of the photo’s own colour, 50 to 160. The Subject has Exposure only. The values are illustrative; no fit was run.</p>'
        f'<section><div class="mx-row sk-grid rg-grid4">{"".join(cells)}</div></section></div>'
    )
    write("06b.html", page("Elements, all cards", inner))


if __name__ == "__main__":
    for f in (page01, page02, page02b, page03, page04, page05, page06, page06c, page07, page08, page09, page10, status_sheet, before_after, states_page, elements_sheet):
        f()
