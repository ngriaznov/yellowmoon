#!/usr/bin/env python3
"""Builds the Depth mask mockup pages (design/depth/html/pages/*.html).

The frame (header, graph, tools panel), the glasscn markup, the mask rows, the sliders, the Add mask
menu and the pick hint are lifted from the node mockups (design/nodes/html/pages 03, 07, 08) and
re-written with the Depth copy. Run make-pictures.py first, then this, then shoot.mjs.
"""
import re, os, json

HERE = os.path.dirname(os.path.abspath(__file__))
N = os.path.join(HERE, '..', 'nodes', 'html', 'pages')
OUT = os.path.join(HERE, 'html', 'pages')
NUM = json.load(open(os.path.join(HERE, 'html', 'assets', 'numbers.json')))
os.makedirs(OUT, exist_ok=True)


def rd(p): return open(p, encoding='utf-8').read()


T07, T08, T03 = (rd(os.path.join(N, n + '.html')) for n in ('07', '08', '03'))
SCRIPT = rd(os.path.join(HERE, '..', 'film', 'html', 'pages', '09.html'))
SCRIPT = SCRIPT[SCRIPT.index('<script>'):]

TAGS = re.compile(r'<(/?)(div|span|button|section|p|aside|main|header|nav|label|kbd)\b[^>]*?>')


def block(s, start):
    depth = 0
    for m in TAGS.finditer(s, start):
        depth += -1 if m.group(1) else 1
        if depth == 0:
            return s[start:m.end()]
    raise ValueError('unbalanced')


def find_block(s, needle, tag='div', frm=0):
    i = s.index(needle, frm)
    return block(s, s.rfind('<' + tag, 0, i))


def svg_of(frag): return re.search(r'<svg.*?</svg>', frag, re.S).group(0)


def icon(paths, size=16, cls=''):
    return ('<svg aria-hidden="true" class="%s" fill="none" height="%d" stroke="currentColor" stroke-linecap="round" '
            'stroke-linejoin="round" stroke-width="2" viewbox="0 0 24 24" width="%d" xmlns="http://www.w3.org/2000/svg">%s</svg>'
            % (cls, size, size, paths))


ICON_DEPTH = '<path d="M3 20h18"></path><path d="M5 17V9"></path><path d="M10 17V6"></path><path d="M15 17v-5"></path><path d="M19.5 17v-3"></path>'
ICON_FOCUS = ('<circle cx="12" cy="12" r="3"></circle><path d="M3 7V5a2 2 0 0 1 2-2h2"></path><path d="M17 3h2a2 2 0 0 1 2 2v2"></path>'
              '<path d="M21 17v2a2 2 0 0 1-2 2h-2"></path><path d="M7 21H5a2 2 0 0 1-2-2v-2"></path>')
ICON_EYE = '<path d="M2.06 12.35a1 1 0 0 1 0-.7 10.75 10.75 0 0 1 19.88 0 1 1 0 0 1 0 .7 10.75 10.75 0 0 1-19.88 0"></path><circle cx="12" cy="12" r="3"></circle>'
ICON_SPIN = '<path d="M21 12a9 9 0 1 1-6.219-8.56"></path>'

# ---------- fragments ----------
ROW_ADD = find_block(T07, 'aria-label="Object mask, 2 points"')
ROW_SUB = find_block(T07, 'aria-label="Radial mask, Feather 45"')
MASKS_CARD = find_block(T07, '>Masks</h3>', 'section')
CARD_OBJ = find_block(T07, '>Object</h3>', 'section')
SLIDER0 = find_block(CARD_OBJ, 'data-slider=""')
MENU = find_block(T08, 'data-menu-anchor="#add-mask-btn"')
ITEM_BG = find_block(MENU, '>Background</span>', 'div')
ITEM_BG = ITEM_BG if 'dropdown-menu-item' in ITEM_BG[:400] else None
if ITEM_BG is None:  # the span sits inside the item: climb to the item itself
    i = MENU.index('>Background</span>')
    ITEM_BG = block(MENU, MENU.rfind('<div class="glass-focus-inset', 0, i))
PICKROW = find_block(T03, 'class="nv-pickrow"')
PICK_TOGGLE = block(PICKROW, PICKROW.index('<span class="inline-flex"'))
HINT = find_block(T03, 'ym-photo-cell__busy nv-found nv-found--plain', 'span')
CROSS = re.search(r'<svg aria-hidden="true" class="nv-cross".*?</svg>', T03, re.S).group(0)
CHIP = find_block(T07, 'data-variant="outline">Mask · Object', 'span')
ADD_BTN = find_block(MASKS_CARD, 'id="add-mask-btn"', 'button')

# the page frame is 07's body, with its tools panel's scroll area and its viewer cell swapped per page
BODY = T07[T07.index('<body'):T07.rindex('</body>')]
BODY = BODY[:BODY.rindex('<script>')] if '<script>' in BODY else BODY


# ---------- builders ----------
def maskrow(kind, sub, ico, mode='add', selected=False, eye=False, sub_html=None, label=None):
    r = ROW_ADD if mode == 'add' else ROW_SUB
    r = re.sub(r'aria-label="(Object|Radial) mask, [^"]*"', 'aria-label="%s"' % (label or '%s mask, %s' % (kind, sub)), r, 1)
    r = r.replace('aria-label="Object mode"', 'aria-label="%s mode"' % kind).replace('aria-label="Radial mode"', 'aria-label="%s mode"' % kind)
    r = re.sub(r'data-selected="(true|false)"', 'data-selected="%s"' % ('true' if selected else 'false'), r, 1)
    r = re.sub(r'<svg.*?</svg>', lambda m: ico, r, 1, flags=re.S)
    r = re.sub(r'<span class="nv-maskrow__name">[^<]*</span>', '<span class="nv-maskrow__name">%s</span>' % kind, r)
    sub_el = sub_html or '<span class="nv-maskrow__sub">%s</span>' % sub
    r = re.sub(r'<span class="nv-maskrow__sub">[^<]*</span>', lambda m: sub_el, r)
    if eye:
        eb = re.search(r'<button aria-label="Show this mask" aria-pressed="false".*?</button>', r, re.S).group(0)
        nb = eb.replace('aria-pressed="false"', 'aria-pressed="true"').replace('data-state="off"', 'data-state="on"').replace('aria-label="Show this mask"', 'aria-label="Hide this mask"')
        r = r.replace(eb, nb).replace('<span class="sr-only">Show this mask</span>', '<span class="sr-only">Hide this mask</span>')
    return r


def spinning(text):
    return '<span class="nv-maskrow__sub dp-finding">%s%s</span>' % (icon(ICON_SPIN, 12, 'dp-spin'), text)


_lab = [0]


def slider(label, value, text=None, start=0.0, disabled=False):
    """One RasterSliders-style row. value 0..100; start is where the fill begins (0, or 50 for a centred one)."""
    _lab[0] += 1
    s = SLIDER0
    text = text if text is not None else str(value)
    lo, hi = min(start, value), max(start, value)
    s = re.sub(r'id="lab-[^"]+">[^<]*</span>', 'id="dp-lab-%d">%s</span>' % (_lab[0], label), s, 1)
    s = re.sub(r'aria-labelledby="lab-[^"]+"', 'aria-labelledby="dp-lab-%d"' % _lab[0], s)
    s = re.sub(r'inset-inline-start: [\d.]+%; width: [\d.]+%', 'inset-inline-start: %.2f%%; width: %.2f%%' % (lo, hi - lo), s)
    s = re.sub(r'left: calc\([\d.]+% \+ [-\d.]+px\)', 'left: calc(%.2f%% + %.2fpx)' % (value, 8 - 16 * value / 100), s)
    s = re.sub(r'aria-valuenow="[^"]*" aria-valuetext="[^"]*"', 'aria-valuenow="%d" aria-valuetext="%s"' % (value, text), s)
    s = re.sub(r'aria-label="[^"]* value"', 'aria-label="%s value"' % label, s)
    s = re.sub(r'value="[^"]*"/>', 'value="%s"/>' % text, s)
    if disabled:
        s = s.replace('aria-disabled="false"', 'aria-disabled="true" data-disabled=""')
    return s


def raster_sliders(disabled=False):
    return (slider('Refine edges', 50, '50%', 50, disabled) + slider('Feather', 0, '0', 0, disabled) +
            slider('Grow', 0, '0', 0, disabled) + slider('Opacity', 100, '100%', 0, disabled))


def toggle(text, ico, on, aria=None):
    t = PICK_TOGGLE
    if ico:
        t = re.sub(r'<svg.*?</svg>', lambda m: icon(ico, 14), t, 1, flags=re.S)
    t = re.sub(r'(</svg>)Pick white</button>', lambda m: m.group(1) + text + '</button>', t)
    if not on:
        t = t.replace('aria-pressed="true"', 'aria-pressed="false"').replace('data-state="on"', 'data-state="off"').replace(' nv-pick-on', '')
    t = re.sub(r'<span class="sr-only">[^<]*</span>', '<span class="sr-only">%s</span>' % (aria or text), t)
    return t


def focus_button(disabled=False):
    b = ADD_BTN.replace('id="add-mask-btn"', '')
    b = re.sub(r'aria-expanded="false" aria-haspopup="menu" ', '', b)
    b = re.sub(r'>Add mask<svg.*?</svg>', '>' + icon(ICON_FOCUS, 14) + 'Focus plane', b, flags=re.S)
    if disabled:
        b = b.replace('type="button"', 'type="button" disabled aria-disabled="true"')
    return '<span class="inline-flex dp-anchor" data-slot="tooltip-trigger">%s<span class="sr-only">%s</span></span>' % (
        b, 'This photo has no focus point' if disabled else 'Select what the camera focused on')


def note(text, cls=''):
    return '<p class="ym-row-note %s">%s</p>' % (cls, text)


def depth_card(near, far, soft, pick=False, show=False, focus='ok', note_html='', sliders_off=False, tip=None, error=None, focus_tip=False):
    """The DEPTH card, in the spec's order: Pick depth, Focus plane, Near, Far, Softness, Show depth, the four raster sliders."""
    h = CARD_OBJ
    head = h[:h.index('<div aria-label="Object tool"')]
    head = head.replace('>Object</h3>', '>Depth</h3>')
    out = head
    if error:
        return head + note(error, 'dp-error') + '</section>'
    out += '<div class="nv-pickrow">%s<span class="nv-pickrow__right">%s</span></div>' % (
        toggle('Pick depth', None, pick, aria='Click the part of the photo to select (D)'),
        focus_button(disabled=(focus == 'none')) + ('<span class="dp-tip dp-tip--focus" role="tooltip">Select what the camera focused on</span>' if focus_tip else ''))
    if focus == 'none':
        out += note('This photo has no focus point.')
    elif focus == 'centre':
        out += note('The camera may not have recorded where it focused. Check the result.')
    out += slider('Near', near) + slider('Far', far) + slider('Softness', soft)
    tt = ''
    if tip:
        tt = '<span class="dp-tip" role="tooltip">%s</span>' % tip
    out += '<div class="nv-pickrow dp-showrow">%s%s</div>' % (toggle('Show depth', ICON_EYE, show, aria='Show how near or far each part is. White is near.'), tt)
    out += raster_sliders()
    out += note_html + '</section>'
    return out


# ---------- the panel's Masks card ----------
NOTE_MASKS = 'Masks add up. Subtract cuts a mask out. Colour and brightness ranges keep only their part of the rest.'


def masks_card(rows):
    c = MASKS_CARD
    a = c.index('<div class="nv-maskrows">')
    blk = block(c, a)
    c = c.replace(blk, '<div class="nv-maskrows">%s</div>' % ''.join(rows))
    return c


def menu(items_html_depth=True, open_for=None):
    m = MENU
    if items_html_depth:
        d = ITEM_BG
        d = d.replace('>Background</span>', '>Depth</span>').replace('What is out of focus, from blur', 'Near or far, from depth')
        d = re.sub(r'<svg.*?</svg>', lambda mm: icon(ICON_DEPTH, 17, 'nv-menu-item__ico'), d, 1, flags=re.S)
        m = m.replace(ITEM_BG, ITEM_BG + d)
    return m


# the viewer cell
def viewer(label, src='photo.jpg', overlay=None, chip='Mask · Depth', hint=None, hint_spin=False, cross=None, camera=None, extra=''):
    v = ('<div aria-label="Photo A1M01739, %s" class="ym-photo-cell__view" role="img">' % label +
         '<img alt="" class="ym-photo-cell__image" draggable="false" src="../assets/%s"/>' % src +
         (CHIP.replace('Mask · Object', chip) if chip else '') +
         '<span class="ym-photo-cell__camera">%s</span>' % (camera or 'Sony A1 · A1M01739 · ISO 100 · 1/4000 · f/2.8'))
    if overlay:
        v += '<img alt="" class="nv-overlay" src="../assets/%s"/>' % overlay
    if cross:
        v += cross
    if hint:
        h = HINT
        h = re.sub(r'(</svg>)[^<]*</span>$', lambda m: m.group(1) + hint + '</span>', h)
        if hint_spin:
            h = re.sub(r'<svg.*?</svg>', lambda m: icon(ICON_SPIN, 13, 'dp-spin'), h, 1, flags=re.S)
        v += h
    v += extra + '</div>'
    return v


def cross_at(fx, fy, w=666, h=444):
    c = CROSS
    c = re.sub(r'left:[\d.]+px;top:[\d.]+px', 'left:%.1fpx;top:%.1fpx' % (fx * w - 24, fy * h - 24), c)
    return c


def page(title, viewer_html, tools_scroll, menu_html='', node_sub='Corrector · 1 mask', show_mask_on=True, tab='Mask'):
    b = BODY
    # the viewer cell
    a = b.index('<div aria-label="Photo A1M01739')
    blk = block(b, a)
    b = b.replace(blk, viewer_html)
    # the node: its name and its line
    b = b.replace('Mountain', 'Haze').replace('Corrector · 2 masks', node_sub)
    # the tools scroll area
    a = b.index('<div aria-labelledby="ym-tab-light" class="ym-tools__scroll"')
    blk = block(b, a)
    inner = blk[:blk.index('>') + 1] + tools_scroll + '</div>'
    b = b.replace(blk, inner)
    # a stale menu from the template, then ours
    if 'data-menu-anchor' in b:
        a = b.index('<div data-menu-anchor')
        b = b.replace(block(b, a), '')
    if not show_mask_on:
        mt = re.search(r'<button[^>]*aria-pressed="true"[^>]*data-state="on"[^>]*>Show mask</button>', b)
        if mt:
            b = b.replace(mt.group(0), mt.group(0).replace('aria-pressed="true"', 'aria-pressed="false"').replace('data-state="on"', 'data-state="off"'))
    return (('<!doctype html>\n<html class="dark" data-theme="dark" lang="en" style="background: rgb(15, 15, 16); color-scheme: dark;">'
             '<head><meta charset="utf-8"/><meta content="width=device-width, initial-scale=1.0" name="viewport"/>'
             '<meta content="dark" name="color-scheme"/><title>%s</title>'
             '<link href="../app.css" rel="stylesheet"/><link href="../nv.css" rel="stylesheet"/><link href="../dp.css" rel="stylesheet"/></head>\n') % title
            + b + menu_html + SCRIPT)


def save(name, html):
    open(os.path.join(OUT, name + '.html'), 'w', encoding='utf-8').write(html)
    print(name, len(html))


# ---------- the pages ----------
ICON_SUBJ = svg_of(block(MENU, MENU.rfind('<div class="glass-focus-inset', 0, MENU.index('>Subject</span>'))))
ICON_RAD_ROW = svg_of(ROW_SUB)
ICON_DEPTH_ROW = icon(ICON_DEPTH, 16)
ICON_PEOPLE = None
cam_pano = 'Sony A1 · A1M01740 · Panorama · ISO 100 · 1/500 · f/8'
N_ = NUM
band_txt = lambda a, b: '%d to %d' % (a, b)

radial = lambda sel=False: maskrow('Radial', 'Feather 45', ICON_RAD_ROW, 'add', sel)


def tools(rows, card=''):
    return masks_card(rows).replace('</section>', note(NOTE_MASKS) + '</section>', 1) if False else masks_card(rows) + card


# 01 the Add mask menu with Depth
save('01', page('Depth mask · Add mask menu',
                viewer('Mask', chip=None), tools([radial()]), menu_html=menu(True),
                node_sub='Corrector · 1 mask', show_mask_on=False))

# 02 a new Depth row while the map is made
save('02', page('Depth mask · Finding',
                viewer('Mask · Depth', hint='Getting ready…', hint_spin=True),
                tools([maskrow('Depth', '', ICON_DEPTH_ROW, 'add', True, eye=True, sub_html=spinning('Finding…'),
                               label='Depth mask, finding')],
                      depth_card(0, 50, 50, pick=True))))

# 03 ready
save('03', page('Depth mask · Ready',
                viewer('Mask · Depth', overlay='mask-default.png'),
                tools([maskrow('Depth', band_txt(0, 50), ICON_DEPTH_ROW, 'add', True, eye=True)],
                      depth_card(0, 50, 50, focus_tip=True))))

# 04 Pick depth on
px, py = N_['picked'][3], N_['picked'][4]
save('04', page('Depth mask · Pick depth',
                viewer('Mask · Depth', overlay='mask-default.png', hint='Click the part of the photo to select. Esc stops.',
                       cross=cross_at(px, py)),
                tools([maskrow('Depth', band_txt(0, 50), ICON_DEPTH_ROW, 'add', True, eye=True)],
                      depth_card(0, 50, 50, pick=True))))

# 04b after the click
pn, pf = N_['picked'][0], N_['picked'][1]
save('04b', page('Depth mask · After a pick',
                 viewer('Mask · Depth', overlay='mask-picked.png', hint='Click the part of the photo to select. Esc stops.',
                        cross=cross_at(px, py)),
                 tools([maskrow('Depth', band_txt(pn, pf), ICON_DEPTH_ROW, 'add', True, eye=True)],
                       depth_card(pn, pf, 50, pick=True))))

# 05 no focus point
save('05', page('Depth mask · No focus point',
                viewer('Mask · Depth', overlay='mask-default.png', camera=cam_pano),
                tools([maskrow('Depth', band_txt(0, 50), ICON_DEPTH_ROW, 'add', True, eye=True)],
                      depth_card(0, 50, 50, focus='none'))))

# 05b the point is the centre of the frame
cn, cf = N_['centre'][0], N_['centre'][1]
save('05b', page('Depth mask · Focus at the centre',
                 viewer('Mask · Depth', overlay='mask-centre.png'),
                 tools([maskrow('Depth', band_txt(cn, cf), ICON_DEPTH_ROW, 'add', True, eye=True)],
                       depth_card(cn, cf, 50, focus='centre'))))

# 06 Show depth
save('06', page('Depth mask · Show depth',
                viewer('Depth', src='depth-grey.jpg', chip='DEPTH'),
                tools([maskrow('Depth', band_txt(0, 50), ICON_DEPTH_ROW, 'add', True, eye=False)],
                      depth_card(0, 50, 50, show=True, tip='Show how near or far each part is. White is near.')),
                show_mask_on=False))

# 07 no depth model: the menu has no Depth, a row brought by a file says why
save('07', page('Depth mask · No model',
                viewer('Mask', chip=None),
                tools([maskrow('Depth', 'Depth masks need the depth model. Reinstall the app.', ICON_DEPTH_ROW, 'add', False),
                       radial()]),
                menu_html=menu(False), node_sub='Corrector · 2 masks', show_mask_on=False))

# 07b the map failed
save('07b', page('Depth mask · Can\'t be made',
                 viewer('Mask · Depth'),
                 tools([maskrow('Depth', 'Can\'t be made', ICON_DEPTH_ROW, 'add', True, eye=True)],
                       depth_card(0, 50, 50, error='The Depth mask can\'t be made: the model stopped before it finished. Try again, or remove the mask.'))))

# 08 Depth with a Subject row
save('08', page('Depth mask · With a Subject row',
                viewer('Mask · Depth, Subject', overlay='mask-far-minus-subject.png'),
                tools([maskrow('Depth', band_txt(55, 100), ICON_DEPTH_ROW, 'add', True, eye=False),
                       maskrow('Subject', 'Foreground', ICON_SUBJ, 'sub', False)],
                      depth_card(55, 100, 50, note_html=note('Depth is soft at edges. Add a Subject mask and subtract it to keep a clean edge.'))),
                node_sub='Corrector · 2 masks'))
