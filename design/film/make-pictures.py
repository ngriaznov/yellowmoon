# Hand-made "film look" illustrations. Not shipped, not the engine, not spektrafilm.
import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter
SRC='/home/user/yellowmoon/design/nodes/html/assets/photo-graded.jpg'
OUT='/home/user/yellowmoon/design/film/html/assets/'

# name: contrast, sat, warm, tint(+magenta/-green), lift, shoulder(highlight roll), grain, halation, gain
S={
 'portra400':  dict(c=1.12,s=0.92,w=0.06,t=0.01,l=0.025,sh=0.55,g=0.030,h=0.35,gain=1.04,soft=0.6),
 'portra160':  dict(c=1.10,s=0.90,w=0.05,t=0.01,l=0.025,sh=0.55,g=0.018,h=0.30,gain=1.02,soft=0.5),
 'portra800':  dict(c=1.14,s=0.94,w=0.08,t=0.01,l=0.03,sh=0.55,g=0.050,h=0.45,gain=1.06,soft=0.7),
 'ektar100':   dict(c=1.22,s=1.28,w=0.03,t=0.0,l=0.012,sh=0.45,g=0.014,h=0.25,gain=1.0,soft=0.0),
 'gold200':    dict(c=1.16,s=1.08,w=0.12,t=-0.01,l=0.02,sh=0.5,g=0.034,h=0.35,gain=1.03,soft=0.3),
 'ultramax400':dict(c=1.18,s=1.12,w=0.09,t=0.0,l=0.02,sh=0.5,g=0.046,h=0.35,gain=1.03,soft=0.3),
 'fujic200':   dict(c=1.14,s=1.0,w=-0.03,t=-0.02,l=0.02,sh=0.5,g=0.030,h=0.3,gain=1.0,soft=0.3),
 'pro400h':    dict(c=1.04,s=0.86,w=-0.04,t=-0.02,l=0.04,sh=0.6,g=0.026,h=0.3,gain=1.05,soft=0.5),
 'xtra400':    dict(c=1.20,s=1.10,w=-0.02,t=-0.02,l=0.018,sh=0.5,g=0.046,h=0.35,gain=1.02,soft=0.3),
 'vision50d':  dict(c=0.98,s=0.82,w=0.01,t=0.0,l=0.045,sh=0.65,g=0.014,h=0.45,gain=1.04,soft=0.4),
 'vision250d': dict(c=0.98,s=0.84,w=0.01,t=0.0,l=0.045,sh=0.65,g=0.024,h=0.45,gain=1.04,soft=0.4),
 'vision200t': dict(c=0.98,s=0.84,w=0.10,t=0.0,l=0.045,sh=0.65,g=0.020,h=0.45,gain=1.03,soft=0.4),
 'vision500t': dict(c=0.98,s=0.84,w=0.08,t=0.01,l=0.045,sh=0.65,g=0.034,h=0.5,gain=1.03,soft=0.4),
 'verita200d': dict(c=1.0,s=0.88,w=0.02,t=0.0,l=0.04,sh=0.65,g=0.020,h=0.45,gain=1.03,soft=0.4),
 'ektachrome': dict(c=1.26,s=1.06,w=-0.03,t=0.0,l=-0.01,sh=0.4,g=0.016,h=0.12,gain=1.0,soft=0.0),
 'kodachrome': dict(c=1.30,s=1.12,w=0.08,t=0.01,l=-0.012,sh=0.4,g=0.022,h=0.12,gain=1.0,soft=0.0),
 'velvia':     dict(c=1.36,s=1.42,w=0.0,t=-0.01,l=-0.015,sh=0.35,g=0.014,h=0.1,gain=0.99,soft=0.0),
 'provia':     dict(c=1.24,s=1.12,w=-0.02,t=0.0,l=-0.008,sh=0.4,g=0.014,h=0.12,gain=1.0,soft=0.0),
}
def lin(x): return np.power(np.clip(x,0,1),2.2)
def enc(x): return np.power(np.clip(x,0,1),1/2.2)
def luma(a): return a@np.array([.2627,.678,.0593])
def render(im, p, grain=True, grain_scale=1.0, seed=7):
    a=np.asarray(im,dtype=np.float32)/255.
    h,w=a.shape[:2]
    L=lin(a)*p['gain']
    # halation: bright areas glow red-orange
    y=luma(L); m=np.clip((y-0.45)/0.55,0,1)**1.5
    r=max(w,h)/2000.0
    glow=np.stack([gaussian_filter(m,14*r),gaussian_filter(m,14*r)*0.45,gaussian_filter(m,14*r)*0.12],-1)
    L=L+p['h']*0.55*glow
    # soft
    if p['soft']>0:
        bl=np.stack([gaussian_filter(L[...,i],1.1*r) for i in range(3)],-1)
        L=L*(1-0.35*p['soft'])+bl*0.35*p['soft']
    # log domain, pivot at the picture's own median so the brightness stays close to the source
    ymed_l=float(np.median(luma(L))); ymed_d=float(np.median(luma(a)))
    e=np.log2(np.maximum(L,1e-4)/max(ymed_l,1e-3))
    k=p['c']; sh=p['sh']
    b0=np.log(ymed_d/(1-ymed_d))
    out=np.empty_like(L)
    for i,wr in enumerate([1.0+p['w'],1.0,1.0-p['w']]):
        y_=1/(1+np.exp(-(e[...,i]*k*0.62+b0)))
        y_=np.where(y_>0.78,0.78+(y_-0.78)*(1-0.55*sh),y_)
        out[...,i]=y_*wr
    # tint
    out[...,1]*=1-p['t']
    # saturation
    yy=luma(out)[...,None]
    out=yy+(out-yy)*p['s']
    # lifted black
    out=out*(1-p['l'])+p['l']
    # warmer highlights
    hl=np.clip((luma(out)-0.55)/0.45,0,1)[...,None]
    out=out*(1+hl*np.array([0.03,0.0,-0.04])*(1+p['w']*3))
    # out is display-referred already; match mean brightness to source for "as the app"
    out=np.clip(out,0,1)
    if grain:
        rng=np.random.default_rng(seed)
        n=rng.standard_normal((h,w,3)).astype(np.float32)
        nl=rng.standard_normal((h,w)).astype(np.float32)
        gs=0.8*grain_scale
        nl=gaussian_filter(nl,gs); nl/= (nl.std()+1e-6)
        nc=np.stack([gaussian_filter(n[...,i],gs*1.1) for i in range(3)],-1); nc/=(nc.std()+1e-6)
        mix=0.75*nl[...,None]+0.45*nc
        yv=luma(out)
        wgt=(0.25+1.2*yv*(1-yv))[...,None]*(1.0-0.5*np.clip((yv-0.6)/0.4,0,1))[...,None]    # most in the mid-tones
        out=out+mix*wgt*p['g']*grain_scale
    return Image.fromarray((np.clip(out,0,1)*255+0.5).astype(np.uint8))
def cover(im,w,h):
    r=max(w/im.width,h/im.height); im2=im.resize((round(im.width*r),round(im.height*r)),Image.LANCZOS)
    l=(im2.width-w)//2; t=(im2.height-h)//2
    return im2.crop((l,t,l+w,t+h))
if __name__=='__main__':
    src=Image.open(SRC).convert('RGB')
    big=src.resize((2000,1333),Image.LANCZOS)
    # the main picture: Portra 400, 2400 wide like the graded photo
    full=render(src,S['portra400'],grain_scale=1.15)
    full.save(OUT+'film-portra400.jpg',quality=88)
    # same without grain (for options where the tab shows the effect off? keep for zoom compare)
    # 100% crop: 1332x888 at image pixels, from the lower-left field+trees region
    g=render(src,S['portra400'],grain_scale=1.6)
    cx,cy=700,640
    g.crop((cx,cy,cx+1332,cy+888)).save(OUT+'film-portra400-100.jpg',quality=90)
    # node thumb
    cover(full,384,192).save(OUT+'node-film.jpg',quality=88)
    # tiles for the picker (small render of this photo)
    small=src.resize((600,400),Image.LANCZOS)
    for k,p in S.items():
        cover(render(small,p,grain_scale=0.9),320,200).save(OUT+f'tile-{k}.jpg',quality=86)
    # slide example (Velvia) for screen 7
    render(src,S['velvia'],grain_scale=1.0).save(OUT+'film-velvia.jpg',quality=88)
    # contact for my own check
    names=list(S)
    sheet=Image.new('RGB',(320*6,200*3))
    for i,k in enumerate(names):
        sheet.paste(Image.open(OUT+f'tile-{k}.jpg'),((i%6)*320,(i//6)*200))
    sheet.save('/tmp/tiles-check.jpg',quality=85)
