# 01 · Adaptive ratio demosaic

A three-pass Bayer demosaic for Yellowmoon's R6 classical path:

1. **Green.** At every R and B site the algorithm classifies the local edge direction (horizontal, vertical or none) with white-balance-aware colour-difference tests. It then estimates green as a noise-regularised colour ratio from the two or four adjacent greens.
2. **Green refinement.** At every R and B site it compares the vertical and horizontal directions with two diagonals and four knight-move directions. Where an oblique direction is smoother, it re-estimates green along that direction. An optional weighting, driven by lateral chromatic aberration (CA) curves, can be added (phase 2).
3. **Chroma.** It fills the missing R and B by cost-weighted offset ratios, first at R/B sites along the smoother diagonal, then at G sites from the four axis neighbours. A diagonal luma sharpening then runs in shadows only. For low-SNR images, a noise-adaptive shadow colour matching step follows.

The C++ reference is `reference/adaptive_ratio_demosaic.cpp`. Test vectors are in `vectors/adaptive_ratio_demosaic/`.

---

## 1. Purpose, place in the chain, contract

### 1.1 Where it runs

| Item | Value |
|---|---|
| Stage | R6, classical demosaic, `Method::AdaptiveRatio` in `ym-gpu/src/cpu/demosaic.rs` (implementation in `demosaic/adaptive_ratio.rs`) |
| Used for | export fallback while no trained network is available, 1:1 tiles, and R7 Detail's `Y_classical` (same method as R6) |
| Before | R4 gains `g0`, R5 highlight reconstruction |
| After | R6 bilateral denoise (unchanged), R7, R8, R9 |

### 1.2 Input and output

| | Contract |
|---|---|
| Algorithm input | `ym_gpu::cpu::Mosaic` content: one f32 per photosite, row-major, black-subtracted, white = 1.0, **not white-balanced**, plus the 4-phase `Cfa` and the `Placement` (used only by the CA path). |
| Yellowmoon R6 input | The R5 output is **balanced** (R4 multiplied it by `g0`). On entry R6 divides every photosite by `g0` of its CFA colour, `m_u(x,y) = m_bal(x,y) / g0[colour(x,y)]`. This is the exact inverse of R4. R6 then passes `gains = g0`. The reference offers this as `demosaic_balanced()`. |
| Gains | `gains = [g_R, g_G, g_B]`, absolute, finite, > 0. Yellowmoon passes `g0 = [g_R, 1, g_B]`. |
| Noise | `NoiseModel { a, b, sigma_row, q }` (normalised, unbalanced); see §2.2. `delta[4]` is not used. |
| Value range | Any finite value. Negatives (Yellowmoon keeps them down to −0.05) and values above 1.0 (R5 output) are handled as §3.1 and §4 say. NaN or infinite input is an error. |
| Size | width ≥ 2 and height ≥ 2 (mirroring needs 2 samples to keep the CFA parity). There is no upper limit and no even-size requirement. |
| Output | `ym_gpu::cpu::RgbImage`: **planar** f32 (all R, then all G, then all B), normalised, **white-balanced**. Channel c of a scene value v is `gains[c]·v`, so white comes out as `g0`. There is no clamp at 0 or 1. The output keeps the input `Placement`. |
| Clip mask | Not read and not changed. It travels on to C7 as before. |

### 1.3 White balance: the formula chain

The algorithm needs both unbalanced and balanced values. It takes the unbalanced mosaic and the gains, and applies them where its maths needs them:

```
R4 (Yellowmoon):   m_bal = g0[c] · m                          (balanced mosaic, R5 works on it)
R6 entry:          m_u   = m_bal / g0[c]                       (exact inverse of R4)
                   M     = S · max(m_u, 0)                     (DN, S = 65535; §3.1)
pass 1 (green):    ρ_R = g_R / g_G at R sites, ρ_B = g_B / g_G at B sites.
                   ρ scales same-colour samples to green in the direction tests and costs only.
                   The ratio estimate itself uses the unbalanced M.        → green G1 (unbalanced)
pass 2 (refine):   no gains                                               → green G2 (unbalanced)
pass 3 (chroma):   planes R = g_R·M (R sites), G = g_G·G2, B = g_B·M (B sites)   → balanced from here on
                   luma proxy of the sharpening divides by the gains again (it works on unbalanced values)
                   bilinear reference of shadow matching: Gauss3(native M) · (4g_R, 2g_G, 4g_B)
                   noise variance of a balanced sample of colour c: g·g_c·x_bal + (r·g_c)²
exit:              out_c = (P3_c + [c = native colour]·g_c·S·min(m_u, 0)) / S      (balanced, normalised)
```

With `g0 = [g_R, 1, g_B]`, `ρ_R = g_R` and `ρ_B = g_B`. The output is camera RGB × `g0`, which is what R6 must hand to R7.

### 1.4 Working scale and scale-dependent quantities

The absolute constants are tuned for a 16-bit working scale. Everything is defined at **S = 65535**:

```
x_dn = S · x          (mosaic and every intermediate plane)
g    = S · a          (shot-noise gain, DN per DN)
r    = S · sqrt(b + sigma_row² + q²/12)       (signal-independent noise sigma, DN)
σ_dn²(x_dn) = g · x_dn + r²
out  = out_dn / S
```

| Scale-dependent (defined in DN at S = 65535) | Dimensionless |
|---|---|
| K, T, k, g, r and the caps 65536 / 65535 | ρ, 1.6, 0.75, 0.03, 5, 0.75, 18, 102, the filter weights, 0.5 |
| The "+1" in every cost (pass 1 costs, pass 3 `1 + …` costs) | the ratios 4T and 3T in the shadow gate |
| The "+1" of the saturation penalty `max(X + G + 1, 1)` | CA constants 0.1 px and 2.5 (in pixels, not DN) |
| The luma floor and offset in `Y = max(… + 1, 1)` | |
| The flat test `d² < ΣG` (unit-gain Poisson noise in DN) | |
| The "+1" in `u = 3D + 1`, `v = V + 1` (shadow matching) | |

If Yellowmoon's normalisation changes, keep S = 65535 relative to white. Do not rescale the constants.

### 1.5 Halo

| Part | Halo | Why |
|---|---|---|
| Pass 1 | 5 px | the direction test reads the mosaic at (0, ±5), (±5, 0) and the 5×5 same-colour grid at ±4 with neighbours at ±5 |
| Pass 2 | 2 px | it reads pass-1 green and native samples up to (±2, ±2), (±1, ±2), (±2, ±1) |
| Pass 3 without shadow matching | 4 px | step 1 reads own-colour samples at (±2, ±2); step 2 reads step-1 results at ±1; the sharpening reads luma at the diagonal neighbours |
| Pass 3 with shadow matching | 6 px | adds the 5×5 Gaussian of the result (±2) |
| **Total** | **11 px** without shadow matching, **13 px** with | sum of the per-pass halos |

The sum is a safe bound. The perturbation test in the self-test measures a reach of 9 px (plain, with the shadow sharpening active) and 10 px (with shadow matching). `Method::halo()` returns 13, the maximum. The CA path does not change the halo. As usual, the scheduler puts mosaic rectangles on even coordinates.

---

## 2. Parameters

### 2.1 Inputs

| Name | Type | Default | Range | Meaning |
|---|---|---|---|---|
| `gains` | `[f32; 3]` | — | finite, > 0 | absolute white-balance gains; Yellowmoon passes `g0` |
| `noise` | `NoiseModel` → `NoiseDn { g, r }` | — | finite, ≥ 0 | §1.4. With a zero model, K and T floor at 1 DN (pure ratio interpolation, no shadow sharpening) |
| `detail` | f32 | 0 | 0…100 | lowers the pass-3 SNR target s (§2.2) |
| `strength` | f32 | 0 | 0…100 | lowers the pass-3 SNR target s (§2.2) |
| `shadow_match` | enum `Auto \| On \| Off` | `Auto` | — | shadow colour matching (§3.4.5); `Auto` = on when the SNR at 18 % grey is below 25 dB |
| `shadow_sharpening` | bool | `true` | — | diagonal shadow luma sharpening (§3.4.4). It is on in the reference behaviour; the switch is for A/B testing |
| `ca` | `Option<CaCurves>` | `None` | — | phase 2 only (§3.6); `None` = plain path |

`CaCurves { centre_x, centre_y, red: RadialCurve, blue: RadialCurve }` uses frame pixel coordinates. The buffer origin comes from `Placement`. A `RadialCurve` is a list of knots `(radius_px, shift_px)` with strictly increasing radius, at least 2 of them.

**`detail` and `strength`.** They enter only through their sum:

```
s = max((100 − (detail + strength)) · 0.2, 1)         default: (100 − 0)·0.2 = 20
```

s is the SNR at which pass 3 changes from colour-difference-like to colour-ratio-like interpolation. It is also the level that defines the shadow-sharpening gate. Raising either parameter lowers s, and with it T and k (§2.2). Colour edges are then rebuilt by ratio down to darker levels: colour transitions in shadows get crisper, at the cost of more chroma noise. The luma sharpening also moves to darker areas. The two names are kept so that two controls can be mapped independently later (for example a user detail preference and a per-profile bias) without changing the formula. Defaults of 0 give s = 20, the reference behaviour. A sum ≥ 95 gives the floor s = 1.

### 2.2 Derived values (computed once per frame, f64)

```
g, r            from the noise model (§1.4)
ρ_R = g_R/g_G,  ρ_B = g_B/g_G

K  = clamp(50 · (g + sqrt(g² + 0.04·r²)), 1, 65536)              pass 1 offset
     (the signal level x with x/σ(x) = 10:  x² = 100·(g·x + r²))

s  = max((100 − (detail + strength)) · 0.2, 1)
T  = clamp(s²/2 · (g + sqrt(g² + 4·r²/s²)), 1, 65535)            pass 3 offset and shadow gate
     (the signal level x with x/σ(x) = s)
k  = T

x18      = 0.18 · 65535 = 11796.3
SNR18_dB = 10·log10(x18² / (g·x18 + r²))
shadow matching (Auto) on  ⇔  SNR18_dB < 25
                           ⇔  g·11796.3 + r² > 440 039.45                  (DN)
                           ⇔  0.18·a + (b + sigma_row² + q²/12) > 1.0246e-4 (normalised)
shadow_match = On → on, Off → off.
halo = 13 if shadow matching is on, else 11
```

The SNR test uses the unbalanced (green) noise model, the same for all CFA positions. Implement it as the product comparison `(g·x18 + r²)·10^2.5 > x18²` (10^2.5 = 316.22776601683796), not through a logarithm.

Examples (the vectors use them):

| Model | g | r | K | T (s = 20) | SNR18 | shadow matching |
|---|---|---|---|---|---|---|
| low: a = 2e-5, b = 1e-8, q = 1/65535 | 1.311 | 6.56 | 158.3 | 555.3 | 39.5 dB | off |
| high: a = 5e-4, b = 4e-5, sigma_row = 1e-3 | 32.77 | 419.6 | 6143 | 17202 | 23.9 dB | on |

---

## 3. Algorithm

### 3.0 Notation and conventions

- Offsets are `(dx, dy)`: x to the right, y down. `P(dx, dy)` is the DN sample at that offset from the site being computed.
- `|·|` is the absolute value. Sums are evaluated left to right, as written. Comparisons are strict unless written `≤`. The order matters only for bit-exact reproduction (§3.7).
- Every per-pixel formula uses f64. Every plane between steps is stored as f32 (DN). `round32(·)` (rounding to f32) happens exactly when a value is written to a plane.
- Reads outside the buffer mirror at its edge without repeating the edge sample (§3.1). After the entry padding, mirroring only affects pixels that are later discarded.

### 3.1 Entry: scale, clamp, pad, phase

1. **Scale and clamp.** `M(x, y) = round32(S · max(m_u(x, y), 0))`. The ratio models below assume `value + offset > 0`, and the flat test in pass 1 treats the sum of four greens as a Poisson variance. The part removed by the clamp is added back at the exit (§3.5), so measured values are never lost.
2. **Pad once.** Let `H` be the halo (11 or 13) and let `(rx, ry)` be the column and row parity of the red photosites (table below). Build a padded buffer:
   - left pad `pad_l = H + ((H + rx) mod 2)`;
   - top pad `pad_t = H + ((H + ry) mod 2)`;
   - right and bottom pads of `H`.
   Each padded sample is `M(mirror(X − pad_l, width), mirror(Y − pad_t, height))`, using Yellowmoon's `mirror` (`−1 → 1`, `n → n − 2`, repeated for pads larger than the buffer).
3. **Phase.** The left and top pads have the parity of the red photosites, so in padded coordinates the pattern is always RGGB: R at (even, even), B at (odd, odd), G elsewhere. All formulas below assume this. Equivalently, an implementation without padding can classify the photosite (x, y) of the buffer by `((x − rx) mod 2, (y − ry) mod 2)`.

| `Cfa` | red at (rx, ry) | pad_l, pad_t for H = 11 | for H = 13 |
|---|---|---|---|
| RGGB | (0, 0) | 12, 12 | 14, 14 |
| GRBG | (1, 0) | 11, 12 | 13, 14 |
| GBRG | (0, 1) | 12, 11 | 14, 13 |
| BGGR | (1, 1) | 11, 11 | 13, 13 |

All passes are evaluated on the whole padded buffer. Only the core is returned.

### 3.2 Pass 1: green

At every G site, `G1 = M`. At every R or B site with value `c = P(0,0)`:

- `ρ = ρ_R` at R sites, `ρ_B` at B sites.
- The adjacent greens are `G_u = P(0,−1)`, `G_d = P(0,1)`, `G_l = P(−1,0)`, `G_r = P(1,0)`.
- Same-colour samples scaled to green are `Q(dx,dy) = ρ·P(dx,dy)` (dx, dy even).
- The centre colour differences are `Δ_u = ρc − G_u`, `Δ_d = ρc − G_d`, `Δ_l = ρc − G_l`, `Δ_r = ρc − G_r`.
- `ca = |shift_R(r) − shift_B(r)|` at the site (§3.6), or 0 on the plain path.

**Direction scores.** A term `|Δ_x + G − Q|` compares the colour difference at the centre with the colour difference at a same-colour sample q, both taken with the green on the same side.

```
B_v = |G_l − P(−1,−2)| + |G_l − P(−1,2)| + |G_r − P(1,−2)| + |G_r − P(1,2)| + (1.6·ρ)·|P(0,−2) − 2c + P(0,2)|
B_h = |G_u − P(−2,−1)| + |G_u − P(2,−1)| + |G_d − P(−2,1)| + |G_d − P(2,1)| + (1.6·ρ)·|P(−2,0) − 2c + P(2,0)|

U = B_v + |Δ_u + P(0,−5) − Q(0,−4)| + |Δ_u + P(0,3) − Q(0,4)|
D = B_v + |Δ_d + P(0,−3) − Q(0,−4)| + |Δ_d + P(0,5) − Q(0,4)|
V = B_v + |P(0,−5) − P(0,−3)| + |P(0,3) − P(0,5)|
L = B_h + |Δ_l + P(−5,0) − Q(−4,0)| + |Δ_l + P(3,0) − Q(4,0)|
R = B_h + |Δ_r + P(−3,0) − Q(−4,0)| + |Δ_r + P(5,0) − Q(4,0)|
H = B_h + |P(−5,0) − P(−3,0)| + |P(3,0) − P(5,0)|

for dy in (−2, 0, 2):            # 3×3 same-colour grid, centre included
  for dx in (−2, 0, 2):
    q = Q(dx,dy)
    L += |Δ_l + P(dx−1,dy) − q|      R += |Δ_r + P(dx+1,dy) − q|
    U += |Δ_u + P(dx,dy−1) − q|      D += |Δ_d + P(dx,dy+1) − q|
    H += |P(dx−1,dy) − P(dx+1,dy)|   V += |P(dx,dy−1) − P(dx,dy+1)|
```

**Decision.** Type 1 interpolates from the left and right greens, type 2 from the upper and lower greens, type 3 from all four.

```
f = 1 + 2.5·ca³  if ca ≥ 0.1, else 1                 (CA relaxation; f = 1 on the plain path)
if max(L,R) < f·min(U,D)  and  H < V:   type 1
elif f·min(L,R) > max(U,D) and V < H:   type 2
else:
  sH = G_l + G_r,  sV = G_u + G_d,  d = |sH − sV| / 2
  if d·d < sH + sV:                     type 3       # pair means agree within ~2σ of unit-gain Poisson noise
  else:                                 # 5×5 fallback
    L5 = R5 = U5 = D5 = 0
    for dy in (−4, −2, 0, 2, 4):
      for dx in (−4, −2, 0, 2, 4):
        q = Q(dx,dy)
        L5 += |Δ_l + P(dx−1,dy) − q|    R5 += |Δ_r + P(dx+1,dy) − q|
        U5 += |Δ_u + P(dx,dy−1) − q|    D5 += |Δ_d + P(dx,dy+1) − q|
    if max(L5,R5) < min(U5,D5):          type 1
    elif min(L5,R5) > max(U5,D5):        type 2
    else:                                type 3
```

**Estimate.** Each direction gets a cost (always ≥ 1):

```
w_u = ρ·|P(0,−2) − c| + |P(0,−3) − G_u| + |P(−1,−2) − G_l| + |P(1,−2) − G_r| + |G_u − G_d| + 1
w_d = ρ·|P(0,2) − c|  + |P(0,3) − G_d|  + |P(−1,2) − G_l|  + |P(1,2) − G_r|  + |G_u − G_d| + 1
w_l = ρ·|P(−2,0) − c| + |P(−3,0) − G_l| + |P(−2,−1) − G_u| + |P(−2,1) − G_d| + |G_l − G_r| + 1
w_r = ρ·|P(2,0) − c|  + |P(3,0) − G_r|  + |P(2,−1) − G_u|  + |P(2,1) − G_d|  + |G_l − G_r| + 1

C_u = (P(0,−2) + c)/2   C_d = (P(0,2) + c)/2   C_l = (P(−2,0) + c)/2   C_r = (P(2,0) + c)/2
```

The weight of each green is the product of the costs of the *other* directions in use. It is therefore proportional to the inverse of its own cost. Unused directions get weight 0:

```
type 1:  α_l = w_r,  α_r = w_l,  α_u = α_d = 0
type 2:  α_u = w_d,  α_d = w_u,  α_l = α_r = 0
type 3:  α_u = w_d·w_l·w_r,  α_d = w_u·w_l·w_r,  α_l = w_u·w_d·w_r,  α_r = w_u·w_d·w_l

Σ  = α_u + α_d + α_l + α_r
NG = α_u·G_u + α_d·G_d + α_l·G_l + α_r·G_r
NC = α_u·C_u + α_d·C_d + α_l·C_l + α_r·C_r
num = max(K·Σ + NG, K·Σ)
den = max(K·Σ + NC, K·Σ)
G1  = round32((c + K)·(num/den) − K)
```

This is the regularised colour ratio `G1 + K = (c + K)·Σα_j(G_j + K)/Σα_j(C_j + K)`. It works like colour-difference interpolation when K is large compared with the signal, and like colour-ratio interpolation when K is small. Both sums are floored at `K·Σ`, so `ΣαG` and `ΣαC` are treated as ≥ 0. With K ≥ 1 and costs ≥ 1, `den > 0`.

### 3.3 Pass 2: green refinement

At G sites, `G2 = G1`. At every R or B site, with G the pass-1 plane and `C(dx,dy)` the native samples `M`:

```
G0 = G(0,0)   N = G(0,−1)  S = G(0,1)  W = G(−1,0)  E = G(1,0)          (native greens)
NW = G(−1,−1) NE = G(1,−1) SW = G(−1,1) SE = G(1,1)                     (pass-1 greens at the other colour)
C0 = C(0,0)

V  = |N − G0| + |S − G0|
H  = |W − G0| + |E − G0|
s2 = |S − E| + |N − W| + |C(2,−2) − C0| + |C(−2,2) − C0| + |C(−1,1) − C(1,−1)|      "/" diagonal
s3 = |S − W| + |N − E| + |C(−2,−2) − C0| + |C(2,2) − C0| + |C(−1,−1) − C(1,1)|      "\" diagonal
s4 = |S − NE| + |N − SW| + |G(−1,2) − G0| + |G(1,−2) − G0|                           knight (1,−2)
s5 = |E − SW| + |W − NE| + |G(−2,1) − G0| + |G(2,−1) − G0|                           knight (2,−1)
s6 = |N − SE| + |S − NW| + |G(−1,−2) − G0| + |G(1,2) − G0|                           knight (1,2)
s7 = |W − SE| + |E − NW| + |G(−2,−1) − G0| + |G(2,1) − G0|                           knight (2,1)

m0 = (G0 + N + S)/3              m1 = (G0 + W + E)/3
m2 = (2·G0 + NE + SW)/4          m3 = (2·G0 + NW + SE)/4
m4 = (G0 + G(−1,2) + G(1,−2))/3  m5 = (G0 + G(−2,1) + G(2,−1))/3
m6 = (G0 + G(−1,−2) + G(1,2))/3  m7 = (G0 + G(−2,−1) + G(2,1))/3

best = 1 if H < V else 0;  low = min(V, H) (H if H < V, else V)
for j = 2..7:  if s_j < low: best = j, low = s_j        # strict: ties keep the lower index
axis = m0 if V < H else m1                               # note: V = H selects m1

G2 = G0                         if best ≤ 1
G2 = round32((axis + m_best)/2) otherwise
```

Along vertical or horizontal structure the pass-1 value is kept bit for bit. On oblique structure the result averages the 3-tap axis mean with the mean along the winning oblique direction. Native greens and the R/B samples are never written.

The CA-aware variant of this pass is in §3.6.

### 3.4 Pass 3: chroma

#### 3.4.1 Gains

```
G(x,y) = round32(g_G · G2(x,y))                  everywhere
R(x,y) = round32(g_R · M(x,y))                   at R sites
B(x,y) = round32(g_B · M(x,y))                   at B sites
```

From here on the planes are white-balanced. `k` is the offset of §2.2.

#### 3.4.2 Step 1: B at R sites, R at B sites

At an R site, `X` is the B plane and `Y` is the R plane. At a B site, `X` is the R plane and `Y` is the B plane. The diagonal neighbours carry X natively.

```
xUL = X(−1,−1)  xUR = X(1,−1)  xDL = X(−1,1)  xDR = X(1,1)      gUL … gDR likewise from G
g0 = G(0,0),  y0 = Y(0,0)

if |xUL − xDR| < |xDL − xUR|:  A = (−1,−1), B = (1,1)            # "\" is smoother
else:                          A = (−1,1),  B = (1,−1)
cA = 1 + |G_A − g0| + |Y(2A) − y0|          # 2A = (−2,−2) or (−2,2): own colour two steps out
cB = 1 + |G_B − g0| + |Y(2B) − y0|
if 0.75·(xUL + xUR + xDL + xDR) < gUL + gUR + gDL + gDR:          # X not dominant over G
    cA = cA · (|X_A − G_A| / max(X_A + G_A + 1, 1) + 0.03)        # penalise high-chroma samples
    cB = cB · (|X_B − G_B| / max(X_B + G_B + 1, 1) + 0.03)
t   = (cA + cB)·k
den = max(G_B·cA + G_A·cB + t, t)
num = X_B·cA + X_A·cB + t
X(0,0) = round32((g0 + k)·(num/den) − k)
```

Each sample is weighted by the other sample's cost. This is the offset ratio `(g0 + k)·Σw(X + k)/Σw(G + k) − k` with `w ∝ 1/cost`. The step reads only native X samples, so the order in which sites are processed does not matter.

#### 3.4.3 Step 2: R and B at G sites

At a G site, the four axis neighbours are R or B sites. After step 1 they carry both R and B.

```
g0 = G(0,0);  G_U = G(0,−1), G_D = G(0,1), G_L = G(−1,0), G_R = G(1,0)
cost_U = 1 + |G_U − G_D| + |G(0,−2) − g0|      cost_D = 1 + |G_U − G_D| + |G(0,2) − g0|
cost_L = 1 + |G_L − G_R| + |G(−2,0) − g0|      cost_R = 1 + |G_L − G_R| + |G(2,0) − g0|
ΣG = G_U + G_D + G_L + G_R

for X in (R, B), each starting from the same four costs:
    X_U = X(0,−1), X_D = X(0,1), X_L = X(−1,0), X_R = X(1,0)
    c_j = cost_j
    if 0.75·(X_U + X_D + X_L + X_R) < ΣG:
        c_j = c_j · (|G_j − X_j| / max(G_j + X_j + 1, 1) + 0.03)      for j in U, D, L, R
    w_j = 1 / c_j
    t   = k·(w_U + w_D + w_L + w_R)
    den = max(Σ_j w_j·G_j + t, t)            (sums in the order U, D, L, R)
    num = Σ_j w_j·X_j + t
    X(0,0) = round32((g0 + k)·(num/den) − k)
```

Compute both colours before writing either. The step reads R and B only at R/B sites.

#### 3.4.4 Step 3: shadow luma sharpening (`shadow_sharpening`, default on)

This step is a soft-limited sharpening along the smoother diagonal. It acts only where every green in the 3×3 neighbourhood is below T, and fades out by 4T. All three channels get the same factor, so colour ratios are kept.

```
gmax(x,y) = max(0, max of M over the G sites of the 3×3 window)          (native greens, unbalanced DN)
Yl(x,y)   = round32(max(R/g_R + 2·G/g_G + B/g_B + 1, 1))                 (unbalanced luma proxy, ×4 scale)

per pixel:
  f = clamp((4T − gmax)/(3T), 0, 1)                  # 1 up to T, linear to 0 at 4T
  if f = 0: leave the pixel unchanged
  y0 = Yl(0,0)
  pm = Yl(−1,−1) + Yl(1,1);   pa = Yl(1,−1) + Yl(−1,1)
  p  = pm if |2·y0 − pm| < |2·y0 − pa| else pa      # the diagonal with the smaller second difference
  d  = 5·(2·y0 − p)                                  # diagonal Laplacian
  c  = min(y0, r)                                    # limit: the noise sigma r (DN)
  a  = c·d/(c + |d|)   (0 if c + |d| = 0)
  gain = (y0 + 0.75·f·a)/y0
  R, G, B ← round32(R·gain), round32(G·gain), round32(B·gain)
```

Compute `gmax` and `Yl` for the whole buffer before changing any pixel. The added luma is at most `0.75·min(y0, r)`, so only detail at noise amplitude is sharpened. The small-signal gain is `3.75·(2y0 − p)`.

#### 3.4.5 Step 4: shadow colour matching (when §2.2 enables it)

The step replaces the low-pass part of the result with the low-pass of a bilinear demosaic, wherever the difference between the two can be explained by noise. In noisy shadows this removes colour mazes and false chroma. High frequencies of the result are always kept.

Let `S3_c` be the three planes after step 3.

```
sparse_c(x,y) = M(x,y) if (x,y) is a site of colour c, else 0            (native samples, unbalanced)
simple_c = Gauss3(sparse_c) · m_c,   m = (4·g_R, 2·g_G, 4·g_B)           (bilinear demosaic, balanced)
L_c   = Gauss3(simple_c)
Mlp_c = Gauss5(S3_c)
E0_c  = (S3_c − L_c)²

per pixel:
  E = Box3(E0_R) + Box3(E0_G) + Box3(E0_B)
  V = Σ_c [ g·g_c·max(L_c, 0) + (r·g_c)² ]          (expected variance of one balanced sample, summed over c)
  D  = max(E − V/2, 0)
  u  = 3·D + 1,   v = V + 1,   m = max(u, v)
  u' = max(u − m/2, 0),   v' = max(v − m/2, 0)
  w  = v' / (u' + v')        # = 1 for u ≤ v/2, v/(2u) for v/2 < u ≤ v, 1 − u/(2v) for v < u < 2v, 0 for u ≥ 2v
  out_c = round32(S3_c + w·(L_c − Mlp_c))
```

The filters are evaluated in f64, and each output plane is stored as f32:

- `Gauss3(p)(x,y) = (Σ_{dy=−1..1} k_dy·(p(x−1,y+dy) + 2·p(x,y+dy) + p(x+1,y+dy)))/16 · scale`, with `k = (1, 2, 1)` and `scale = m_c` for `simple_c`, 1 for `L_c`.
- `Gauss5(p)(x,y) = (Σ_{dy,dx=−2..2} W[dy][dx]·p(x+dx,y+dy))/571`, with rows top to bottom and
  `W = [[2,7,12,7,2],[7,31,52,31,7],[12,52,127,52,12],[7,31,52,31,7],[2,7,12,7,2]]` (sum 571; not separable).
- `Box3(p)(x,y) = (Σ_{dy,dx=−1..1} p(x+dx,y+dy))/9`.
- The planes `simple`, `L`, `Mlp` and `E0` are stored as f32.

Since `max(u, v) ≥ 1`, `u' + v' ≥ 1/2` and the division is safe. Without shadow matching, the output of pass 3 is `S3`.

### 3.5 Exit

```
out_c(x,y) = round32( (P3_c(x + pad_l, y + pad_t) + [c = colour(x,y)]·g_c·S·min(m_u(x,y), 0)) / S )
```

The clamp residual `min(m_u, 0) ≤ 0` goes back into the photosite's own channel, white-balanced. A photosite measured at −0.03 therefore keeps its negative contribution in its own channel. The interpolated channels around it are computed from the clamped value 0. The output is planar R, G, B.

### 3.6 CA-aware path (optional, phase 2)

The plain path (`ca = None`) is the default and is what RA-03 ships. The CA path changes two things:

- **Pass 1:** the relaxation factor `f` of the direction decision (§3.2).
- **Pass 2:** the scores and the blend (below).

**Input.** Two radial curves, `shift_R(r)` and `shift_B(r)`, give the lateral displacement in pixels of the R and B planes relative to G, as a function of the radius r in pixels from the optical centre `(centre_x, centre_y)`. Coordinates are frame pixels, and pixel n covers [n, n + 1). Each curve is a knot list with strictly increasing radius, at least 2 knots. Between knots it is interpolated linearly. Beyond the first and last knot it is extrapolated linearly from the end segments.

**Per site.** The photosite at buffer (x, y) sits at frame position `(place.x0 + x + 0.5, place.y0 + y + 0.5)`.

```
dx = place.x0 + x + 0.5 − centre_x,  dy = place.y0 + y + 0.5 − centre_y,  r = sqrt(dx² + dy²)
ca = |shift_R(r) − shift_B(r)|          (R/B misregistration at the site, px; used by pass 1, unclamped)
κ  = min(ca, 1)                         (pass 2 weight)
```

**Pass 2 with CA.** Same neighbourhood, scores `s2…s7` and means as in §3.3, except:

```
V = |N − G0| + |S − G0| + κ·(|C(0,−2) − C0| + |C(0,2) − C0|)
H = |W − G0| + |E − G0| + κ·(|C(−2,0) − C0| + |C(2,0) − C0|)
(best, axis as in §3.3 with these V and H)

best = 0:  t = (1 − min(1, 2κ))²,  w = 1 + 18·t + 102·t³,  G2 = (w·G0 + N + S)/(w + 2)
best = 1:  same w,                                          G2 = (w·G0 + W + E)/(w + 2)
best ≥ 2:  G2 = (axis + (2κ + 1)·m_best)/(2κ + 2)
```

In words: the vertical and horizontal scores also see same-colour R/B gradients, scaled by κ. The vertical/horizontal case becomes a 3-tap mean along the winning axis, which gets stronger as κ grows: plain 3-tap from κ = 0.5. The oblique blend leans more toward the oblique mean. At κ = 0 the oblique case equals the plain path. The axis case still keeps a small smoothing there, with weight 121/123 on G0 (see §7).

**Deriving the curves from Yellowmoon's R9 lens model** (tags 0x7035 and 0x7037):

- CA knots of colour c: `K_c,j = ca_j·2^−21 + 1`.
- Distortion knots: `F_j = d_j·2^−14 + 1`.
- Knot j of n sits at the normalised radius `(j + 0.5)/(n − 1)`, where 1 is the half diagonal `hd = sqrt(W_f² + H_f²)/2` of the frame. Values between knots are interpolated linearly; outside the knots the end values are held (`LinearKnots::sony`).
- R9 samples colour c at source radius `r·F(r)·K_c(r)` and green at `r·F(r)`, with `r` the output radius. So at the source (mosaic) radius `r_src = r·F(r)`, plane c is displaced by `r_src·(K_c(r) − 1)`.
- Without distortion this reduces to `shift_c(r) = r·(K_c(r) − 1)`.

Sample the curve every `step` = 8 px of output radius:

```
for j = 0 .. ceil((hd + 2·step)/step):
    r_out = j·step/hd                        (normalised)
    r_src = hd · r_out · F(r_out)            (F = 1 when 0x7037 is absent or lens corrections are off)
    knot j = (r_src, r_src·(K_c(r_out) − 1))
```

- `r_src` must increase strictly. A distortion table that breaks this is an error, and the CA path is then off.
- `centre = (frame_width/2, frame_height/2)`, the R9 centre.
- Use the CA knots only when `decode.chromatic_aberration` is on.
- The sign convention does not matter, because only `|shift_R − shift_B|` is used.
- The reference provides `curve_from_lens_knots()`. The vector `ca_weighting` carries both the knots and the derived curves.

**Scale.** The curves are in pixels of the mosaic being demosaiced. If R6 ever runs on a binned or downscaled mosaic, scale both radius and shift by the same factor, or disable the CA path. The reference behaviour applies the CA path only at 1:1.

### 3.7 Numeric precision

- **f64 everywhere.** Use f64 for every per-pixel formula and for the derived values. In pass 1 the type-3 weights are products of three costs. With high-contrast data a cost reaches ~1e6 DN, so a product reaches ~1e18. That is fine in f64. In f32 it still fits the range but loses the small terms.
- **f32 planes.** Store the planes as f32 in DN, rounding once at each write: M, G1, G2, the pass-3 R/G/B planes, `gmax`, `Yl`, and the shadow-matching planes.
- **Bit-for-bit reproduction.** A port that follows §3.0–§3.6 with this precision, evaluation order and tie rules reproduces the vectors bit for bit or within a few f32 ulps. The decision comparisons (pass 1 types, pass 2 argmin, pass 3 diagonal choice) are exact under these rules.
- **f32 ports (GPU).** These see isolated decision flips at near-ties (§5).
- **Data path.** Only `+ − × ÷`, `sqrt`, `min`, `max` and `abs` are used on the data path, so results do not depend on the platform's libm. Do not fuse multiply-adds in the reference.

---

## 4. Edge cases

| Case | Behaviour |
|---|---|
| Frame borders | Mirror once at entry (§3.1), parity-preserving, then compute every pass on the padded buffer. This differs from mirroring each pass's output separately only inside the outer halo ring of the frame. Tiles get the same values because their halo is real data. |
| Tiny regions | width or height < 2 is an error (`BadSize`). Buffers smaller than the halo are fine: the mirror repeats. The vector `tiny_5x3` covers this. |
| Zero input | All-zero in gives exact zeros out: with K, T ≥ 1 every ratio is `(0 + K)·(KΣ/KΣ) − K = 0`, `Yl = 1` gives a sharpening gain of 1, and the matching planes are 0. |
| Negative input | Clamped at 0 for all three passes. The residual goes back into the photosite's own channel at the exit (§3.5). The residual never goes into interpolated values. Justification: the offset-ratio formulas need `value + K > 0` and `value + k > 0`, and the flat test needs a non-negative green sum. Yellowmoon's negatives are noise around black, and restoring them in the measured channel keeps that channel's noise statistics for R6's denoise. A constant negative field comes out as 0 in the interpolated channels and as `g_c·value` in the native channel. |
| Values above 1.0 | Allowed, with no upper clamp anywhere. The caps on K (65536) and T (65535) bound the offsets, not the data. Above white the arithmetic is the same, and f64 keeps the products exact enough. The shadow gate is closed there (gmax ≫ 4T). |
| NaN or infinite input | `BadInput`. Non-positive or non-finite gains: `BadGains`. Negative or non-finite noise fields: `BadNoise`. Invalid CA curves: `BadCurve`. |
| Zero noise model | K = T = 1 DN (floors): near-pure ratio interpolation, and the sharpening gate is closed for anything above 4 DN. Allowed; `demosaic(m, method)` without a noise model uses it. |
| CFA phases | All four, through the pad parity (§3.1). Results are equivariant: a buffer cropped one photosite to the right is processed as the next phase and gives identical interior pixels. |
| Unusable noise model in a render | The algorithm refuses it (`BadNoise`), a render never does: the chain reads a model that fails `NoiseDn::is_valid` (negative or non-finite numbers, or a variance that overflows at white) as the zero model, K = T = 1 DN, and renders the photo with that. `INTEGRATION.md` section 10. |
| Non-Bayer patterns | Not supported. Fall back to Menon. |
| CA path off-centre or outside the curve range | Linear extrapolation of the curves. `ca` can exceed 1 px; pass 2 clamps it to κ ≤ 1, pass 1 uses it unclamped. |

---

## 5. Determinism, tiling, performance, GPU notes

- **Determinism.** No reductions, no data-dependent iteration order, and every tie rule is fixed. Each output pixel is a pure function of its neighbourhood and the frame constants. The derived constants (K, T, k, shadow-matching flag, halo) are per frame and come from the noise model, not from image statistics. So the result does not depend on band, tile or thread order.
- **Tiling.** A tile with ≥ 11 px of real context (≥ 13 with shadow matching) gives the whole-frame result bit for bit. The self-test measures a difference of 0 for 24×20 tiles on a 77×61 frame in all three configurations. Frame-edge tiles mirror at the frame edge like the whole frame. The CA path uses `Placement` for absolute positions.
- **Cost.** These counts are for each output pixel:

  | Part | Work |
  |---|---|
  | Pass 1, per R/B site | ~100 absolute differences (3×3 grid, extensions and costs). The 5×5 fallback (~100 more) runs only when the first test and the flat test do not decide. |
  | Pass 2, per R/B site | ~40 operations |
  | Pass 3 steps 1–3 | ~60 operations |
  | Shadow matching | two 3×3 Gaussians, one 5×5 Gaussian per channel, a box filter |

  The reference (unoptimised, mirror on every read, one thread) takes 0.16 s/MP, or 0.28 s/MP with shadow matching. A Rust port with direct indexing on the padded buffer and band parallelism should land near Menon's cost.
- **Memory.** About 9 f32 planes of the padded tile: M, G1, G2, R, G, B, gmax, Yl and the outputs. Shadow matching adds 12 more transient planes, one per channel × 4.
- **GPU port.** Every step is a per-pixel gather with no reductions. The CPU computes the frame constants (K, T, k, ρ, flags) and passes them as uniforms. Kernel sequence:
  1. mirror pad (or index mirroring);
  2. pass 1;
  3. pass 2 (CA weight evaluated per pixel from a small curve buffer);
  4. gains + step 1;
  5. step 2;
  6. `gmax` + `Yl`;
  7. sharpening gain;
  8. if matching: sparse Gauss3, Gauss3, Gauss5, squared error, box, blend.

  To match the CPU reference at frame edges, intermediate passes must be evaluated on the padded region. Do not mirror each kernel's input at the frame edge; per-kernel mirroring changes only the outer halo ring of the frame. In f32, prefer the normalised inverse-cost form in pass 1, `α_j = 1/w_j` (identical in exact arithmetic), and expect isolated decision flips at near-ties. The parity test should therefore report the max abs error and the fraction of samples above 1e-4, not require bit equality.

---

## 6. Test plan

### 6.1 Vectors (`vectors/adaptive_ratio_demosaic/*.json`)

Each vector has these parts:

| Part | Contents |
|---|---|
| `params` | `gains`, `noise {a, b, sigma_row, q}`, `detail`, `strength`, `shadow_match`, `shadow_sharpening`, `ca` (null or centre, origin, curves, plus the lens knots they came from), and `derived {rho_r, rho_b, g_dn, r_dn, K, s, T, k, snr18_db, shadow_match, halo}` |
| `input` | `width`, `height`, `cfa`, row-major **unbalanced** normalised mosaic |
| `expected` | planar RGB, balanced, normalised |

Tolerance: `1e-5` absolute per output sample in normalised units. The derived values should match to 1e-9 relative. Doubles are printed with 17 significant digits and floats with 9, so the inputs round-trip exactly.

| Vector | Size | Covers |
|---|---|---|
| `flat_field` | 16×16 | constant colour: output = gain × value exactly |
| `edge_horizontal`, `edge_vertical` | 24×24 | pass-1 direction types 1 and 2 at colour edges |
| `edge_diagonal` | 24×24 | 45° edge: pass-2 diagonal directions |
| `edge_knight` | 24×24 | 2:1 edge: pass-2 knight directions |
| `zone_plate` | 32×32 | frequencies up to Nyquist, correlated colour |
| `noisy_dark_shadow_match` | 32×32 | high-noise model (SNR18 23.9 dB): `Auto` turns matching on, halo 13; negatives down to −0.017 |
| `noisy_dark_no_match` | 32×32 | same input, `shadow_match = Off` |
| `highlights_and_negatives` | 24×24 | samples up to ~3.05 and a border column block down to −0.05: no clamp at 1, negative residual restore |
| `shadow_sharpening`, `shadow_sharpening_off` | 24×24 | dark texture under the 4T gate, with the step on and off |
| `snr_target_s8` | 24×24 | `detail = 40`, `strength = 20` give s = 8, plus added noise |
| `phase_rggb`, `phase_grbg`, `phase_gbrg`, `phase_bggr` | 40×40 each | one RGGB frame sampled at offsets (0/1, 0/1). Pixels ≥ 11 px from every edge agree across the four vectors after the shift (checked: 3675 samples, max difference 0) |
| `tiny_5x3` | 5×3 | GBRG, pad larger than the buffer |
| `ca_weighting` | 32×32 | phase 2: tile at frame (4800, 3330) of 6048×4024; curves derived from 16 CA knots per colour and 16 distortion knots; `ca` ≈ 0.68–0.72 px across the tile |

### 6.2 Property tests (Rust, mirroring the reference self-test)

- **Flat field.** All 4 phases × 3 gain sets × 3 levels (0.003 to 2.4) × {plain, matching on, CA}. Output = `gain·value` within 1.2e-7 relative; the reference measures 6.4e-8.
- **Balanced entry.** A balanced grey through `demosaic_balanced` comes back as the same grey (< 1e-7). `demosaic(m)` and `demosaic_balanced(m·g0)` agree (measured 3e-8).
- **CFA-phase equivariance.** Crop a frame at offsets (0/1, 0/1), so the four phases appear. The interiors (≥ halo from the crop edges) are identical to the whole-frame result: plain, matching and CA, all exact.
- **Tiling parity.** Tiles with the declared halo against the whole frame. The limit is 1e-4; the reference measures 0. Use odd frame sizes and a non-RGGB frame phase.
- **Receptive field.** Perturb single photosites. The changed outputs must stay within the declared halo. Measured: 9 px plain with sharpening active, 10 px with matching.
- **Scale behaviour.** Scale the input by λ and the noise model consistently (a·λ, b·λ², q·λ). The relative RMS change of `out(λm)/λ` against `out(m)` is reported, and must be below 2 %. Measured: 0.12 % at λ = 1/4 and 0.02 % at λ = 4. The method is not exactly scale-invariant because of the DN constants in §1.4.
- **Robustness.** Random inputs in [−0.05, 4], sizes 2×2 to 30×17, zero and extreme noise models, s floored at 1. Every output must be finite. NaN input, zero gains and bad curves are rejected.
- **Negative residual.** Compare a negative sample with the same sample at 0: the own channel differs by exactly `gain·value`.
- **Determinism.** `--jobs 1` and `--jobs 4` give byte-identical output.

### 6.3 Quality gate (RA-05)

- **PSNR.** Measure against Menon and LMMSE on Kodak, McMaster and PixelShift200 (and Gharbi hard/moiré where available), in linear light at signal scales 1, 0.1 and 0.01, as `docs/05` does for LMMSE. Exclude a 10 px border. Use unit gains and a nominal low-noise model (`a = 2e-5`, `b = 1e-8`, `q = 1/65535`). Report the noise model with the numbers, since K and T depend on it.
- **Other metrics.** Report zipper and false-colour metrics (`cpu/metrics.rs`) and CPU time per 50 MP.
- **Noisy inputs.** Add synthetic Poisson–Gaussian noise at ISO 100 / 1600 / 6400-like levels, pass the matching noise model, and report PSNR after the R6 bilateral denoise. This evaluates shadow matching and the shadow sharpening in the chain they actually run in.
- **Sanity check.** On the self-test's correlated texture scene (64×64, 12 px border), the reference gives 42.2 dB against 33.3 dB for bilinear.

---

## 7. Evaluation notes and open questions

**Evaluation notes** (reference behaviour kept; alternatives worth an A/B):

1. **Flat test.** The pass-1 flat test `d² < ΣG` assumes unit-gain Poisson noise in DN, whatever the noise model. Try the model-aware form `d² < g·ΣG + 4r²`.
2. **Unbalanced ratio.** The pass-1 ratio uses unbalanced values, while its costs and direction tests are balanced (via ρ). Because of the offset K the estimate is not gain-invariant. Try applying ρ to `c` and `C_j` and dividing it out of the result.
3. **CA relaxation in pass 1.** `f = 1 + 2.5·ca³` is unbounded: 3.5 px of misregistration gives f ≈ 108, and the first direction test then decides almost every site. Try capping `ca` at 1 px as pass 2 does.
4. **CA path at κ = 0.** Pass 2's CA path at κ = 0 still smooths the vertical/horizontal case (G0 weight 121/123), unlike the plain path, which keeps G0. Near the optical centre that is a slight softening. Try returning G0 when κ is below about 0.05.
5. **Shadow sharpening.** The luma proxy weights unbalanced R, G, B as 1:2:1, not as Yellowmoon's `LUMA` on balanced data. Also check how the step adds up with R8 capture sharpening, which sharpens shadows again, and with the R6 bilateral denoise. A/B `shadow_sharpening = false` with R8 on.
6. **Shadow matching and denoise.** Shadow matching and the R6 bilateral denoise both target shadow chroma noise. Measure whether matching plus a weaker bilateral beats the bilateral alone.
7. **`detail` and `strength`.** Test tying `detail` to R7's Detail control, for example `detail = max(0, D − 50)`. Keep the default 0 until measured.
8. **Negative residual.** The residual goes back into the own channel only. The alternative is a neutral offset `g_c·S·min(m_u, 0)` added to all three channels of that photosite. Compare the mean bias of dark frames after R6 + R7 for both.
9. **Saturation penalty.** The constants 0.75 (dominance test) and 0.03 (floor) are tuned for the reference behaviour. Evaluate them on saturated red/blue edges (flowers, signage) against Menon's false colour.

**Open questions, with the default to use meanwhile:**

| Question | Default |
|---|---|
| Should `Method::halo()` be configuration-dependent (11 vs 13)? | Return 13 |
| R5-reconstructed photosites (clip mask set): treat them differently? | No; treat them as normal samples |
| Use the per-position black offsets `delta_c` of the noise model? | No (they are tiny); `a·mu + b_total` only |
| Does the CA path interact with R9, which corrects lateral CA after R6? | Measured in RA-03b (section 11): the R9 residual is lower with the CA path on synthetic files that are consistent with their tables, and no higher (to within 0.001 px) on the real files that have tables; the CA path costs 4 to 8 % of the green gradient energy on those real files, where the tables do not match the raw data. Not the default until RA-H4 |
| Run the CA path on scaled or binned mosaics? | No; only at 1:1 |
| Should the zero-noise fallback (`demosaic(m, method)`) use a nominal model instead of K = T = 1? | Keep the floors; R6 always passes the real model |

---

## 8. Reference notes

The reference specifies the clean, symmetric, full-precision behaviour. Where an implementation could differ in rounding details, the reference does this:

- It uses exact division everywhere: no approximate reciprocals. In particular, the shadow-sharpening gain is exactly 1 where `f = 0`.
- The white-balance scaling of same-colour samples in the pass-1 direction test is exact (`ρ·P`), with no fixed-point quantisation of ρ. The second-derivative weight is exactly `1.6·ρ`.
- The averages `C_j = (c + P_j)/2` and the flat-test half-difference `|sH − sV|/2` are exact, with no integer truncation or rounding variant.
- The CA relaxation compares in floating point (`f·min(U, D)`), without rounding to an integer.
- The CA magnitude of every site comes from that site's own pixel centre.
- The CA magnitude is evaluated per pixel, not on a coarse grid with nearest-node lookup. The curves are knot lists with linear interpolation; a smooth spline through sparse lens-profile points is an acceptable input if it is sampled into knots first.
- The saturation-penalty denominator `max(X + G + 1, 1)` is applied to both diagonal samples (symmetric form).
- The input has no black pedestal term, because Yellowmoon's mosaic is black-subtracted.
- K and T are floored at 1 DN, so a degenerate noise model cannot divide by zero. T is computed with `4r²/s²` directly; s ≥ 1, so no guard term is needed.
- Every R/B site of the buffer is computed, for any width and height ≥ 2. Odd sizes and all four phases are handled by the padding parity.
- Borders are mirrored once at entry. Intermediate results are not re-mirrored per pass.
- Negative input is clamped at 0, and the residual is restored at exit (§3.5).
- All per-pixel arithmetic is f64, and every plane is stored as f32.

---

## 9. As built in `ym-gpu` (RA-03, plain path)

The Rust port is `crates/ym-gpu/src/cpu/demosaic/adaptive_ratio.rs`. It follows §3 line by line; these are the decisions the spec left open.

- **Entry points.** `adaptive_ratio::{demosaic, demosaic_whole, demosaic_balanced, demosaic_balanced_whole}` return `Result<RgbImage, Error>` (`BadSize`, `BadInput`, `BadGains`, `BadNoise`, `BadParams`; `BadCurve` cannot occur because `CaCurves` is checked when it is built). `Params { detail, strength, shadow_match, shadow_sharpening }` (defaults as §2.1) and `Derived` (§2.2: `rho_r`, `rho_b`, `g_dn`, `r_dn`, `offset1_dn` = K, `snr_target` = s, `gate_dn` = T, `offset3_dn` = k, `snr18_db`, `shadow_match`, `halo_px`) are public. `Method::AdaptiveRatio` through `demosaic_with` runs `demosaic_balanced` with `Params::default()` (see the next item).
- **Input convention (RA-04, follows `INTEGRATION.md` section 4).** `demosaic_with(m, AdaptiveRatio, inputs)` and `demosaic_whole_with` take the **balanced** mosaic, like every other method, and return the balanced image. They divide each photosite by `inputs.gains` of its colour (the exact inverse of R4, §1.3: `adaptive_ratio::demosaic_balanced` and `demosaic_balanced_whole`, an `f64` division rounded to `f32` once as the oracle's `demosaic_balanced` does) and run the three passes on that. `Chain`, the tile scheduler and R7's `Y_classical` therefore get the division for free and do not do it themselves. `adaptive_ratio::demosaic` and `demosaic_whole` stay the unbalanced entry points of §1.2 (the vectors use them). `demosaic(m, method)` is unit gains and a zero noise model (K = T = 1 DN); with unit gains the division is the identity, so it gives the same bits as the unbalanced entry point. The earlier text of this item (RA-03: the division is not inside `demosaic_with`) is replaced by this one.
- **Invalid input through the infallible entry point.** `demosaic_with` and `demosaic_whole_with` cannot fail (RA-02). When the adaptive ratio demosaic refuses its input they return Menon's result on the same (balanced) mosaic.
- **Regions, not buffer edges.** §3.1 evaluates every pass on the whole padded buffer and sums the per-pass halos (11 px, 13 with shadow matching). The port evaluates each stage only on the rectangle the output core needs. The exact receptive field is 10 px (12 with shadow matching): step 1 of pass 3 reads *native* samples at ±2, which come straight from the mosaic and do not extend the chain. The values in the core are the spec's, because every value the core depends on is inside those rectangles. The padded buffer keeps the §3.1 parity trick (red at (even, even)), with 10 or 12 px of margin (one more where the parity needs it). `Method::halo()` stays at the spec's 13.
- **Banding.** `demosaic` cuts the frame into bands of 128 rows with `halo_px` rows of real context and computes only each band's core rows. The layout depends on the image height alone, so the output is byte-identical for every thread count and equal to `demosaic_whole`.
- **Gains** are `f32` in `DemosaicInputs`; the vectors' `f64` gains (1.9, 1.4) differ from their `f32` roundings by up to 1.7e-8 relative, which moves the four `phase_*` vectors by one `f32` ulp (1.2e-7) and no other vector at all. `Derived::with_f64_gains` takes `f64` gains and reproduces the vectors' `derived` values to 1e-9 relative.
- **CA path.** Built in RA-03b: see section 11. `DemosaicInputs::ca = None` is the plain path described in this section, unchanged bit for bit.
- **Checks.** Vectors: 13 of 17 bit-identical to the oracle, the four phase vectors within 1.2e-7 (tolerance 1e-5). A further 1000 random cases (sizes 17x17 to 599x699, all four phases, gains, noise models, `detail`/`strength`, matching and sharpening on and off, white-noise mosaics) were compared with the C++ reference through a scratch driver: all bit-identical. Measured reach: 9 px (plain, sharpening active), 10 px (shadow matching). Tiled and cropped results equal the whole frame exactly.
- **Speed**, one thread, 4 MP, dev profile with opt-level 3, best of three, CPU seconds per megapixel: plain 0.23 (Menon 0.16, LMMSE 0.16), shadow matching 0.35 to 0.43. On four threads the wall time is 0.11 to 0.16 (plain) and 0.18 to 0.20 (matching). A shared, busy 4-core machine. `cargo run -p ym-gpu --example adaptive_ratio_timing` runs the benchmark (add `CARGO_PROFILE_DEV_OPT_LEVEL=3` for the opt-level 3 builds above); at the workspace's own dev profile (opt-level 1) the review measured, at 2 MP on one thread, plain 0.33, shadow matching 0.60 to 0.63, Menon 0.17 and LMMSE 0.27 CPU s/MP, so plain costs about twice Menon there, not "near Menon" as §5 hopes. No `--release` number exists yet (RA-05).

---

## 10. As built in the chain (RA-04)

- **R6 in `Chain::run_tile`.** For every method the chain calls `demosaic_with(mosaic, method, inputs)` with `inputs = { gains: chain.gains (g0), noise: chain.noise (zero when None), ca: None }`. For `AdaptiveRatio` that is the whole of RA-04: undo R4 on the tile, run the passes with `gains = g0` and the chain's noise, keep the balanced output. Bilinear, Menon and LMMSE ignore the inputs, so the chain gives exactly what it gave before (checked bit for bit in `tests/cpu_adaptive_ratio_chain.rs`); the log-guided demosaic reads the read noise from them. The gains and the noise belong to the frame, not the tile, so every tile of a frame sees the same inputs. The R6 bilateral denoise and R7 are unchanged. `Y_classical` of R7 is a copy of the R6 output, so it is the same method without further code.
- **Halo.** The `Demosaic(AdaptiveRatio)` stage uses `Method::halo()` = 13 in the backward walk of the tile scheduler, so a tile has real context for either configuration (11 without shadow matching, 13 with). Tiles equal the whole frame bit for bit for the R6-only chain and within 1e-4 with every stage on (the other stages' tolerance), for all four CFA phases and both a low and a high noise model.
- **Checks in the chain.** `Chain::validate` refuses a chain whose method is `AdaptiveRatio` and whose gains are not positive and finite, with `CpuError::Mismatch("the adaptive ratio demosaic cannot run: ...")` (`RenderError::Image` in `ym-core`): the infallible `demosaic_with` would otherwise run Menon quietly. Other methods are not checked. The noise model is *not* a reason to refuse (changed when the adaptive ratio demosaic became the default, 2026-10-13): `Chain::noise_dn` reads a model that is missing or fails `NoiseDn::is_valid` as the zero model, so a photo with a broken model still renders (`INTEGRATION.md` section 10). The infallible entry point `demosaic_with` still falls back to Menon's bits for unusable gains or noise (tested in `tests/cpu_adaptive_ratio_chain.rs`).
- **The test-only switch.** `ym_core::pipeline::graph::RenderOptions::demosaic` (default `DEFAULT_METHOD`) is the R6 method of the full-resolution render (the 100 % view and every export); the fit view and thumbnails bin the mosaic and do not demosaic. `FullChain::with_method` takes it (`FullChain::new` is the default method). It is an internal switch: no settings field, no IPC, no UI, no CLI flag. It is a property of the renderer, fixed when the renderer is made, so it is not part of any cache key. Builds with the cargo feature `force-demosaic` of `ym-core` (off by default; `ym-cli` forwards it) read the initial value from the environment variable `YM_FORCE_DEMOSAIC=<method name>` (`adaptive_ratio`, `menon`, ...; an unknown name panics); without the feature no code reads the variable. The feature is enabled only by the dev-dependency of `ym-cli` (so `cargo test -p ym-cli` starts a `ym` that honours the variable) and by `cargo build -p ym-cli --features force-demosaic`; a normal `cargo build -p ym-cli`, the sidecar build and the app never have it. The app's side of a parity test sets `options.render.demosaic` directly.
- **Parity.** The adaptive ratio demosaic is the default, so every default parity test (all formats, a several-tile photo with `--jobs 1` and `--jobs 3`) runs it. `crates/ym-cli/tests/parity.rs` also exports every format and the several-tile photo through the app path with `RenderOptions::demosaic = Menon` and through the binary with `YM_FORCE_DEMOSAIC=menon`: byte-identical, so the method that is no longer the default keeps its parity; and it checks that `YM_FORCE_DEMOSAIC=adaptive_ratio` gives the default's bytes and Menon other bytes. `src-tauri/tests/app_cli_parity.rs` does the same against a separately built `ym` (build it with `--features force-demosaic`; with a plain `ym` the test says so and skips unless `YM_REQUIRE_FORCE_DEMOSAIC` is set).
- **Whole-frame comparison.** `ym-core`'s `a_full_render_with_the_adaptive_ratio_demosaic_is_the_whole_frame_demosaic` builds a `FullChain` with the method forced and every stage but R1, R4 and R6 off, runs it as one tile and as grids of tiles, and compares with R1 and R4 done by hand over the whole frame followed by one whole-frame `demosaic_whole_with` call: equal bit for bit, for a low-noise and a high-noise model.
- **Default renders.** While RA-04 was built `DEFAULT_METHOD` was Menon and the pixels of a default render did not change (the golden set, `golden check --set synthetic`, 61 files, was bit-identical with a `ym` built without the feature, once the output sharpening of RA-08 was set to 0). Since 2026-10-13 `DEFAULT_METHOD` is `AdaptiveRatio` (the owner's decision after RA-H4, `docs/05` section R6), so every default render uses this demosaic and the single-file pixels moved, by design; the golden set is re-baselined with that change.

---

## 11. As built: the CA path (RA-03b)

- **Where.** Pass 1 and pass 2 in `demosaic/adaptive_ratio.rs` (`misregistration_plane`, the relaxation factor `f` in `green_estimate`, the weight `kappa` in `refine_green`); the curves in `LensGeometry::ca_curves` (`cpu/lens.rs`, step `CA_CURVE_STEP_PX` = 8); the engagement in `Chain::ca_curves` (`cpu/chain.rs`); the measurement in `metrics::lateral_ca_residual`. Nothing else changed: no settings field, no IPC, no `ym-core` code (the full chain gets the curves from its own `lens`).
- **The path as the spec says.** Per site: `ca = |shift_R(r) - shift_B(r)|` from the site's own pixel centre, in `f64` with `sqrt` and the curves' linear interpolation, stored once per band in a plane that both passes read. A padded pixel outside the buffer (mirrored data) is at its geometric frame position, as in the oracle's `CaField`, so a tile at the frame edge equals the whole frame. The relaxation factor is unbounded above, as the spec has it (note 3 of section 7); `kappa = min(ca, 1)`.
- **`Error::BadCurve`.** The fields of `CaCurves` are public, so its centre can be NaN or infinite; `adaptive_ratio::{demosaic, demosaic_whole, ...}` refuse it with `Error::BadCurve`, and `demosaic_with` falls back to Menon like for every refused input. (The curves themselves are checked when a `RadialCurve` is built.)
- **The curves from the lens.** `LensGeometry::ca_curves() -> Result<Option<CaCurves>, CaCurveError>` implements section 3.6 literally: knots `(r_src, r_src (K_c(r) - 1))` every 8 px of output radius from 0 to past the corner, `r_src = hd r F(r)`, `hd` from `sqrt` (not `hypot`), the centre at the middle of the frame. Against the curves the vector `ca_weighting` carries (derived by the oracle from the same 16 + 16 CA knots and 16 distortion knots) it agrees to 2.2e-16 relative (the vector prints 17 digits; `serde_json` may read one ulp off). Against the R9 model itself (`LensGeometry::source_positions`): at the source radius of green, red and blue are displaced by `shift_R` and `shift_B` to within 1.5e-3 px, with and without distortion.
  - **Decision (neutral tables).** A table whose knots are all zero moves no pixel in R9, and the CA path would only add the pass-2 smoothing for nothing (note 4 of section 7), so `ca_curves` returns `None` for it (and for no 0x7035 table, and for `decode.chromatic_aberration` off, which `LensGeometry::new` already applied). The golden synthetic files have neutral tables.
  - **The curve ends.** The knots are in source radius. For a pincushion correction (`F < 1`) the last knot is short of the frame corner, which R9 never samples; the curve extends linearly there. For a barrel correction (`F > 1`) it passes the corner.
- **Engagement.** `Chain::run_tile` builds the curves from `Chain::lens` for `Method::AdaptiveRatio` and passes them in `DemosaicInputs::ca`; every other method ignores them. So the CA path runs exactly when the R9 stage has CA: the file has a non-neutral 0x7035 table and `decode.chromatic_aberration` is on. (To switch it off in the renderer, return `None` from `Chain::ca_curves`.) Tiles equal the whole frame bit for bit (the curves belong to the frame); `RenderOptions::demosaic` is `DEFAULT_METHOD`, the adaptive ratio demosaic since 2026-10-13, so the CA path runs in default renders of files that have CA tags.
- **Checks against the oracle.** `ca_weighting` is bit-identical (largest difference 0). A further 1500 random cases through a scratch driver of the C++ reference (1279 with CA: sizes 17 x 17 to 277 x 437, all four phases, curves of 2 to 31 knots with negative radii and shifts up to 3 px, a centre moved off the middle of the frame, the placement anywhere in a frame of up to 6000 x 4500, random gains, noise models, `detail`, `strength`, shadow-matching modes): bit-identical. Changing any one of four constants of section 3.6 (2.5, 102, the `2 kappa + 1` weight, the cap of `2 kappa`) makes 167 to 336 of a 400-case set differ from the oracle. Property tests in `tests/cpu_adaptive_ratio_ca.rs`: banded equals whole, tiles with the halo equal the whole frame, crops at the four phases agree in the interior (all bit for bit, with a placement deep in a big frame), a constant misregistration does not depend on the placement and a growing one does, finite output for every size and phase with misregistrations up to 12 px, the chain runs the path only when the lens has CA.
- **R9 residual, synthetic files that are consistent with their tables** (`examples/ra03b_ca_residual.rs`). A grey (R = G = B) analytic scene with soft edges (2.4 px 10 to 90 %) at every orientation, 1800 x 1200; every mosaic site holds the scene at the position R9 moves it from, so the correct output has no colour. Tables with `K - 1` growing with the square of the radius, red and blue displaced by about `1.2 s` and `-s` px at the corner (`s` = 0.5, 1, 2; `|R - B|` at the corner 1.1, 2.2, 4.4 px), with and without a 1.5 % barrel distortion. Displacement of red and blue against green along the radius after R9, RMS over 48 px patches, all radii from 0.2 of the half diagonal (R / B, px); means of three scenes; R9 not correcting CA is the scale of what there was:

  | tables | R9 without CA (Menon) | adaptive ratio, plain | adaptive ratio, CA path | change | edge chroma, plain to CA path (R-G) | PSNR RGB dB, plain to CA path | PSNR green dB, plain to CA path |
  |---|---|---|---|---|---|---|---|
  | s = 0.5 | 0.129 / 0.108 | 0.064 / 0.052 | 0.057 / 0.046 | -11 % / -12 % | 0.0076 to 0.0067 | 43.28 to 43.71 | 43.70 to 43.92 |
  | s = 1 | 0.248 / 0.208 | 0.138 / 0.111 | 0.120 / 0.097 | -13 % / -13 % | 0.0108 to 0.0097 | 39.79 to 40.25 | 42.74 to 41.99 |
  | s = 2 | 0.514 / 0.405 | 0.261 / 0.228 | 0.215 / 0.186 | -18 % / -18 % | 0.0170 to 0.0149 | 35.22 to 36.22 | 40.21 to 40.17 |
  | s = 1, barrel | 0.242 / 0.202 | 0.131 / 0.108 | 0.115 / 0.096 | -12 % / -11 % | 0.0102 to 0.0092 | 40.19 to 40.54 | 44.10 to 43.03 |
  | s = 2, barrel | 0.496 / 0.401 | 0.243 / 0.209 | 0.203 / 0.174 | -16 % / -17 % | 0.0159 to 0.0140 | 35.81 to 36.72 | 42.03 to 41.43 |

  The CA path is lower in every radial band of every configuration (largest change in the outer bands: 0.303 to 0.258 and 0.503 to 0.396 px for `s` = 1 and 2 at 0.8 to 1.0), also with Gaussian noise of 0.004 and 0.012 (the noise model passed matching). The PSNR of all three planes goes up by 0.4 to 1.0 dB; the PSNR of green changes by +0.2 to -1.1 dB (the 3-tap smoothing of pass 2 where `kappa` is large), and is higher at the noisy levels. Menon leaves a lower displacement than either adaptive ratio variant (0.111 / 0.092 at `s` = 1) at a green PSNR 1 dB below plain adaptive ratio.
  - **Over-correction.** The displacement R9 leaves after a colour-difference demosaic has the opposite sign to the one it started with (`--diagnose`): red is displaced by +0.52 px at the outer band before R9 where the tables say +0.75, so R9 removes 0.75 and leaves -0.23 (adaptive ratio plain: -0.29, CA path: -0.25, Menon: -0.23 at s = 1). A demosaic that builds red and blue as differences from green pulls their edges toward green's by about a quarter to a third of the shift, and R9 corrects the full tag value afterwards. The CA path pulls less; correcting before the demosaic would remove the effect.
- **R9 residual, real files** (`cargo test -p ym-core --lib ra03b_real_files -- --ignored --nocapture`, the nine files of `testdata/real/files`, results in `testdata/ra03b/real/`). Five have a 0x7035 table (A300, A500, NEX-5N, NEX-7 have none). 256 px windows toward the four corners and four edges at 0.3 to 0.97 of the half diagonal, the whole chain (R1 to R9, R7 and R8 as the photo would get them) with the CA path and, by hand, without it, 48 px patches with radial gradient whose red and blue correlate with green after normalising; paired by patch:

  | file (lens) | `LateralChromaticAberration` in the file | patches | R9 without CA, R / B | R9 with the tables, plain | with the tables and the CA path | green gradient energy, CA path over plain |
  |---|---|---|---|---|---|---|
  | ILCE-5000 (E PZ 16-50) | Auto | 198 | 0.164 / 0.198 | 0.472 / 0.555 | 0.473 / 0.536 | 0.952 |
  | ILCE-6300 cRAW (E PZ 18-105 G) | Auto | 49 | 0.110 / 0.060 | 0.704 / 0.305 | 0.701 / 0.304 | 0.933 |
  | ILCE-7M2 (E 25 F2) | Off | 156 | 0.206 / 0.377 | 0.511 / 0.248 | 0.503 / 0.239 | 0.962 |
  | ILCE-7M3 (FE 24-105 G) | Auto | 7 | 0.041 / 0.172 | 0.040 / 0.747 | 0.041 / 0.742 | 0.919 |
  | ILCE-7M4 (FE 24-70 GM II) | Auto | 0 | | | | 0.953 |

  In the RMS over the patches the CA path is lower than or equal to the plain path in R and B on every file with patches, within the 0.001 px the table prints: the red of ILCE-5000 and ILCE-7M3 is 0.001 px higher (0.472 to 0.473, 0.040 to 0.041), everything else is 0.001 to 0.019 px lower. Patch by patch (paired) the red is closer to green in 45 % of 192 patches on ILCE-5000, 78 % of 46 on ILCE-6300, 78 % of 151 on ILCE-7M2 and 2 of 7 on ILCE-7M3, the blue in 76 %, 70 %, 68 % and 5 of 7. So on real photographs it gains next to nothing and costs 4 to 8 % of the green gradient energy (a softer image), because the tables describe a shift the raw data does not have: see the finding below. Flipping the sign of every knot leaves more (0.61 / 0.77 on ILCE-5000), so the direction is not the problem.
  - **Finding for R9 (not part of RA-03b).** On the files whose camera says `LateralChromaticAberration: Auto` and that have usable patches (ILCE-5000, ILCE-6300, ILCE-7M3; the ILCE-7M4 windows have none), R9 with the tables raises the measured displacement of red and blue from 0.04 to 0.2 px (the raw data without any CA correction) to 0.3 to 0.7 px (the red of ILCE-7M3 stays at 0.04), and colour fringes appear in the crops (`testdata/ra03b/real/*.png`: left R9 without CA, middle with the tables, right with the CA path; the ILCE-6300 crop shows red and cyan fringes on every edge that the left one does not have, the ILCE-5000 and ILCE-7M3 crops are too flat to show anything). The tables move red by -1.2 px and blue by -0.4 px at half the half diagonal on the ILCE-6300 file, against a raw image with no such misregistration. The one file with the setting `Off` (ILCE-7M2) is mixed: blue improves (0.38 to 0.24), red gets worse (0.21 to 0.51), with milder fringes. A possible reading, to be checked on more files and against the camera's JPEG: with the camera's CA compensation on, the correction was already applied to the raw data and the 0x7035 table only records it, so R9 would correct twice. The same may hold for the vignetting table (`VignettingCorrection: Auto`) in R3; nothing here tests that. `docs/05` R9 (implementation notes) says what remains for real files; this is that. Until it is settled the CA path inherits the problem, because it follows R9: it is engaged exactly when R9 has CA. If R9 skips CA for files the camera has already corrected, the CA path skips with it, with no change here.
- **Cost** (`examples/ra03b_timing.rs`, 4 MP, one thread, `opt-level = 3` for `ym-gpu`, a busy 4-core machine, best of three, CPU seconds per megapixel): plain path 0.210 (the plain path is not slower than before the change: RA-03 measured 0.23), CA path 0.243, of which the misregistration plane (one `sqrt` and two curve evaluations per R and B site) is 0.033. That is 16 % more than the plain path, 1.7 CPU-s on a 50 MP frame. The plane is `f64` and one per band; a merged curve would be cheaper but not bit-identical to the oracle's two evaluations and a subtraction.
- **GPU.** The CA weight of a site is a pure function of its position (spec 5: "evaluated per pixel from a small curve buffer"): the kernel needs the two curves (about 450 knots each for a 50 MP frame) and the placement.


---

## 12. As built: the GPU kernel (RA-09)

The WGSL port is `crates/ym-gpu/src/shaders/adaptive_ratio.wgsl` with the wrapper `kernels/adaptive_ratio.rs` (`demosaic`, `demosaic_tiled`, `demosaic_planned`, `Balance`, `MAX_BALANCED_DN`) and the parity tests `tests/gpu_adaptive_ratio.rs`. It is not wired into the render path: the raw stages still run the CPU reference (`Method::AdaptiveRatio`), and wiring the kernel in is a later decision. Metal behaviour is a Mac-only check (M0-H4).

- **Contract.** The same as the CPU entry points (sections 1.2 and 1.3), including the plain and the CA path, the shadow sharpening, the shadow colour matching and every parameter of section 2. `Balance::Unbalanced` is the oracle's `demosaic`; `Balance::Balanced` is R6's entry, which divides every photosite by the gain of its colour (`m_u = m_bal / g0[colour]`, the exact inverse of R4) and returns a balanced image like every method. The frame constants (`K`, `T`, `k`, `rho`, the matching switch, the halo) come from the CPU's own `Derived::new` in `f64` and reach the shader as `f32`. The checks of section 4 happen once per frame on the host (a scan, not per tile) and are `GpuError::InvalidInput`: a mosaic below 2 x 2, NaN or infinity, bad gains, a bad noise model, non-finite parameters, a non-finite CA centre. One addition for the `f32` range: a sample whose balanced magnitude `S |m_u| max(g)` exceeds `MAX_BALANCED_DN` = 1e17 DN (about 1.5e12 times white, either sign) is refused too, where the CPU accepts up to about 5e33; the squares of the matching step would overflow `f32` and give NaN. CA knots that do not fit `f32` are refused, and so are gains or a noise model whose per-frame constants (`rho`, `4 g_c`, `g g_c`, `(r g_c)^2`, `K`, `T`, `r`) do not fit `f32` (a shot-noise gain of 1e50 DN per DN, a read noise of 1e40 DN, a gain of 1e30 against a read noise of a few DN: valid for the `f64` CPU port, infinities and an image of NaN here).
- **Layout and passes.** Eleven pipelines, up to 14 dispatches (the matching step runs its two passes once per channel): prepare, green, refine, step 1, step 2, luma and sharpen (when `shadow_sharpening`), then either simple, low and error per channel and the blend (when matching is on) or the exit. All planes (5, or 12 with matching) are `f32` buffers of the padded size, in every storage mode; the input and the output are `f32` buffers too. The padding is made once at the entry, mirrored at the edge of the uploaded rectangle (section 3.1, the red photosites at (even, even) of the plane), and every pass is evaluated on the rectangle of the plane that the core depends on, the CPU port's `Margins`: 10 px of context plain, 12 with matching (section 9). The frame-edge pixels are therefore the CPU's, not those of a per-kernel mirror. The CA weight is computed in the green and refine passes from the site's position and the `f32` knots (a binary search per curve), not stored.
- **`f32` instead of `f64`, and what differs.** Every formula keeps the operation order of section 3 and every plane is stored where the CPU stores `f32`. Two weightings are evaluated in the normalised inverse-cost form, which equals the spec's in exact arithmetic: pass 1 (`alpha_j = 1 / w_j`, the form section 5 recommends) and step 1 of pass 3 (`1 / w_A`, `1 / w_B` instead of `w_B`, `w_A`; the spec's products of two costs reach 1e12 for 1e6 DN data, the inverse form never exceeds the largest sample). The CA field is `f32` (the distance to the optical centre is an exact integer difference plus a half-integer for the usual centre, so only the square root and the interpolation round), a few 1e-7 px from the CPU's.
- **Parity.** The 16 vectors without ties match the oracle to 1.4e-6 at worst (`highlights_and_negatives`, values to 3) and 3e-7 on the others, inside the vectors' 1e-5. Random frames (240 runs: 4 phases, zero, low and high noise, defaults, matching on and off, sharpening off, a lower SNR target; 9 x 7 to 67 x 41; 894 240 samples) differ from the CPU by 7.4e-6 at worst with no sample over 1e-4 or even 2e-5; the balanced entry (1.7e-6), the CA path with random curves and placements (178 224 samples, 2.4e-6), 2 x 2 to 15 x 15 and two-wide strips (2.3e-6), values to 3e11 (2.7e-6 of the peak), flat fields (4.2e-7 absolute, 6e-7 relative; with the high noise model's 17202 DN offset 4.8e-7 absolute) all agree. The CPU port itself moves under 1 ulp of input noise exactly as the GPU differs (next item).
- **Decision flips.** Where the algorithm decides (pass-1 direction type, the pass-2 argmin, the step-1 and step-3 diagonals), two sums that are equal or nearly equal in exact arithmetic can compare the other way after `f32` rounding, and the pixel takes the other of two equally plausible estimates (section 5: "expect isolated decision flips at near-ties"). Noise-free content with exact ties is full of them: the vectors `edge_knight` (a 2:1 edge) and `ca_weighting` (a 45 degree edge) differ at 10 and 58 of 1 728 and 3 072 samples, by 4.2e-4 and 6.1e-4. The evidence that these are flips and not a defect: the CPU port under a perturbation of 1 ulp of the input (relative 1.2e-7) moves the same vectors at 8 and 26 samples by up to 6.6e-4 and 1.1e-3, the zone plate at 13 to 72 samples by up to 3.3e-2 (the GPU happens to match it to 2.4e-7), and a 2048 x 1365 frame at 344 samples (low noise) and 118 (high noise) by up to 1.8e-2, against 70 and 18 samples (8.3e-6 and 2.1e-6 of 8.4 million, worst 1.2e-2 and 4.0e-3) for the GPU through tiles of 1024. The test bounds are the measured counts with margin (`TIE_VECTORS`, `FRAME_FRACTION`, `LARGE_FRACTION`).
- **Real mosaics (review of RA-09).** The numbers above are for synthetic content. On crops of real raw data (the review's scratch drivers, not committed: the real files are local-only and the C++ references live outside the repository) the GPU differs from the CPU port at more samples, because quantised ADC data is full of exact ties that the `f32` and `f64` sums break differently. 240 runs on five crops of 900 x 700 to 1100 x 800 (ILCE-7M3 at three places, one at an odd origin so the pattern is BGGR; A1; DSLR-A300), each with the noise models zero, low, mid and high, the default parameters, matching forced on and the sharpening and matching off, the plain and the CA path, the unbalanced and the balanced entry (532.5 million samples): 88 090 samples (1.65e-4) differ from the CPU by more than 1e-4, worst 1.55e-2 of white; per crop 4e-6 to 1.2e-5 (A1), 1.0e-4 to 4.5e-4 (ILCE-7M3), 1.5e-4 to 3.4e-4 (A300). All 240 tiled runs (tiles of 400 px) equal the whole GPU frame bit for bit. The synthetic fractions of `tests/gpu_adaptive_ratio.rs` (`FRAME_FRACTION`, `LARGE_FRACTION`) are therefore bounds for synthetic content only; a check on real photographs, such as the wiring decision, should count flips against an `f32` emulation (next item) rather than hold the CPU bound.
- **An independent `f32` emulation reproduces the flips.** The C++ oracle compiled with every `double` turned into `float` (and `-fsingle-precision-constant`, `-ffp-contract=off`), an `f32` version of the algorithm that shares no code with the WGSL, gives the same result as the kernel: on the vectors it differs from the `f64` oracle at the same 58 samples of `ca_weighting` (6.05e-4) and the same 10 of `edge_knight` (4.20e-4) as the GPU, and the GPU differs from it by 9.5e-7 at worst with no sample over 1e-4. On 24 real-mosaic jobs (three crops, low and high noise, plain and CA, unbalanced and balanced; 56.8 million samples) the `f32` emulation differs from the `f64` oracle at 8 856 samples and the GPU at 8 936, and the GPU differs from the emulation at 420 (7e-6), which the inverse-cost weights of pass 1 and step 1 and the `f64` frame constants account for (a flip upstream of a decision). The CPU port under 1 ulp of input noise moves 8 500 to 175 500 samples of each real job (0.3 % to 8 %), so that comparison, used in the synthetic tests, is a loose upper bound on real content, not a measure.
- **Determinism and tiles.** No pass reads a value another thread of the same dispatch writes, there are no reductions and every sum has a fixed order: two runs, runs on recycled pool buffers (checked for every parameter set, which would show a pass reading a value no pass wrote) and runs on every storage mode give the same bits. Tiles with the declared context (11 px, 13 with matching; `demosaic_planned` refuses a plan with less) equal the whole frame bit for bit, for all four phases, odd sizes, the CA path with a placement inside a big frame, the balanced entry and tile sizes down to 2; a crop by one photosite gives the identical interior. The reach found by perturbing one photosite is the CPU port's (8 px plain with that perturbation, 10 px with matching) and never exceeds the halo.
- **Limits and cost.** The planes of one run live in one storage binding, 20 bytes per padded pixel (48 with matching): 128 MiB holds 6.7 million (2.8 million) padded pixels, a larger frame gets `GpuError::BufferTooLarge` and goes through `demosaic_tiled`; a tile of 1024 px needs 53 MB of planes with matching. A padded plane of more than 16 x `max_compute_workgroups_per_dimension` positions on a side is refused with `InvalidInput` before any allocation. On lavapipe (a CPU rasteriser sharing four busy cores, so not a GPU figure) 2048 x 1365 through four tiles takes 1.0 s (1.6 s with matching) against 1.0 s (2.4 s) for the CPU port.
