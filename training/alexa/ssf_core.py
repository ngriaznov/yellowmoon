"""Spectral model used to estimate ALEXA (ALEV III) spectral sensitivities.

Inputs (all permissively licensed):
  * ARRI's ALEXA SUP 3.x camera-native -> ACES AP0 matrices at 17 CCTs
    (aces-dev v1.3.1, AMPAS license, notice retained in the deliverable).
  * 52 measured camera curves from rawtoaces-data (Apache-2.0), used as a
    statistical prior on what camera spectral sensitivities look like.
  * The rawtoaces 190-patch training reflectances (Apache-2.0).
  * CIE 1931 2-degree CMFs and CIE illuminant formulas (via colour-science).

Conventions follow Academy P-2013-001 as implemented by colour-science
(`colour.matrix_idt`): white-balanced camera RGB, CAT02 adaptation of the
scene colorimetry to the ACES white, CIE Lab error.

Copied from research/alexa-ssf/scripts/ssf_core.py. The only change is where the
fetched inputs live (testdata/sources/, see tools/fetch/spectral.sh); the numerics are
untouched so that the published leave-one-out numbers reproduce. Run scripts that
import this module from the repository root as ``python3 -m training.<pkg>.<module>``.
"""

from __future__ import annotations

import glob
import json
import os

import colour
import numpy as np
from colour.characterisation.aces_it import read_training_data_rawtoaces_v1
from scipy.optimize import least_squares

from training.common import paths

# Fetched by tools/fetch/spectral.sh into testdata/sources/ (pinned commits).
ROOT = str(paths.REPO_ROOT)
CAM_DIR = str(paths.CAMERA_CURVES_DIR)
ARRI_DIR = str(paths.ARRI_MATRIX_DIR)

SHAPE = colour.SpectralShape(380, 780, 5)
WL = np.arange(380, 781, 5).astype(float)  # 81 samples

_CMFS = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"].copy().align(SHAPE)
CMF = _CMFS.values  # (81, 3)
ACES = colour.RGB_COLOURSPACES["ACES2065-1"]
AP0_TO_XYZ = ACES.matrix_RGB_to_XYZ
XYZ_TO_AP0 = ACES.matrix_XYZ_to_RGB
ACES_WHITE_XY = ACES.whitepoint
ACES_WHITE_XYZ = colour.xy_to_XYZ(ACES_WHITE_XY)


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def load_cameras() -> dict[str, np.ndarray]:
    """Return {name: (81, 3) curves, each channel normalised to peak 1}."""
    cams = {}
    for path in sorted(glob.glob(os.path.join(CAM_DIR, "*.json"))):
        with open(path) as fh:
            doc = json.load(fh)
        data = {float(k): v for k, v in doc["spectral_data"]["data"]["main"].items()}
        arr = np.array([data[w] for w in WL], dtype=float)
        arr = np.clip(arr, 0, None)
        arr /= arr.max(axis=0, keepdims=True)
        hdr = doc["header"]
        name = f'{hdr["manufacturer"]} {hdr["model"]}'
        cams[name] = arr
    return cams


def load_arri_matrices(nd: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return (ccts (17,), matrices (17, 3, 3)) from ARRI's 2013 table."""
    fname = "Alexa-st-nd-aces_matrix.txt" if nd else "Alexa_aces_matrix.txt"
    tab = np.loadtxt(os.path.join(ARRI_DIR, fname), skiprows=3)
    return tab[:, 0], tab[:, 1:].reshape(-1, 3, 3)


def interpolate_matrix(ccts: np.ndarray, mats: np.ndarray, cct: float) -> np.ndarray:
    """Linear interpolation in mired between bracketing table entries.

    (ARRI's v3_IDT_maker.py interpolates between entries i and i+1 where
    i = searchsorted(cct), which extrapolates for non-table CCTs; this uses
    the bracketing pair instead and clamps outside the table.)
    """
    if cct <= ccts[0]:
        return mats[0]
    if cct >= ccts[-1]:
        return mats[-1]
    i = np.searchsorted(ccts, cct)
    lo, hi = i - 1, i
    a = (1 / cct - 1 / ccts[lo]) / (1 / ccts[hi] - 1 / ccts[lo])
    return (1 - a) * mats[lo] + a * mats[hi]


def training_reflectances() -> np.ndarray:
    """rawtoaces 190-patch training set, (81, 190)."""
    return read_training_data_rawtoaces_v1().copy().align(SHAPE).values


def test_reflectances() -> dict[str, np.ndarray]:
    """Held-out reflectance sets, each (81, N)."""
    sets = {}
    ces = colour.quality.cfi2017.load_TCS_CIE2017(SHAPE)
    sets["CIE 2017 CES (99)"] = ces.values
    cc = colour.SDS_COLOURCHECKERS["ColorChecker N Ohta"]
    sets["ColorChecker (Ohta, 24)"] = np.stack(
        [cc[k].copy().align(SHAPE).values for k in cc], axis=1
    )
    vs = colour.quality.datasets.SDS_VS["NIST CQS 9.0"] if "NIST CQS 9.0" in colour.quality.datasets.SDS_VS else None
    if vs is not None:
        sets["NIST CQS VS (15)"] = np.stack(
            [vs[k].copy().align(SHAPE).values for k in vs], axis=1
        )
    return sets


# --------------------------------------------------------------------------
# Illuminants
# --------------------------------------------------------------------------
def illuminant(cct: float, convention: str = "rta") -> np.ndarray:
    """Illuminant SPD at a CCT.

    'rta'    : blackbody below 4000 K, CIE D series at and above (rawtoaces v1).
    'planck' : blackbody at every CCT.
    """
    if convention == "rta" and cct >= 4000:
        xy = colour.temperature.CCT_to_xy_CIE_D(cct * 1.4388 / 1.4380)
        sd = colour.sd_CIE_illuminant_D_series(xy)
    else:
        sd = colour.sd_blackbody(float(cct), SHAPE)
    v = sd.copy().align(SHAPE).values
    return v / v.max()


def named_illuminant(name: str) -> np.ndarray:
    if name in colour.SDS_ILLUMINANTS:
        sd = colour.SDS_ILLUMINANTS[name]
    else:
        sd = colour.SDS_LIGHT_SOURCES[name]
    v = sd.copy().align(SHAPE, extrapolator_kwargs={"method": "Constant", "left": None, "right": None}).values
    v = np.clip(v, 0, None)
    return v / v.max()


def illuminant_cct(E: np.ndarray) -> float:
    XYZ = CMF.T @ E
    xy = colour.XYZ_to_xy(XYZ)
    uv = colour.UCS_to_uv(colour.XYZ_to_UCS(colour.xy_to_XYZ(xy)))
    return float(colour.temperature.uv_to_CCT_Ohno2013(uv)[0])


# --------------------------------------------------------------------------
# Colorimetry
# --------------------------------------------------------------------------
def _f(t):
    d = 6 / 29
    return np.where(t > d**3, np.cbrt(t), t / (3 * d**2) + 4 / 29)


def XYZ_to_Lab(XYZ: np.ndarray) -> np.ndarray:
    r = XYZ / ACES_WHITE_XYZ
    fx, fy, fz = _f(r[..., 0]), _f(r[..., 1]), _f(r[..., 2])
    return np.stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], axis=-1)


def target_XYZ(E: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Scene XYZ of reflectances under E, CAT02-adapted to ACES white, (N,3)."""
    XYZ = (E[:, None] * R).T @ CMF
    norm = CMF[:, 1] @ E
    XYZ /= norm
    XYZ_w = (CMF.T @ E) / norm
    M = colour.adaptation.matrix_chromatic_adaptation_VonKries(
        XYZ_w, ACES_WHITE_XYZ, "CAT02"
    )
    return XYZ @ M.T


def camera_rgb_wb(S: np.ndarray, E: np.ndarray, R: np.ndarray) -> np.ndarray:
    """White-balanced camera RGB of reflectances, white -> (1, 1, 1), (N,3)."""
    raw = (E[:, None] * R).T @ S
    white = E @ S
    return raw / white


def ap0_to_Lab(AP0: np.ndarray) -> np.ndarray:
    return XYZ_to_Lab(AP0 @ AP0_TO_XYZ.T)


def delta_E00(Lab1: np.ndarray, Lab2: np.ndarray) -> np.ndarray:
    return colour.difference.delta_E_CIE2000(Lab1, Lab2)


# --------------------------------------------------------------------------
# Matrix fitting (forward problem: camera curves -> IDT matrices)
# --------------------------------------------------------------------------
def _wp_matrix(p: np.ndarray) -> np.ndarray:
    M = np.zeros((3, 3))
    M[:, :2] = p.reshape(3, 2)
    M[:, 2] = 1 - M[:, 0] - M[:, 1]
    return M


def fit_matrix_lab(rgb: np.ndarray, XYZ_t: np.ndarray, M0: np.ndarray | None = None) -> np.ndarray:
    """P-2013-001-style white-preserving matrix minimising CIE Lab error."""
    Lab_t = XYZ_to_Lab(XYZ_t)
    p0 = (M0[:, :2] if M0 is not None else np.array([[1, 0], [0, 1], [0, 0]])).ravel()

    def res(p):
        return (ap0_to_Lab(rgb @ _wp_matrix(p).T) - Lab_t).ravel()

    return _wp_matrix(least_squares(res, p0, method="lm").x)


def _fprime(t):
    d = 6 / 29
    return np.where(t > d**3, (1 / 3) * np.cbrt(np.maximum(t, 1e-12)) ** -2, 1 / (3 * d**2))


def fit_matrix_lab_gn(rgb, XYZ_t, M0, iters=2):
    """Gauss-Newton refinement of the Lab-optimal white-preserving matrix.

    Analytic Jacobian; starting from M0. One step from the published matrix
    gives (to first order) how far the camera's own optimum is from it.
    """
    Lab_t = XYZ_to_Lab(XYZ_t)
    p = M0[:, :2].ravel().copy()
    n = rgb.shape[0]
    # dXYZ/dp, shape (n, 3, 6), constant in p
    dX = np.zeros((n, 3, 6))
    for r in range(3):
        for j in range(2):
            dX[:, :, 2 * r + j] = AP0_TO_XYZ[:, r][None, :] * (rgb[:, j] - rgb[:, 2])[:, None]
    for _ in range(iters):
        M = _wp_matrix(p)
        XYZ = rgb @ M.T @ AP0_TO_XYZ.T
        t = XYZ / ACES_WHITE_XYZ
        fp = _fprime(t) / ACES_WHITE_XYZ  # (n, 3)
        J_lab_xyz = np.zeros((n, 3, 3))
        J_lab_xyz[:, 0, 1] = 116 * fp[:, 1]
        J_lab_xyz[:, 1, 0] = 500 * fp[:, 0]
        J_lab_xyz[:, 1, 1] = -500 * fp[:, 1]
        J_lab_xyz[:, 2, 1] = 200 * fp[:, 1]
        J_lab_xyz[:, 2, 2] = -200 * fp[:, 2]
        J = np.einsum("nij,njk->nik", J_lab_xyz, dX).reshape(-1, 6)
        res = (XYZ_to_Lab(XYZ) - Lab_t).ravel()
        dp, *_ = np.linalg.lstsq(J, -res, rcond=None)
        p = p + dp
    return _wp_matrix(p)


def fit_matrix_linear(rgb: np.ndarray, XYZ_t: np.ndarray) -> np.ndarray:
    """Fast white-preserving weighted linear LS (weights approximate Lab)."""
    AP0_t = XYZ_t @ XYZ_TO_AP0.T
    Y = np.clip(XYZ_t[:, 1], 1e-3, None)
    w = Y ** (-2 / 3)
    A = np.stack([rgb[:, 0] - rgb[:, 2], rgb[:, 1] - rgb[:, 2]], axis=1) * w[:, None]
    M = np.zeros((3, 3))
    for r in range(3):
        b = (AP0_t[:, r] - rgb[:, 2]) * w
        m, *_ = np.linalg.lstsq(A, b, rcond=None)
        M[r] = [m[0], m[1], 1 - m[0] - m[1]]
    return M


def camera_matrices(S, ccts, R, convention="rta", lab=True):
    """IDT matrices a camera with curves S would get at each CCT."""
    out, M_prev = [], None
    for T in ccts:
        E = illuminant(T, convention)
        rgb = camera_rgb_wb(S, E, R)
        XYZ_t = target_XYZ(E, R)
        M = fit_matrix_lab(rgb, XYZ_t, M_prev) if lab else fit_matrix_linear(rgb, XYZ_t)
        out.append(M)
        M_prev = M
    return np.stack(out)


# --------------------------------------------------------------------------
# Prior
# --------------------------------------------------------------------------
class Prior:
    """Per-channel PCA of peak-normalised camera curves."""

    def __init__(self, curves: list[np.ndarray], k: int):
        X = np.stack(curves)  # (n, 81, 3)
        self.k = k
        self.mean = X.mean(axis=0)  # (81, 3)
        self.P = np.zeros((3, k, X.shape[1]))
        self.sigma = np.zeros((3, k))
        for c in range(3):
            D = X[:, :, c] - self.mean[:, c]
            U, s, Vt = np.linalg.svd(D, full_matrices=False)
            self.P[c] = Vt[:k]
            self.sigma[c] = s[:k] / np.sqrt(max(len(X) - 1, 1))

    def curves(self, a: np.ndarray) -> np.ndarray:
        a = a.reshape(3, self.k)
        S = self.mean.copy()
        for c in range(3):
            S[:, c] += a[c] @ self.P[c]
        return S


# --------------------------------------------------------------------------
# Inverse problem: published matrices -> camera curves
# --------------------------------------------------------------------------
class Estimator:
    """Estimate curves S from published CCT-dependent IDT matrices.

    objective 'forward'     : the published matrices, applied to S's
                              white-balanced RGB, should reproduce the target
                              colorimetry (CIE Lab) at every CCT.
    objective 'consistency' : the matrices S itself would get (fast weighted
                              linear fit) should equal the published ones.
    """

    def __init__(self, prior: Prior, ccts, mats, R, convention="rta",
                 objective="forward", lam=1.0, neg_weight=50.0):
        self.prior, self.ccts, self.mats, self.R = prior, ccts, mats, R
        self.objective, self.lam, self.neg_weight = objective, lam, neg_weight
        self.E = [illuminant(T, convention) for T in ccts]
        self.XYZ_t = [target_XYZ(E, R) for E in self.E]
        self.Lab_t = [XYZ_to_Lab(x) for x in self.XYZ_t]
        n_res = len(ccts) * R.shape[1]
        self.scale = 1 / np.sqrt(n_res)

    def residuals(self, a):
        S = self.prior.curves(a)
        parts = []
        if self.objective == "forward":
            for E, M, Lab_t in zip(self.E, self.mats, self.Lab_t):
                rgb = camera_rgb_wb(S, E, self.R)
                parts.append(((ap0_to_Lab(rgb @ M.T) - Lab_t) * self.scale).ravel())
        elif self.objective == "consistency":
            # Unbiased: minimised exactly at the true curves if the published
            # matrices were made with this recipe (P-2013-001-style Lab fit).
            for E, M, XYZ_t in zip(self.E, self.mats, self.XYZ_t):
                rgb = camera_rgb_wb(S, E, self.R)
                parts.append(((fit_matrix_lab_gn(rgb, XYZ_t, M, iters=1) - M) * 10).ravel())
        else:  # 'consistency_linear': fast approximation, recipe-mismatched
            for E, M, XYZ_t in zip(self.E, self.mats, self.XYZ_t):
                rgb = camera_rgb_wb(S, E, self.R)
                parts.append(((fit_matrix_linear(rgb, XYZ_t) - M) * 10).ravel())
        parts.append(np.sqrt(self.lam) * (a / self.prior.sigma.ravel()) * 0.1)
        parts.append(self.neg_weight * np.minimum(S, 0).ravel())
        return np.concatenate(parts)

    def fit(self, n_starts: int = 1, seed: int = 0):
        rng = np.random.default_rng(seed)
        best = None
        for i in range(n_starts):
            a0 = np.zeros(3 * self.prior.k) if i == 0 else rng.normal(
                0, 1, 3 * self.prior.k) * self.prior.sigma.ravel()
            r = least_squares(self.residuals, a0, method="trf",
                              x_scale=self.prior.sigma.ravel(), diff_step=1e-3)
            if best is None or r.cost < best.cost:
                best = r
        S = np.clip(self.prior.curves(best.x), 0, None)
        return S / S.max(axis=0, keepdims=True), best


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def rfse(S_est: np.ndarray, S_true: np.ndarray, band=(380, 780)) -> float:
    """Relative full-scale error (%), per-channel scale aligned, mean of RGB."""
    m = (WL >= band[0]) & (WL <= band[1])
    errs = []
    for c in range(3):
        t, e = S_true[m, c], S_est[m, c]
        k = (e @ t) / (e @ e)
        errs.append(np.sqrt(np.mean((k * e - t) ** 2)) / t.max())
    return 100 * float(np.mean(errs))


def emulation_dE(S_true, S_est, ccts, mats, R, illums):
    """ΔE00 between images rendered from true and estimated curves.

    Both go through the same (true camera's) matrix at the illuminant's CCT,
    so this isolates the error a wrong curve would cause in emulation.
    Returns {illuminant name: array of ΔE00 per reflectance}.
    """
    out = {}
    for name, E in illums.items():
        M = interpolate_matrix(ccts, mats, illuminant_cct(E))
        Lab_t = ap0_to_Lab(camera_rgb_wb(S_true, E, R) @ M.T)
        Lab_e = ap0_to_Lab(camera_rgb_wb(S_est, E, R) @ M.T)
        out[name] = delta_E00(Lab_t, Lab_e)
    return out


def colorimetric_dE(S, ccts, mats, R, illums):
    """ΔE00 between a camera rendered through the matrices and true scene colour."""
    out = {}
    for name, E in illums.items():
        M = interpolate_matrix(ccts, mats, illuminant_cct(E))
        Lab_c = ap0_to_Lab(camera_rgb_wb(S, E, R) @ M.T)
        Lab_t = XYZ_to_Lab(target_XYZ(E, R))
        out[name] = delta_E00(Lab_c, Lab_t)
    return out


TEST_ILLUMINANTS = {
    "Tungsten (A)": "A",
    "Studio tungsten (ISO 7589)": "ISO 7589 Studio Tungsten",
    "D55": "D55",
    "D65": "D65",
    "Fluorescent FL2": "FL2",
    "Tri-band fluorescent FL11": "FL11",
    "White LED (LED-B3)": "LED-B3",
    "Hybrid LED (LED-BH1)": "LED-BH1",
    "RGB LED (LED-RGB1)": "LED-RGB1",
}


def test_illuminants() -> dict[str, np.ndarray]:
    return {k: named_illuminant(v) for k, v in TEST_ILLUMINANTS.items()}
