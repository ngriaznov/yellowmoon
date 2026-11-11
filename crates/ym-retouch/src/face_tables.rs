//! The two tables of the face mesh, drawn by us (`docs/27` S-09, section 4): textures over the mesh's `(u, v)` space.
//!
//! The mesh has 468 vertices, and a 4000 px face has about 234 px between them, so a value per vertex would be a coarse
//! mosaic. Instead each table is a small smooth texture, evaluated procedurally from the texture coordinates of landmarks
//! (`UV` of the canonical model: the eye corners, the lid lines, the nostrils, the mouth corners), and the rasteriser
//! ([`crate::face_raster::render_mesh`]) reads it at each pixel. Nothing here is measured or learned; the shapes are ours.
//!
//! * [`MeshTable::UnderEye`]: bright in a crescent below each eye: from the lower lid line down to a depth of about half the
//!   eye's width at its deepest, widest toward the nose, fading at both corners (`docs/27` 5.4 multiplies it by the skin
//!   and takes the eyes out).
//! * [`MeshTable::InvertedSkinUniformity`]: bright where the skin is not uniform and even skin must leave it alone (the
//!   lids and the sockets, the sides of the nose and the nostrils, the folds from nose to mouth, the corners of the mouth,
//!   the crease under the lip), zero on the cheeks, the forehead, the temples and the jaw.
//!
//! The textures are built once (256 x 256 texels) and shared.

use std::sync::OnceLock;

use crate::face_raster::Texture;
use crate::face_topology::UV;
use crate::models::MeshTable;

/// The side of the textures, in texels.
pub const TEXTURE_PX: usize = 256;

fn uv(i: usize) -> [f64; 2] {
    [f64::from(UV[i][0]), f64::from(UV[i][1])]
}

fn mid(ids: &[usize]) -> [f64; 2] {
    let mut s = [0.0, 0.0];
    for &i in ids {
        let p = uv(i);
        s[0] += p[0];
        s[1] += p[1];
    }
    [s[0] / ids.len() as f64, s[1] / ids.len() as f64]
}

fn smoothstep(e0: f64, e1: f64, x: f64) -> f64 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// One eye: the corners and the lower lid line, outer corner first, as landmark indices.
struct Eye {
    outer: usize,
    inner: usize,
    lower_lid: [usize; 9],
}

const EYES: [Eye; 2] = [
    Eye {
        outer: 33,
        inner: 133,
        lower_lid: [33, 7, 163, 144, 145, 153, 154, 155, 133],
    },
    Eye {
        outer: 263,
        inner: 362,
        lower_lid: [263, 249, 390, 373, 374, 380, 381, 382, 362],
    },
];

impl Eye {
    /// The crescent below this eye at `(u, v)`, `0..1`.
    fn under_eye(&self, p: [f64; 2]) -> f64 {
        let (o, i) = (uv(self.outer), uv(self.inner));
        let axis = [i[0] - o[0], i[1] - o[1]];
        let width = axis[0].hypot(axis[1]);
        let t = ((p[0] - o[0]) * axis[0] + (p[1] - o[1]) * axis[1]) / (width * width);
        // The lower lid's height at this position along the eye.
        let lid_v = self.lid_v_at(t);
        let below = p[1] - lid_v;
        // Along the eye: from just outside the outer corner to a little past the inner one (the tear trough).
        let along = smoothstep(-0.10, 0.18, t) * (1.0 - smoothstep(0.92, 1.22, t));
        // How far down: about half the eye's width at its deepest, toward the nose.
        let depth = width * (0.12 + 0.48 * (-((t - 0.62) / 0.40).powi(2)).exp());
        let down =
            smoothstep(0.0, 0.05 * width, below) * (1.0 - smoothstep(0.35 * depth, depth, below));
        along * down
    }

    /// The `v` of the lower lid line at the position `t` along the eye (`0` outer corner, `1` inner), held to its ends.
    fn lid_v_at(&self, t: f64) -> f64 {
        let (o, i) = (uv(self.outer), uv(self.inner));
        let axis = [i[0] - o[0], i[1] - o[1]];
        let w2 = axis[0] * axis[0] + axis[1] * axis[1];
        let tt = |idx: usize| {
            let q = uv(idx);
            ((q[0] - o[0]) * axis[0] + (q[1] - o[1]) * axis[1]) / w2
        };
        let pts: Vec<(f64, f64)> = self.lower_lid.iter().map(|&k| (tt(k), uv(k)[1])).collect();
        if t <= pts[0].0 {
            return pts[0].1;
        }
        for pair in pts.windows(2) {
            let ((t0, v0), (t1, v1)) = (pair[0], pair[1]);
            if t <= t1 && t1 > t0 {
                return v0 + (v1 - v0) * (t - t0) / (t1 - t0);
            }
        }
        pts[pts.len() - 1].1
    }

    /// The eyelid and socket region (the eye itself is cut by the parser), `0..1`.
    fn socket(&self, p: [f64; 2]) -> f64 {
        let (o, i) = (uv(self.outer), uv(self.inner));
        let width = (i[0] - o[0]).hypot(i[1] - o[1]);
        let centre = [(o[0] + i[0]) / 2.0, (o[1] + i[1]) / 2.0];
        // An ellipse about the eye, taller above it (the lid and its crease).
        let dx = (p[0] - centre[0]) / (0.68 * width);
        let dy = if p[1] < centre[1] {
            (p[1] - centre[1]) / (0.55 * width)
        } else {
            (p[1] - centre[1]) / (0.30 * width)
        };
        1.0 - smoothstep(0.6, 1.0, dx.hypot(dy))
    }
}

/// A soft blob: `strength` at `centre`, falling to 0 at the ellipse of semi-axes `radii`.
fn blob(p: [f64; 2], centre: [f64; 2], radii: [f64; 2], strength: f64) -> f64 {
    let d = ((p[0] - centre[0]) / radii[0]).hypot((p[1] - centre[1]) / radii[1]);
    strength * (1.0 - smoothstep(0.55, 1.0, d))
}

/// A soft capsule: `strength` along the segment `a`-`b`, falling to 0 at `radius` from it.
fn capsule(p: [f64; 2], a: [f64; 2], b: [f64; 2], radius: f64, strength: f64) -> f64 {
    let ab = [b[0] - a[0], b[1] - a[1]];
    let len2 = ab[0] * ab[0] + ab[1] * ab[1];
    let t = (((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / len2).clamp(0.0, 1.0);
    let q = [a[0] + t * ab[0], a[1] + t * ab[1]];
    let d = (p[0] - q[0]).hypot(p[1] - q[1]) / radius;
    strength * (1.0 - smoothstep(0.5, 1.0, d))
}

/// The under-eye table at `(u, v)`.
fn under_eye(u: f64, v: f64) -> f64 {
    EYES.iter().map(|e| e.under_eye([u, v])).fold(0.0, f64::max)
}

/// The non-uniform-skin table at `(u, v)`.
fn non_uniformity(u: f64, v: f64) -> f64 {
    let p = [u, v];
    let mut best = 0.0f64;
    for eye in &EYES {
        best = best.max(eye.socket(p));
        // The inner corner and the side of the nose bridge beside it.
        best = best.max(blob(p, uv(eye.inner), [0.045, 0.06], 0.9));
    }
    // The nose: the nostrils and the wings, the base, the tip.
    best = best.max(blob(p, mid(&[98, 97, 2, 326, 327]), [0.13, 0.05], 1.0));
    best = best.max(blob(p, uv(129), [0.04, 0.045], 1.0));
    best = best.max(blob(p, uv(358), [0.04, 0.045], 1.0));
    best = best.max(blob(p, uv(4), [0.05, 0.045], 0.7));
    // The folds from the wings of the nose to the corners of the mouth.
    best = best.max(capsule(p, uv(129), uv(61), 0.03, 0.85));
    best = best.max(capsule(p, uv(358), uv(291), 0.03, 0.85));
    // The corners of the mouth, the philtrum, the crease under the lip.
    best = best.max(blob(p, uv(61), [0.06, 0.055], 1.0));
    best = best.max(blob(p, uv(291), [0.06, 0.055], 1.0));
    best = best.max(blob(p, mid(&[2, 0]), [0.05, 0.04], 0.75));
    best = best.max(blob(p, mid(&[17, 200]), [0.09, 0.045], 0.9));
    best
}

/// The texture of a table (built once).
pub fn texture(table: MeshTable) -> &'static Texture {
    static UNDER_EYE: OnceLock<Texture> = OnceLock::new();
    static NON_UNIFORM: OnceLock<Texture> = OnceLock::new();
    match table {
        MeshTable::UnderEye => UNDER_EYE.get_or_init(|| Texture::from_fn(TEXTURE_PX, under_eye)),
        MeshTable::InvertedSkinUniformity => {
            NON_UNIFORM.get_or_init(|| Texture::from_fn(TEXTURE_PX, non_uniformity))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn value(table: MeshTable, p: [f64; 2]) -> f32 {
        texture(table).sample(p[0], p[1])
    }

    /// The point `fraction` of the way from landmark `a` to landmark `b`, in texture space.
    fn between(a: usize, b: usize, fraction: f64) -> [f64; 2] {
        let (pa, pb) = (uv(a), uv(b));
        [
            pa[0] + (pb[0] - pa[0]) * fraction,
            pa[1] + (pb[1] - pa[1]) * fraction,
        ]
    }

    #[test]
    fn the_crescent_sits_below_each_eye_and_nowhere_else() {
        let t = MeshTable::UnderEye;
        for (lower, cheek, upper, brow) in [(145, 50, 159, 105), (374, 280, 386, 334)] {
            // Just under the middle of the lower lid: bright. Level with the lid line itself: dark. Above the eye: dark.
            let mut under = uv(lower);
            under[1] += 0.03;
            assert!(
                value(t, under) > 0.8,
                "under the {lower}: {}",
                value(t, under)
            );
            assert!(value(t, uv(lower)) < 0.5, "on the lid line");
            assert!(value(t, uv(upper)) < 0.05, "on the upper lid");
            assert!(value(t, uv(brow)) < 0.02, "on the brow");
            // A cheek and the forehead are untouched; so is the nose bridge between the eyes.
            assert!(
                value(t, uv(cheek)) < 0.02,
                "cheek {cheek}: {}",
                value(t, uv(cheek))
            );
        }
        assert!(value(t, uv(10)) < 0.01 && value(t, uv(168)) < 0.02 && value(t, uv(152)) < 0.01);
        // Fades at the outer corner side: well beyond the corner (toward the temple) it is 0.
        let mut far = uv(33);
        far[0] -= 0.08;
        far[1] += 0.03;
        assert!(value(t, far) < 0.05);
        // Symmetric: the two eyes agree about a mirrored point.
        let (a, b) = (
            value(t, between(145, 145, 0.0)),
            value(t, between(374, 374, 0.0)),
        );
        assert!((a - b).abs() < 0.15);
    }

    #[test]
    fn the_uniformity_table_is_bright_on_the_features_and_dark_on_the_cheeks() {
        let t = MeshTable::InvertedSkinUniformity;
        for (i, name) in [
            (98, "nostril"),
            (327, "nostril"),
            (129, "wing"),
            (358, "wing"),
            (61, "mouth corner"),
            (291, "mouth corner"),
        ] {
            assert!(value(t, uv(i)) > 0.9, "{name} {i}: {}", value(t, uv(i)));
        }
        // The upper eyelid and the lid crease.
        assert!(value(t, uv(159)) > 0.8 && value(t, uv(386)) > 0.8);
        for (i, name) in [
            (205, "cheek"),
            (425, "cheek"),
            (10, "forehead"),
            (9, "forehead"),
            (152, "chin tip"),
            (234, "jaw"),
        ] {
            assert!(value(t, uv(i)) < 0.1, "{name} {i}: {}", value(t, uv(i)));
        }
    }

    #[test]
    fn the_textures_are_smooth_deterministic_and_shared() {
        let a = texture(MeshTable::UnderEye);
        assert!(std::ptr::eq(a, texture(MeshTable::UnderEye)));
        assert_eq!(a.size(), TEXTURE_PX);
        // Smooth: neighbouring texels differ little (the steepest ramp, over 0.05 of an eye width, is a few texels).
        let mut worst = 0.0f32;
        for j in 0..TEXTURE_PX - 1 {
            for i in 0..TEXTURE_PX - 1 {
                let (u, v) = (
                    (i as f64 + 0.5) / TEXTURE_PX as f64,
                    (j as f64 + 0.5) / TEXTURE_PX as f64,
                );
                let d = (a.sample(u, v) - a.sample(u + 1.0 / TEXTURE_PX as f64, v)).abs();
                worst = worst.max(d);
            }
        }
        assert!(worst < 0.45, "step {worst}");
        let rebuilt = Texture::from_fn(TEXTURE_PX, under_eye);
        assert_eq!(&rebuilt, a);
    }
}
