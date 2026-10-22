//! The subject and distraction maps (`docs/20-auto-crop.md` §6).

use crate::picture::{self, CELL, Plane, percentile};
use crate::saliency;
use crate::{CameraHint, Picture};

/// The camera point's weight far from the point.
const POINT_FLOOR: f64 = 0.35;
/// The point's spread as a fraction of the map's short side, at least.
const POINT_SPREAD: f64 = 0.15;
/// The point's spread as a multiple of the AF box's larger side, at least.
const POINT_BOX_SPREAD: f64 = 0.75;
/// Cells with evidence at least this times its 99th percentile form components.
const COMPONENT_LEVEL: f64 = 0.5;
/// The component holding the camera's point is the subject when it holds at least this share of
/// the thresholded mass.
const POINT_COMPONENT_SHARE: f64 = 0.1;
/// Other components of at least this times the subject's mass join it.
const JOIN_SHARE: f64 = 0.5;

/// The maps the proposer reads, at a quarter of the picture's size (`docs/20` §6 step 5).
#[derive(Debug, Clone, Default)]
pub(crate) struct SubjectMaps {
    /// Cells across: `⌊W / 4⌋`.
    pub width: usize,
    /// Cells down: `⌊H / 4⌋`.
    pub height: usize,
    /// The subject map `S`: the evidence on the subject's cells, summing to 1 (all 0 when the
    /// picture has no subject).
    pub subject: Vec<f64>,
    /// The distraction map `D`: the evidence on the distractions' cells, summing to 1 when there
    /// are any.
    pub distraction: Vec<f64>,
    /// `ρ`: the distractions' share of the thresholded mass.
    pub rho: f64,
}

/// The subject and distraction maps of a picture with the camera's hints.
pub(crate) fn subject_maps(picture: &Picture<'_>, hint: &CameraHint) -> SubjectMaps {
    let lab = picture::lab_cells(picture);
    let (mw, mh) = (lab[0].width, lab[0].height);
    if mw == 0 || mh == 0 {
        return SubjectMaps::default();
    }
    let m = saliency::saliency(&lab);
    let (focus, f) = saliency::focus(&picture::luma(picture), mw, mh);
    let mut evidence = Plane::zeros(mw, mh);
    for (i, e) in evidence.data.iter_mut().enumerate() {
        *e = (m.data[i] + f * focus.data[i]) / (1.0 + f);
    }
    // The camera's point in cells (cell i spans [i, i + 1)), if it is inside the picture.
    let point = hint.point.and_then(|[a, b]| {
        let inside = (0.0..=1.0).contains(&a) && (0.0..=1.0).contains(&b);
        inside.then(|| {
            (
                a * picture.width() as f64 / CELL as f64,
                b * picture.height() as f64 / CELL as f64,
            )
        })
    });
    if let Some((px, py)) = point {
        let box_side = hint.box_size.map_or(0.0, |[bw, bh]| {
            (bw * picture.width() as f64 / CELL as f64)
                .max(bh * picture.height() as f64 / CELL as f64)
        });
        let sigma = (POINT_SPREAD * mw.min(mh) as f64).max(POINT_BOX_SPREAD * box_side);
        for y in 0..mh {
            for x in 0..mw {
                let (dx, dy) = (x as f64 + 0.5 - px, y as f64 + 0.5 - py);
                let g = (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp();
                evidence.data[y * mw + x] *= POINT_FLOOR + (1.0 - POINT_FLOOR) * g;
            }
        }
    }
    let point_cell = point.map(|(px, py)| {
        let x = (px.floor().max(0.0) as usize).min(mw - 1);
        let y = (py.floor().max(0.0) as usize).min(mh - 1);
        y * mw + x
    });
    split(&evidence, point_cell)
}

/// Splits the evidence into the subject and the distractions (`docs/20` §6 step 5).
fn split(evidence: &Plane, point_cell: Option<usize>) -> SubjectMaps {
    let (w, h) = (evidence.width, evidence.height);
    let level = COMPONENT_LEVEL * percentile(&evidence.data, 0.99);
    let mut maps = SubjectMaps {
        width: w,
        height: h,
        subject: vec![0.0; w * h],
        distraction: vec![0.0; w * h],
        rho: 0.0,
    };
    if level <= 0.0 {
        return maps;
    }
    let on = |i: usize| evidence.data[i] >= level;
    // 4-connected components, labelled in raster order.
    let mut label = vec![usize::MAX; w * h];
    let mut masses: Vec<f64> = Vec::new();
    let mut stack = Vec::new();
    for start in 0..w * h {
        if !on(start) || label[start] != usize::MAX {
            continue;
        }
        let id = masses.len();
        let mut mass = 0.0;
        label[start] = id;
        stack.push(start);
        while let Some(i) = stack.pop() {
            mass += evidence.data[i];
            let (x, y) = (i % w, i / w);
            let mut visit = |j: usize| {
                if on(j) && label[j] == usize::MAX {
                    label[j] = id;
                    stack.push(j);
                }
            };
            if y > 0 {
                visit(i - w);
            }
            if x > 0 {
                visit(i - 1);
            }
            if x + 1 < w {
                visit(i + 1);
            }
            if y + 1 < h {
                visit(i + w);
            }
        }
        masses.push(mass);
    }
    let total: f64 = masses.iter().sum();
    if masses.is_empty() || total <= 0.0 {
        return maps;
    }
    let heaviest = masses
        .iter()
        .enumerate()
        .fold(0, |best, (i, &m)| if m > masses[best] { i } else { best });
    let core = point_cell
        .map(|c| label[c])
        .filter(|&id| id != usize::MAX && masses[id] >= POINT_COMPONENT_SHARE * total)
        .unwrap_or(heaviest);
    let in_subject: Vec<bool> = masses
        .iter()
        .enumerate()
        .map(|(i, &m)| i == core || m >= JOIN_SHARE * masses[core])
        .collect();
    let subject_mass: f64 = masses
        .iter()
        .zip(&in_subject)
        .filter(|(_, s)| **s)
        .map(|(m, _)| m)
        .sum();
    let distraction_mass = total - subject_mass;
    for (i, &id) in label.iter().enumerate() {
        if id == usize::MAX {
            continue;
        }
        if in_subject[id] {
            maps.subject[i] = evidence.data[i] / subject_mass;
        } else if distraction_mass > 0.0 {
            maps.distraction[i] = evidence.data[i] / distraction_mass;
        }
    }
    maps.rho = if in_subject.iter().all(|s| *s) {
        0.0
    } else {
        distraction_mass / total
    };
    maps
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Evidence of 0.1 everywhere with blobs of given value and rectangle.
    fn evidence(w: usize, h: usize, blobs: &[(usize, usize, usize, usize, f64)]) -> Plane {
        let mut p = Plane::zeros(w, h);
        p.data.fill(0.1);
        for &(x0, y0, x1, y1, v) in blobs {
            for y in y0..y1 {
                for x in x0..x1 {
                    p.data[y * w + x] = v;
                }
            }
        }
        p
    }

    #[test]
    fn the_heaviest_component_is_the_subject_and_light_ones_distract() {
        // A big blob, a blob of 60 % of its mass (joins it) and one of 20 % (distracts).
        let e = evidence(
            40,
            30,
            &[
                (2, 2, 12, 12, 1.0),
                (20, 2, 26, 12, 1.0),
                (30, 20, 35, 24, 1.0),
            ],
        );
        let maps = split(&e, None);
        let sum_s: f64 = maps.subject.iter().sum();
        let sum_d: f64 = maps.distraction.iter().sum();
        assert!((sum_s - 1.0).abs() < 1e-12 && (sum_d - 1.0).abs() < 1e-12);
        assert!(maps.subject[5 * 40 + 5] > 0.0 && maps.subject[5 * 40 + 22] > 0.0);
        assert!(maps.distraction[22 * 40 + 32] > 0.0 && maps.subject[22 * 40 + 32] == 0.0);
        assert!((maps.rho - 20.0 / 180.0).abs() < 1e-12, "{}", maps.rho);
    }

    #[test]
    fn the_camera_point_picks_its_component() {
        let e = evidence(40, 30, &[(2, 2, 12, 12, 1.0), (30, 20, 35, 26, 1.0)]);
        // The small blob holds 30 / 130 of the mass, at least 10 %: it is the subject, and the
        // big one (more than half of it) joins.
        let maps = split(&e, Some(22 * 40 + 32));
        assert!(maps.subject[22 * 40 + 32] > 0.0 && maps.subject[5 * 40 + 5] > 0.0);
        assert_eq!(maps.rho, 0.0);
        // A point on the background falls back to the heaviest.
        let maps = split(&e, Some(0));
        assert!(maps.subject[5 * 40 + 5] > 0.0 && maps.subject[22 * 40 + 32] == 0.0);
        assert!(maps.rho > 0.0);
    }

    #[test]
    fn a_flat_picture_has_no_subject() {
        let rgb = vec![128u8; 64 * 48 * 3];
        let p = Picture::new(64, 48, &rgb).unwrap();
        let maps = subject_maps(&p, &CameraHint::default());
        assert_eq!((maps.width, maps.height), (16, 12));
        assert!(maps.subject.iter().all(|v| *v == 0.0));
        assert_eq!(maps.rho, 0.0);
    }
}
