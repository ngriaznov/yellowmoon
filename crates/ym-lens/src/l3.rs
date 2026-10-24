//! L3: highlight optics (`docs/21` section 4, L3): glow, veiling glare, ghosts, starbursts and streaks.
//!
//! The composition is `out = (1 - a) Z + F (a_g K_g * R + a_v K_v * R + sum a_i G_i[H] + a_* B[H] + a_s ... S[H])`
//! with `Z` the output of L2, `R` the restored light and `H` the highlights; [`terms`] holds the budget `a`.
//!
//! The parts, in the order the light travels:
//!
//! * [`restore`]: the clip test, the 3 EV of headroom and the soft knee;
//! * [`lights`]: the extra headroom a small clipped light is assumed to hold;
//! * [`sources`]: the highlights of the lens plane as a list of lights;
//! * [`glow`]: the glow and veiling kernels, split into a core gathered per tile and a tail for the plane;
//! * [`fft`] and [`planes`]: the tail plane, convolved once per photo;
//! * [`items`]: ghosts, starbursts and streaks, drawn analytically per pixel from the list;
//! * [`pass`]: the per-tile pass that puts them together.

mod device;
mod fft;
mod geometry;
mod glow;
mod items;
mod lights;
mod pass;
mod planes;
mod profile;
mod restore;
mod sources;
mod terms;

pub use lights::boost_ev;
pub(crate) use device::{lens_data, mix, plane_data, scale_data};
pub(crate) use pass::{apply, combine, halo, terms, terms_exact};
pub(crate) use planes::FlarePlanes;
