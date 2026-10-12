//! The viewport's state: the photo cell's rectangle, fit or a fixed zoom, the pan position and the
//! Before switch, and what pointer input does to them (`docs/01` section Native viewport, `docs/11`
//! sections Photos and the viewport checklist).
//!
//! The wire types have the shape of `src/api/types.ts` (`ViewportRect`, `ViewportInput`,
//! `ViewportMode`). The core keeps one [`ViewportState`] for the current photo; the native layer
//! asks [`Renderer::viewport_frame`](super::Renderer::viewport_frame) for a frame of it whenever
//! the state, the photo or its settings change.
//!
//! # Conventions
//!
//! * **Zoom** is *device pixels per image pixel* of the final image (after crop and rotation, at
//!   full resolution). `Fit` is the zoom at which the whole image fits the cell; `Percent(v)` is
//!   `v / 100` (so 100 % shows one image pixel on one device pixel, as the header's "Fit 24 %"
//!   counts). A fixed zoom below the fit zoom is the fit zoom.
//! * **The centre** is the point of the final image at the middle of the cell, as fractions of the
//!   image. At Fit it is the middle of the image; when the image is larger than the cell it is
//!   clamped so the cell never shows beyond the image.
//! * **Pointer coordinates** are CSS pixels inside the cell, as the webview reports them; the cell's
//!   `scale` converts them to device pixels.

use serde::{Deserialize, Serialize};

use super::geometry::Affine;

/// Largest zoom the viewport allows, device pixels per image pixel (800 %).
pub const MAX_ZOOM: f64 = 8.0;

/// Longest side of the cell the core renders, in device pixels (an 8K display's photo cell is
/// 7680 wide). A cell reported larger than this, or not a number, would ask for a frame of
/// gigabytes; it is drawn at this size instead.
pub const MAX_CELL_PX: usize = 8192;

/// The photo cell's rectangle in CSS pixels (`viewport_set_rect`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ViewportRect {
    /// Left edge in the window.
    pub x: f64,
    /// Top edge in the window.
    pub y: f64,
    /// Width in CSS pixels.
    pub width: f64,
    /// Height in CSS pixels.
    pub height: f64,
    /// Device pixels per CSS pixel.
    pub scale: f64,
}

impl ViewportRect {
    /// A cell of `width x height` CSS pixels at `scale` device pixels per CSS pixel.
    #[must_use]
    pub fn cell(width: f64, height: f64, scale: f64) -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            width,
            height,
            scale,
        }
    }

    /// The cell's size in device pixels, from 1 x 1 to [`MAX_CELL_PX`] on each side.
    #[must_use]
    pub fn device_size(&self) -> (usize, usize) {
        let s = if self.scale.is_finite() && self.scale > 0.0 {
            self.scale
        } else {
            1.0
        };
        let side = |css: f64| {
            let v = (css * s).round();
            if v.is_nan() {
                1
            } else {
                v.clamp(1.0, MAX_CELL_PX as f64) as usize
            }
        };
        (side(self.width), side(self.height))
    }
}

/// Fit to the cell, or a fixed zoom (50, 100, 200 %).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ViewportMode {
    /// The whole image in the cell.
    Fit,
    /// A fixed zoom in percent of one device pixel per image pixel.
    Percent {
        /// 50, 100, 200, ...
        value: f64,
    },
}

/// Pointer input over the photo (`viewport_input`). Coordinates are CSS pixels in the cell.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ViewportInput {
    /// Wheel or pinch: multiply the zoom by `factor` about the point.
    Zoom {
        /// The factor, above 1 zooms in.
        factor: f64,
        /// Pointer x in the cell.
        x: f64,
        /// Pointer y in the cell.
        y: f64,
    },
    /// Drag: move the image with the pointer by this many CSS pixels.
    Pan {
        /// Horizontal movement.
        dx: f64,
        /// Vertical movement.
        dy: f64,
    },
    /// Double click: toggle between Fit and 100 % about the point.
    ToggleZoom {
        /// Pointer x in the cell.
        x: f64,
        /// Pointer y in the cell.
        y: f64,
    },
    /// Click in Pick white mode.
    PickWhite {
        /// Pointer x in the cell.
        x: f64,
        /// Pointer y in the cell.
        y: f64,
    },
}

/// What an input did.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InputOutcome {
    /// The state changed (or stayed): ask for a new frame.
    Updated,
    /// A click for Pick white at this CSS position; the state is unchanged.
    PickWhite {
        /// Pointer x in the cell.
        x: f64,
        /// Pointer y in the cell.
        y: f64,
    },
}

/// The viewport (see the module documentation).
#[derive(Debug, Clone, PartialEq)]
pub struct ViewportState {
    /// The cell.
    pub rect: ViewportRect,
    /// Fit or a fixed zoom.
    pub mode: ViewportMode,
    /// The point of the image at the middle of the cell, as fractions of the final image.
    pub centre: [f64; 2],
    /// Whether the photo is shown before the look and the display transform.
    pub before: bool,
}

impl ViewportState {
    /// A viewport in `rect`, at Fit.
    #[must_use]
    pub fn new(rect: ViewportRect) -> Self {
        Self {
            rect,
            mode: ViewportMode::Fit,
            centre: [0.5, 0.5],
            before: false,
        }
    }

    /// The cell in device pixels.
    #[must_use]
    pub fn device_size(&self) -> (usize, usize) {
        self.rect.device_size()
    }

    /// The zoom at which an `image` (final-image pixels at full resolution) fits the cell.
    #[must_use]
    pub fn fit_zoom(&self, image: (usize, usize)) -> f64 {
        let (dw, dh) = self.device_size();
        (dw as f64 / image.0.max(1) as f64).min(dh as f64 / image.1.max(1) as f64)
    }

    /// The zoom in effect: device pixels per image pixel.
    #[must_use]
    pub fn zoom(&self, image: (usize, usize)) -> f64 {
        let fit = self.fit_zoom(image);
        match self.mode {
            ViewportMode::Fit => fit,
            ViewportMode::Percent { value } => (value / 100.0).clamp(fit.min(MAX_ZOOM), MAX_ZOOM),
        }
    }

    /// Whether the state shows the whole image (Fit, or a fixed zoom no larger than the fit).
    #[must_use]
    pub fn is_fit(&self, image: (usize, usize)) -> bool {
        match self.mode {
            ViewportMode::Fit => true,
            ViewportMode::Percent { value } => value / 100.0 <= self.fit_zoom(image) * 1.0001,
        }
    }

    /// The map from device pixels of the cell (pixel centres at `i + 0.5`) to pixels of the final
    /// image at full resolution.
    ///
    /// At exactly 100 % the cell's pixels are the image's pixels: the origin is rounded to a whole
    /// pixel, so the frame is a window on the image and not a resample of it half a pixel off (a
    /// pan, or a cell of odd size, would otherwise leave the origin between pixels and soften every
    /// edge, at the zoom people use to judge sharpness).
    #[must_use]
    pub fn view_to_final(&self, image: (usize, usize)) -> Affine {
        let (dw, dh) = self.device_size();
        let z = self.zoom(image);
        let c = self.centre_px(image);
        let origin = |centre: f64, view: usize, size: usize| {
            let o = centre - view as f64 * 0.5 / z;
            if (z - 1.0).abs() > 1e-9 {
                return o;
            }
            let whole = o.round();
            if size >= view {
                whole.clamp(0.0, (size - view) as f64)
            } else {
                whole
            }
        };
        Affine::scale_translate(
            1.0 / z,
            1.0 / z,
            origin(c.0, dw, image.0),
            origin(c.1, dh, image.1),
        )
    }

    /// The centre in pixels of the final image, clamped.
    #[must_use]
    pub fn centre_px(&self, image: (usize, usize)) -> (f64, f64) {
        let z = self.zoom(image);
        let (dw, dh) = self.device_size();
        let axis = |frac: f64, img: usize, view: usize| {
            let img = img as f64;
            let half = view as f64 / z * 0.5;
            if half * 2.0 >= img {
                img * 0.5
            } else {
                (frac * img).clamp(half, img - half)
            }
        };
        (
            axis(self.centre[0], image.0, dw),
            axis(self.centre[1], image.1, dh),
        )
    }

    /// The point of the final image (pixels) under a CSS point of the cell, or `None` outside the
    /// image (in the letterbox).
    #[must_use]
    pub fn final_point_at(&self, image: (usize, usize), css: (f64, f64)) -> Option<(f64, f64)> {
        let s = self.rect.scale;
        let (x, y) = self.view_to_final(image).apply(css.0 * s, css.1 * s);
        let inside = x >= 0.0 && y >= 0.0 && x <= image.0 as f64 && y <= image.1 as f64;
        inside.then_some((x, y))
    }

    /// Sets the zoom and keeps `image` point `anchor` (final-image pixels) under the device point
    /// `pointer`.
    fn zoom_about(
        &mut self,
        image: (usize, usize),
        z: f64,
        anchor: (f64, f64),
        pointer: (f64, f64),
    ) {
        let fit = self.fit_zoom(image);
        let (dw, dh) = self.device_size();
        let z = z.clamp(fit.min(MAX_ZOOM), MAX_ZOOM);
        if z <= fit * 1.0001 {
            self.mode = ViewportMode::Fit;
            self.centre = [0.5, 0.5];
            return;
        }
        self.mode = ViewportMode::Percent { value: z * 100.0 };
        let cx = anchor.0 - (pointer.0 - dw as f64 * 0.5) / z;
        let cy = anchor.1 - (pointer.1 - dh as f64 * 0.5) / z;
        self.centre = [cx / image.0.max(1) as f64, cy / image.1.max(1) as f64];
        self.settle(image);
    }

    /// Clamps the stored centre to what [`ViewportState::centre_px`] allows.
    fn settle(&mut self, image: (usize, usize)) {
        let (cx, cy) = self.centre_px(image);
        self.centre = [cx / image.0.max(1) as f64, cy / image.1.max(1) as f64];
    }

    /// Applies pointer input. `image` is the size of the final image at full resolution.
    pub fn input(&mut self, event: &ViewportInput, image: (usize, usize)) -> InputOutcome {
        let s = self.rect.scale;
        match *event {
            ViewportInput::Zoom { factor, x, y } => {
                if factor.is_finite() && factor > 0.0 {
                    let pointer = (x * s, y * s);
                    let anchor = self.view_to_final(image).apply(pointer.0, pointer.1);
                    let z = self.zoom(image) * factor;
                    self.zoom_about(image, z, anchor, pointer);
                }
                InputOutcome::Updated
            }
            ViewportInput::Pan { dx, dy } => {
                if !self.is_fit(image) && dx.is_finite() && dy.is_finite() {
                    let z = self.zoom(image);
                    let c = self.centre_px(image);
                    self.centre = [
                        (c.0 - dx * s / z) / image.0.max(1) as f64,
                        (c.1 - dy * s / z) / image.1.max(1) as f64,
                    ];
                    self.settle(image);
                }
                InputOutcome::Updated
            }
            ViewportInput::ToggleZoom { x, y } => {
                if self.is_fit(image) {
                    let pointer = (x * s, y * s);
                    let anchor = self.view_to_final(image).apply(pointer.0, pointer.1);
                    let anchor = (
                        anchor.0.clamp(0.0, image.0 as f64),
                        anchor.1.clamp(0.0, image.1 as f64),
                    );
                    self.zoom_about(image, 1.0, anchor, pointer);
                } else {
                    self.mode = ViewportMode::Fit;
                    self.centre = [0.5, 0.5];
                }
                InputOutcome::Updated
            }
            ViewportInput::PickWhite { x, y } => InputOutcome::PickWhite { x, y },
        }
    }

    /// Sets the mode (`viewport_set_mode`): Fit, or a fixed zoom about the middle of the cell.
    pub fn set_mode(&mut self, mode: ViewportMode, image: (usize, usize)) {
        match mode {
            ViewportMode::Fit => {
                self.mode = ViewportMode::Fit;
                self.centre = [0.5, 0.5];
            }
            ViewportMode::Percent { value } if value.is_finite() && value > 0.0 => {
                let (dw, dh) = self.device_size();
                let anchor = self.centre_px(image);
                self.zoom_about(
                    image,
                    value / 100.0,
                    anchor,
                    (dw as f64 * 0.5, dh as f64 * 0.5),
                );
            }
            ViewportMode::Percent { .. } => {}
        }
    }

    /// Sets the cell's rectangle (`viewport_set_rect`) and re-clamps the pan.
    pub fn set_rect(&mut self, rect: ViewportRect, image: (usize, usize)) {
        self.rect = rect;
        self.settle(image);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state() -> ViewportState {
        // A 600 x 400 cell at scale 2: 1200 x 800 device pixels.
        ViewportState::new(ViewportRect::cell(600.0, 400.0, 2.0))
    }

    const IMAGE: (usize, usize) = (6000, 4000);

    #[test]
    fn fit_zoom_is_the_header_percentage() {
        let v = state();
        assert_eq!(v.device_size(), (1200, 800));
        assert!((v.fit_zoom(IMAGE) - 0.2).abs() < 1e-12);
        assert!((v.zoom(IMAGE) - 0.2).abs() < 1e-12);
        assert!(v.is_fit(IMAGE));
    }

    #[test]
    fn zooming_keeps_the_point_under_the_pointer() {
        let mut v = state();
        // Pointer at CSS (450, 100): the image point under it before and after must be equal.
        let before = v.final_point_at(IMAGE, (450.0, 100.0)).expect("inside");
        v.input(
            &ViewportInput::Zoom {
                factor: 4.0,
                x: 450.0,
                y: 100.0,
            },
            IMAGE,
        );
        assert!(!v.is_fit(IMAGE));
        assert!((v.zoom(IMAGE) - 0.8).abs() < 1e-9);
        let after = v.final_point_at(IMAGE, (450.0, 100.0)).expect("inside");
        assert!((before.0 - after.0).abs() < 1e-6 && (before.1 - after.1).abs() < 1e-6);
        // Zooming back out below the fit returns to Fit and the middle.
        v.input(
            &ViewportInput::Zoom {
                factor: 0.1,
                x: 450.0,
                y: 100.0,
            },
            IMAGE,
        );
        assert_eq!(v.mode, ViewportMode::Fit);
        assert_eq!(v.centre, [0.5, 0.5]);
    }

    #[test]
    fn panning_moves_the_image_with_the_pointer_and_stops_at_the_edges() {
        let mut v = state();
        // No pan at Fit.
        v.input(&ViewportInput::Pan { dx: 50.0, dy: 0.0 }, IMAGE);
        assert_eq!(v.centre, [0.5, 0.5]);
        v.set_mode(ViewportMode::Percent { value: 100.0 }, IMAGE);
        assert!((v.zoom(IMAGE) - 1.0).abs() < 1e-12);
        let c0 = v.centre_px(IMAGE);
        v.input(&ViewportInput::Pan { dx: 10.0, dy: 0.0 }, IMAGE);
        let c1 = v.centre_px(IMAGE);
        // 10 CSS px = 20 device px = 20 image px at 100 %: the centre moves left by 20.
        assert!((c0.0 - c1.0 - 20.0).abs() < 1e-9 && (c0.1 - c1.1).abs() < 1e-9);
        // Dragging far to the right stops at the left edge: the cell shows x = 0.
        v.input(&ViewportInput::Pan { dx: 1e9, dy: 1e9 }, IMAGE);
        let (dw, dh) = v.device_size();
        let c = v.centre_px(IMAGE);
        assert_eq!(c, (dw as f64 / 2.0, dh as f64 / 2.0));
    }

    #[test]
    fn double_click_toggles_between_fit_and_100_percent_about_the_pointer() {
        let mut v = state();
        let p = v.final_point_at(IMAGE, (150.0, 300.0)).expect("inside");
        v.input(&ViewportInput::ToggleZoom { x: 150.0, y: 300.0 }, IMAGE);
        assert!((v.zoom(IMAGE) - 1.0).abs() < 1e-12);
        let after = v.final_point_at(IMAGE, (150.0, 300.0)).expect("inside");
        assert!((p.0 - after.0).abs() < 1e-6 && (p.1 - after.1).abs() < 1e-6);
        v.input(&ViewportInput::ToggleZoom { x: 10.0, y: 10.0 }, IMAGE);
        assert_eq!(v.mode, ViewportMode::Fit);
    }

    #[test]
    fn a_small_image_is_letterboxed_and_clicks_outside_it_find_nothing() {
        let v = state();
        // A square image in a 3:2 cell: the letterbox is left and right.
        let image = (1000, 1000);
        assert!(v.final_point_at(image, (5.0, 200.0)).is_none());
        assert!(v.final_point_at(image, (300.0, 200.0)).is_some());
        let m = v.view_to_final(image);
        let (x, y) = m.apply(600.0, 400.0); // the middle of the 1200 x 800 device cell
        assert!((x - 500.0).abs() < 1e-9 && (y - 500.0).abs() < 1e-9);
    }

    #[test]
    fn pick_white_reports_the_click_and_changes_nothing() {
        let mut v = state();
        let before = v.clone();
        assert_eq!(
            v.input(&ViewportInput::PickWhite { x: 12.0, y: 34.0 }, IMAGE),
            InputOutcome::PickWhite { x: 12.0, y: 34.0 }
        );
        assert_eq!(v, before);
    }

    #[test]
    fn the_wire_shapes_are_those_of_the_typescript_types() {
        let input: ViewportInput =
            serde_json::from_str(r#"{"kind":"zoom","factor":1.5,"x":10,"y":20}"#).expect("zoom");
        assert_eq!(
            input,
            ViewportInput::Zoom {
                factor: 1.5,
                x: 10.0,
                y: 20.0
            }
        );
        let input: ViewportInput =
            serde_json::from_str(r#"{"kind":"toggle_zoom","x":1,"y":2}"#).expect("toggle");
        assert_eq!(input, ViewportInput::ToggleZoom { x: 1.0, y: 2.0 });
        let input: ViewportInput =
            serde_json::from_str(r#"{"kind":"pick_white","x":1,"y":2}"#).expect("pick");
        assert_eq!(input, ViewportInput::PickWhite { x: 1.0, y: 2.0 });
        let mode: ViewportMode = serde_json::from_str(r#"{"kind":"fit"}"#).expect("fit");
        assert_eq!(mode, ViewportMode::Fit);
        let mode: ViewportMode =
            serde_json::from_str(r#"{"kind":"percent","value":200}"#).expect("percent");
        assert_eq!(mode, ViewportMode::Percent { value: 200.0 });
        assert_eq!(
            serde_json::to_string(&ViewportMode::Fit).expect("json"),
            r#"{"kind":"fit"}"#
        );
        let rect: ViewportRect =
            serde_json::from_str(r#"{"x":1,"y":2,"width":300,"height":200,"scale":2}"#)
                .expect("rect");
        assert_eq!(rect.device_size(), (600, 400));
    }
}
