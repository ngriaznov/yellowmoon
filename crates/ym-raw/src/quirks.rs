//! The `CameraQuirks` trait and its registry keyed by model ID, for body-specific decode quirks (`docs/03-camera-profiles.md` section Per-camera code hook).
//!
//! Profiles carry everything about a camera that is data (colour, noise priors). Code is needed
//! only when a body has a decoding quirk: a wrong crop, a black level the file states
//! incorrectly, a sensor that is prone to the "star eater" artefact. Such bodies get a type that
//! implements [`CameraQuirks`] and an entry in [`REGISTRY`], keyed by Sony's model ID
//! (`RawMetadata::model_id`, maker note tag 0xB001).
//!
//! The registry is empty today: no body needs a quirk yet. To add one:
//!
//! 1. Write `struct MyBody; impl CameraQuirks for MyBody { ... }` overriding only the methods
//!    that apply. The defaults change nothing.
//! 2. Add `(model_id, &MyBody)` to [`REGISTRY`].
//! 3. Add a test that builds a file for that model with `testgen` and checks the adjusted
//!    metadata.
//!
//! The probe calls [`apply`] last, so quirks see and may change the fully assembled metadata.
//! Everything else reads quirks through [`quirks_for`].

use crate::layout::Rect;
use crate::metadata::RawMetadata;

/// Per-body adjustments applied after the metadata has been read.
pub trait CameraQuirks: Send + Sync {
    /// Fixes up metadata the file states wrongly (a bad crop, a missing lens name).
    fn adjust_metadata(&self, _meta: &mut RawMetadata) {}

    /// The crop to use instead of the one in the file, inside the raw readout.
    fn crop(&self, _meta: &RawMetadata) -> Option<Rect> {
        None
    }

    /// A black level to use instead of the one in the file, per CFA position.
    fn black_level_override(&self, _meta: &RawMetadata) -> Option<[u16; 4]> {
        None
    }

    /// True when this body can lose isolated bright stars to in-camera spatial filtering of the
    /// raw data, so the pipeline should be careful with hot-pixel removal.
    fn star_eater_risk(&self, _meta: &RawMetadata) -> bool {
        false
    }
}

/// Quirks of a body that has none. Every method keeps its default.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoQuirks;

impl CameraQuirks for NoQuirks {}

/// Bodies with quirks, keyed by Sony model ID. Empty for now (see the module documentation).
pub static REGISTRY: &[(u16, &'static dyn CameraQuirks)] = &[];

/// The quirks of the body with Sony model ID `model_id`, or [`NoQuirks`] when it has none or the
/// ID is unknown.
pub fn quirks_for(model_id: Option<u16>) -> &'static dyn CameraQuirks {
    quirks_in(REGISTRY, model_id)
}

/// Looks `model_id` up in an explicit registry (the production one is [`REGISTRY`]).
pub fn quirks_in(
    registry: &'static [(u16, &'static dyn CameraQuirks)],
    model_id: Option<u16>,
) -> &'static dyn CameraQuirks {
    static NONE: NoQuirks = NoQuirks;
    model_id
        .and_then(|id| registry.iter().find(|(key, _)| *key == id))
        .map_or(&NONE, |(_, quirks)| *quirks)
}

/// Runs the body's `adjust_metadata` hook, then applies its crop and black-level overrides.
pub fn apply(meta: &mut RawMetadata) {
    apply_from(REGISTRY, meta);
}

/// [`apply`] against an explicit registry.
pub fn apply_from(registry: &'static [(u16, &'static dyn CameraQuirks)], meta: &mut RawMetadata) {
    let quirks = quirks_in(registry, meta.model_id);
    quirks.adjust_metadata(meta);
    if let Some(crop) = quirks.crop(meta) {
        meta.crop = Some(crop);
    }
    if let Some(black) = quirks.black_level_override(meta) {
        let positions = meta.cfa.rggb_to_positions(black.map(f32::from));
        meta.black = positions;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testgen::ArwBuilder;

    struct Odd;

    impl CameraQuirks for Odd {
        fn adjust_metadata(&self, meta: &mut RawMetadata) {
            meta.lens_model = Some("Adjusted".to_owned());
        }
        fn crop(&self, _meta: &RawMetadata) -> Option<Rect> {
            Some(Rect {
                x: 2,
                y: 2,
                width: 8,
                height: 8,
            })
        }
        fn black_level_override(&self, _meta: &RawMetadata) -> Option<[u16; 4]> {
            Some([100, 101, 102, 103])
        }
        fn star_eater_risk(&self, _meta: &RawMetadata) -> bool {
            true
        }
    }

    static ODD: Odd = Odd;
    static TEST_REGISTRY: &[(u16, &dyn CameraQuirks)] = &[(0x1234, &ODD)];

    fn meta_with_model_id(model_id: u16) -> RawMetadata {
        let file = ArwBuilder::new("ILCE-7RM4", 16, 16)
            .model_id(model_id)
            .build(&[1000u16; 16 * 16]);
        crate::metadata::probe_bytes(&file).expect("synthetic file probes")
    }

    #[test]
    fn production_registry_is_empty_and_changes_nothing() {
        assert!(REGISTRY.is_empty());
        let meta = meta_with_model_id(0x1234);
        let before = meta.clone();
        let mut after = meta;
        apply(&mut after);
        assert_eq!(before, after);
        assert!(!quirks_for(Some(0x1234)).star_eater_risk(&after));
        assert!(!quirks_for(None).star_eater_risk(&after));
    }

    #[test]
    fn registered_quirks_are_found_by_model_id_and_applied() {
        let mut meta = meta_with_model_id(0x1234);
        assert!(quirks_in(TEST_REGISTRY, Some(0x1234)).star_eater_risk(&meta));
        assert!(!quirks_in(TEST_REGISTRY, Some(0x4321)).star_eater_risk(&meta));
        apply_from(TEST_REGISTRY, &mut meta);
        assert_eq!(meta.lens_model.as_deref(), Some("Adjusted"));
        assert_eq!(
            meta.crop,
            Some(Rect {
                x: 2,
                y: 2,
                width: 8,
                height: 8
            })
        );
        assert_eq!(meta.black, [100.0, 101.0, 102.0, 103.0]);
    }

    #[test]
    fn other_model_ids_are_untouched() {
        let mut meta = meta_with_model_id(0x4321);
        let before = meta.clone();
        apply_from(TEST_REGISTRY, &mut meta);
        assert_eq!(meta, before);
    }
}
