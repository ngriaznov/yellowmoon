//! The system's accessibility display options (`docs/28` A-P12, `docs/11` *Display options*).
//!
//! On macOS, System Settings has three switches under Accessibility, Display that the page cannot
//! count on WebKit to report: Reduce Transparency, Reduce Motion and Increase Contrast. This module
//! reads them from `NSWorkspace` ([`read`], the command `display_options()`) and, through
//! [`watch`], sends the event `display_options` each time the system says one of them changed
//! (`NSWorkspaceAccessibilityDisplayOptionsDidChangeNotification`, posted on the workspace's own
//! notification centre, on the main thread). The page turns the three booleans into the root
//! attributes `data-reduce-transparency`, `data-reduce-motion` and `data-increase-contrast`, which
//! the CSS answers next to the media queries.
//!
//! Off macOS there is nothing to ask: [`read`] says all three are off and [`watch`] does nothing, so
//! no event is ever sent. Nothing is saved; the system is asked each time.

use crate::wire::DisplayOptions;

/// The three display options now. All false off macOS.
#[must_use]
pub fn read() -> DisplayOptions {
    #[cfg(target_os = "macos")]
    {
        read_system()
    }
    #[cfg(not(target_os = "macos"))]
    {
        DisplayOptions::default()
    }
}

/// Sends `display_options` whenever the system's settings change. Call it once, on the main thread
/// (the setup of the app). A no-op off macOS.
pub fn watch<R: tauri::Runtime>(app: tauri::AppHandle<R>) {
    #[cfg(target_os = "macos")]
    watch_system(app);
    #[cfg(not(target_os = "macos"))]
    let _ = app;
}

/// Asks `NSWorkspace`.
#[cfg(target_os = "macos")]
fn read_system() -> DisplayOptions {
    let workspace = objc2_app_kit::NSWorkspace::sharedWorkspace();
    DisplayOptions {
        reduce_transparency: workspace.accessibilityDisplayShouldReduceTransparency(),
        reduce_motion: workspace.accessibilityDisplayShouldReduceMotion(),
        increase_contrast: workspace.accessibilityDisplayShouldIncreaseContrast(),
    }
}

/// Registers the observer. The block runs on the main thread, reads three booleans and emits; it
/// takes no lock and waits for nothing.
#[cfg(target_os = "macos")]
fn watch_system<R: tauri::Runtime>(app: tauri::AppHandle<R>) {
    use std::ptr::NonNull;

    use block2::RcBlock;
    use objc2_app_kit::{NSWorkspace, NSWorkspaceAccessibilityDisplayOptionsDidChangeNotification};
    use objc2_foundation::NSNotification;
    use tauri_specta::Event;

    let block = RcBlock::new(move |_notification: NonNull<NSNotification>| {
        if let Err(e) = read_system().emit(&app) {
            tracing::debug!(error = %e, "could not send the display options");
        }
    });
    let center = NSWorkspace::sharedWorkspace().notificationCenter();
    // SAFETY: the name is the notification's own constant; no object filter and no queue means the
    // block runs on the posting thread; the block is `Fn` and owns what it captures.
    let observer = unsafe {
        center.addObserverForName_object_queue_usingBlock(
            Some(NSWorkspaceAccessibilityDisplayOptionsDidChangeNotification),
            None,
            None,
            &block,
        )
    };
    // The observer lives as long as the app: the centre keeps the block, and dropping the token
    // would not remove it (only `removeObserver:` does), so there is nothing to hold.
    std::mem::forget(observer);
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn the_payload_has_the_three_names_of_the_page() {
        let all = DisplayOptions {
            reduce_transparency: true,
            reduce_motion: false,
            increase_contrast: true,
        };
        assert_eq!(
            serde_json::to_value(all).expect("serialise"),
            json!({ "reduce_transparency": true, "reduce_motion": false, "increase_contrast": true })
        );
        assert_eq!(
            serde_json::from_value::<DisplayOptions>(json!({
                "reduce_transparency": false, "reduce_motion": true, "increase_contrast": false
            }))
            .expect("deserialise"),
            DisplayOptions {
                reduce_transparency: false,
                reduce_motion: true,
                increase_contrast: false
            }
        );
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn off_macos_all_three_are_off_and_watching_does_nothing() {
        use tauri::test::{mock_builder, mock_context, noop_assets};

        assert_eq!(read(), DisplayOptions::default());
        assert_eq!(
            serde_json::to_value(read()).expect("serialise"),
            json!({ "reduce_transparency": false, "reduce_motion": false, "increase_contrast": false })
        );
        let app = mock_builder()
            .build(mock_context(noop_assets()))
            .expect("the mock app");
        watch(app.handle().clone());
    }
}
