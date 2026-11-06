//! A stderr logger for diagnosing a device that wgpu or the driver reports lost (`YM_GPU_LOG`).
//!
//! wgpu and wgpu-hal report the cause of a lost device through the `log` crate (on Metal, for example, "No active
//! command buffers for fence value", "refusing to create new command buffer" or a failed command buffer), and
//! nothing in a test or in `ym` installs a logger, so the cause is lost with the message. With `YM_GPU_LOG` set to a
//! level (`error`, `warn`, `info`, `debug`, `trace`) the first device that opens installs a logger that writes the
//! `wgpu*` and `naga` records of that level, and every record of `ym_gpu`, to stderr. It writes nowhere else and
//! does nothing when the variable is unset or another logger is installed already.

use std::io::Write as _;

struct StderrLog {
    level: log::LevelFilter,
}

impl log::Log for StderrLog {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= self.level
    }

    fn log(&self, record: &log::Record<'_>) {
        if self.enabled(record.metadata()) {
            let _ = writeln!(
                std::io::stderr(),
                "[gpu log {} {}] {}",
                record.level(),
                record.target(),
                record.args()
            );
        }
    }

    fn flush(&self) {}
}

/// Installs the stderr logger when `YM_GPU_LOG` names a level. Safe to call any number of times.
pub fn install_from_env() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let Some(level) = std::env::var("YM_GPU_LOG")
            .ok()
            .and_then(|v| v.trim().parse::<log::LevelFilter>().ok())
        else {
            return;
        };
        if log::set_boxed_logger(Box::new(StderrLog { level })).is_ok() {
            log::set_max_level(level);
        }
    });
}
