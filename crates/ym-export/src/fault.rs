//! A write fault for tests (feature `fault-injection`; compiled out of every normal build).
//!
//! `YM_WRITE_FAULT=enospc@<bytes>[,after=<files>]` makes a file of `<bytes>` or more fail with "no space left on
//! device" (errno 28) after it is encoded and before it is moved to its name, as a full disk would; the first
//! `<files>` such files are let through. `enospc@0` fails every file. A test in the same process calls [`arm`]
//! instead (it overrides the variable). The temporary file is removed as for any failed write. Same pattern as
//! `YM_GPU_FAULT` in `ym-gpu`.

/// "No space left on device" (`ENOSPC`, 28 on Linux and macOS).
#[cfg(all(feature = "fault-injection", unix))]
const ENOSPC: i32 = 28;
#[cfg(all(feature = "fault-injection", not(unix)))]
const ENOSPC: i32 = 112;

/// When files fail (feature `fault-injection`).
#[cfg(feature = "fault-injection")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WriteFault {
    /// A file of at least this many bytes fails.
    pub at_bytes: u64,
    /// This many files that qualify are let through first.
    pub after_files: u32,
}

#[cfg(feature = "fault-injection")]
mod armed {
    use super::WriteFault;
    use std::sync::{Mutex, OnceLock, PoisonError};

    static STATE: Mutex<Option<WriteFault>> = Mutex::new(None);
    static FROM_ENV: OnceLock<()> = OnceLock::new();

    /// Parses `enospc@<bytes>[,after=<files>]`.
    pub(super) fn parse(text: &str) -> Option<WriteFault> {
        let rest = text.trim().strip_prefix("enospc@")?;
        let (bytes, after) = match rest.split_once(',') {
            Some((b, a)) => (b, a.strip_prefix("after=")?.parse().ok()?),
            None => (rest, 0),
        };
        Some(WriteFault {
            at_bytes: bytes.parse().ok()?,
            after_files: after,
        })
    }

    pub(super) fn set(fault: Option<WriteFault>) {
        // The variable is read once, before the first arm or check, so a later arm always wins.
        FROM_ENV.get_or_init(|| ());
        *STATE.lock().unwrap_or_else(PoisonError::into_inner) = fault;
    }

    pub(super) fn check(bytes: u64) -> bool {
        FROM_ENV.get_or_init(|| {
            if let Ok(v) = std::env::var("YM_WRITE_FAULT") {
                match parse(&v) {
                    Some(f) => *STATE.lock().unwrap_or_else(PoisonError::into_inner) = Some(f),
                    None => {
                        eprintln!("YM_WRITE_FAULT ignored: expected enospc@<bytes>[,after=<files>]")
                    }
                }
            }
        });
        let mut st = STATE.lock().unwrap_or_else(PoisonError::into_inner);
        match st.as_mut() {
            Some(f) if bytes >= f.at_bytes => {
                if f.after_files > 0 {
                    f.after_files -= 1;
                    false
                } else {
                    true
                }
            }
            _ => false,
        }
    }
}

/// Arms (`Some`) or disarms (`None`) the fault for this process (feature `fault-injection`).
#[cfg(feature = "fault-injection")]
pub fn arm(fault: Option<WriteFault>) {
    armed::set(fault);
}

/// Fails with `ENOSPC` when the armed fault says a file of `bytes` fails. Always `Ok` in a normal build.
#[cfg(feature = "fault-injection")]
pub(crate) fn check_write(bytes: u64) -> std::io::Result<()> {
    if armed::check(bytes) {
        Err(std::io::Error::from_raw_os_error(ENOSPC))
    } else {
        Ok(())
    }
}

/// Always `Ok` in a normal build.
#[cfg(not(feature = "fault-injection"))]
#[inline]
pub(crate) fn check_write(_bytes: u64) -> std::io::Result<()> {
    Ok(())
}

#[cfg(all(test, feature = "fault-injection"))]
mod tests {
    use super::WriteFault;
    use super::armed::parse;

    #[test]
    fn the_variable_is_parsed() {
        assert_eq!(
            parse("enospc@100"),
            Some(WriteFault {
                at_bytes: 100,
                after_files: 0
            })
        );
        assert_eq!(
            parse("enospc@0,after=2"),
            Some(WriteFault {
                at_bytes: 0,
                after_files: 2
            })
        );
        assert_eq!(parse("eio@1"), None);
        assert_eq!(parse("enospc@x"), None);
    }
}
