//! The breaker (`docs/22` section 9.2): three failures in a row (`rate_limited`, `timeout`, `cli_failed`)
//! make an advisor answer `unavailable` at once for 60 s, then one trial request goes through. A
//! success, or a `bad_answer`, `refused` or `not_signed_in`, resets the count.

use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

/// Failures in a row that open the breaker.
pub const FAILURES: u32 = 3;

/// How long an open breaker refuses requests before it lets one trial through.
pub const COOLDOWN: Duration = Duration::from_secs(60);

/// How a request ended, as far as the breaker is concerned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Record {
    /// The request answered.
    Success,
    /// The request reached the program and got an answer the breaker does not count against it
    /// (`bad_answer`, `refused`, `not_signed_in`): the count restarts.
    Neutral,
    /// `rate_limited`, `timeout` or `cli_failed`; the words say which (shown when the breaker opens).
    Failure(String),
    /// The request was abandoned (cancelled): nothing is learned.
    Abandoned,
}

#[derive(Debug)]
struct State {
    fails: u32,
    /// When the breaker opened; `Some` until a success or a neutral answer closes it again.
    opened_at: Option<Instant>,
    trial_in_flight: bool,
    last_failure: String,
}

/// One advisor's breaker. Shared by the requests that run at once.
#[derive(Debug)]
pub struct Breaker {
    state: Mutex<State>,
    failures: u32,
    cooldown: Duration,
}

/// Permission to start one request; hand it back with [`Breaker::finish`]. Dropping it without
/// finishing (a panic) gives a trial's place back.
#[derive(Debug)]
#[must_use]
pub struct Ticket<'a> {
    breaker: &'a Breaker,
    trial: bool,
    finished: bool,
}

fn lock(mutex: &Mutex<State>) -> MutexGuard<'_, State> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

impl Breaker {
    /// A closed breaker with the spec's numbers.
    #[must_use]
    pub fn new() -> Self {
        Self::with(FAILURES, COOLDOWN)
    }

    /// A closed breaker that opens after `failures` and rests for `cooldown` (the tests shorten it).
    #[must_use]
    pub fn with(failures: u32, cooldown: Duration) -> Self {
        Self {
            state: Mutex::new(State {
                fails: 0,
                opened_at: None,
                trial_in_flight: false,
                last_failure: String::new(),
            }),
            failures,
            cooldown,
        }
    }

    /// Asks to start a request. `Err` carries the last failure's words: the breaker is open, or a trial
    /// is out and this request must wait for its verdict.
    pub fn check(&self) -> Result<Ticket<'_>, String> {
        let mut state = lock(&self.state);
        let Some(opened) = state.opened_at else {
            return Ok(Ticket {
                breaker: self,
                trial: false,
                finished: false,
            });
        };
        if opened.elapsed() < self.cooldown || state.trial_in_flight {
            return Err(state.last_failure.clone());
        }
        state.trial_in_flight = true;
        Ok(Ticket {
            breaker: self,
            trial: true,
            finished: false,
        })
    }

    /// Records how the request of `ticket` ended.
    pub fn finish(&self, mut ticket: Ticket<'_>, record: Record) {
        ticket.finished = true;
        let mut state = lock(&self.state);
        if ticket.trial {
            state.trial_in_flight = false;
        }
        match record {
            Record::Success | Record::Neutral => {
                state.fails = 0;
                state.opened_at = None;
            }
            Record::Failure(words) => {
                state.fails += 1;
                state.last_failure = words;
                if ticket.trial || state.fails >= self.failures {
                    state.opened_at = Some(Instant::now());
                }
            }
            Record::Abandoned => {}
        }
    }

    /// The last failure's words while the breaker is open (it stays open until a request goes
    /// through, also after its cool-down), else `None`.
    #[must_use]
    pub fn open_reason(&self) -> Option<String> {
        let state = lock(&self.state);
        state.opened_at.map(|_| state.last_failure.clone())
    }
}

impl Default for Breaker {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Ticket<'_> {
    fn drop(&mut self) {
        if self.trial && !self.finished {
            lock(&self.breaker.state).trial_in_flight = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fail(b: &Breaker) {
        let t = b.check().expect("closed or trial");
        b.finish(t, Record::Failure("boom".into()));
    }

    #[test]
    fn three_failures_in_a_row_open_it() {
        let b = Breaker::with(3, Duration::from_secs(60));
        fail(&b);
        fail(&b);
        assert!(b.check().is_ok());
        assert!(b.open_reason().is_none());
        // (that ticket was dropped unfinished: nothing was learned)
        fail(&b);
        assert_eq!(b.check().unwrap_err(), "boom");
        assert_eq!(b.open_reason().as_deref(), Some("boom"));
    }

    #[test]
    fn a_success_or_a_neutral_answer_resets_the_count() {
        let b = Breaker::with(3, Duration::from_secs(60));
        fail(&b);
        fail(&b);
        b.finish(b.check().unwrap(), Record::Success);
        fail(&b);
        fail(&b);
        b.finish(b.check().unwrap(), Record::Neutral);
        fail(&b);
        fail(&b);
        assert!(b.check().is_ok(), "never three in a row");
    }

    #[test]
    fn one_trial_goes_through_after_the_cooldown() {
        let b = Breaker::with(1, Duration::from_millis(50));
        fail(&b);
        assert!(b.check().is_err());
        std::thread::sleep(Duration::from_millis(80));
        let trial = b.check().expect("the trial");
        assert!(b.check().is_err(), "only one trial at a time");
        // A failed trial opens it again.
        b.finish(trial, Record::Failure("again".into()));
        assert_eq!(b.check().unwrap_err(), "again");
        std::thread::sleep(Duration::from_millis(80));
        let trial = b.check().expect("the next trial");
        b.finish(trial, Record::Success);
        assert!(b.check().is_ok());
        assert!(b.open_reason().is_none());
    }

    #[test]
    fn a_trial_that_panics_gives_its_place_back() {
        let b = Breaker::with(1, Duration::from_millis(20));
        fail(&b);
        std::thread::sleep(Duration::from_millis(40));
        drop(b.check().expect("the trial"));
        assert!(b.check().is_ok(), "the next request is the trial");
    }

    #[test]
    fn an_abandoned_request_changes_nothing() {
        let b = Breaker::with(2, Duration::from_secs(60));
        fail(&b);
        b.finish(b.check().unwrap(), Record::Abandoned);
        fail(&b);
        assert!(b.check().is_err());
    }
}
