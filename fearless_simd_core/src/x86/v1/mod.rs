//! Target features enabled in the `x86-64-v1` [microarchitecture level](https://en.wikipedia.org/wiki/X86-64#Microarchitecture_levels) on x86 and x86-64.
//!
//! This is the baseline for x86-64 support.

pub use crate::x86::sse::Fxsr;
pub use crate::x86::sse::Sse;

/// A token that the current CPU is on the x86-64-v1 microarchitecture level.
// TODO: (This is currently incomplete)
pub struct V1 {
    pub sse: Sse,
    pub fxsr: Fxsr,
}

impl V1 {
    /// Create a new token if the current CPU is at the x86-64-v1 microarchitecture level or better.
    ///
    /// This does not do any caching internally, although note that the standard
    /// library does internally cache the features it detects.
    #[cfg(feature = "std")]
    pub fn try_new() -> Option<Self> {
        // TODO: Caching
        Some(Self {
            fxsr: Fxsr::try_new()?,
            sse: Sse::try_new()?,
        })
    }
}

const _: () = {
    assert!(
        core::mem::size_of::<V1>() == 0,
        "Target feature tokens should be zero sized."
    );
};
