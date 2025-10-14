//! The FXSR target feature.

use core::fmt::Debug;

use crate::{TargetFeatureToken, trampoline};

/// A token indicating that the current CPU has the FXSR target feature.
///
/// The Rust target feature name for this feature is `fxsr`.
/// For example, this can be used to [`trampoline!`] into:
///
/// ```rust
/// #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
/// #[target_feature(enable = "fxsr")]
/// fn uses_fxsr() {
///     // ...
/// }
/// ```
#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub struct Fxsr {
    // We don't use non_exhaustive because we don't want this struct to be constructible.
    // in different modules in this crate.
    _private: (),
}

impl Debug for Fxsr {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, r#""fxsr" enabled."#)
    }
}

unsafe impl TargetFeatureToken for Fxsr {
    const FEATURES: &[&str] = &["fxsr"];

    #[inline(always)]
    fn vectorize<R>(self, f: impl FnOnce() -> R) -> R {
        trampoline!([Self = self] => "fxsr", <(R)> fn<(R)>(f: impl FnOnce() -> R = f) -> R { f() })
    }
}

impl Fxsr {
    #[cfg(feature = "std")]
    /// Create a new token if the "fxsr" target feature is detected as enabled.
    ///
    /// This does not do any caching internally, although note that the standard
    /// library does internally cache the features it detects.
    // TODO: Consider a manual override feature/env var?
    pub fn try_new() -> Option<Self> {
        // Feature flag required to make docs compile.
        // TODO: Extract into a (private) crate::x86::is_x86_feature_detected?
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if std::arch::is_x86_feature_detected!("fxsr") {
            // Safety: The required CPU feature was detected.
            unsafe { Some(Self::new()) }
        } else {
            None
        }
    }

    #[target_feature(enable = "fxsr")]
    /// Create a new token for the "fxsr" target feature is enabled.
    ///
    /// This method is useful to get a new token if you have an
    /// external proof that FXSR is available.
    ///
    /// # Safety
    ///
    /// No conditions other than those inherited from the target feature attribute,
    /// i.e. that the "fxsr" target feature is available.
    pub fn new() -> Self {
        Self { _private: () }
    }
}

const _: () = {
    assert!(
        core::mem::size_of::<Fxsr>() == 0,
        "Target feature tokens should be zero sized."
    );
};
