//! The SSE target feature.

use crate::{TargetFeatureToken, trampoline};

use core::fmt::Debug;

/// A token indicating that the current CPU has the SSE target feature.
///
/// The Rust target feature name for this feature is `sse`.
///
/// See <https://en.wikipedia.org/wiki/Streaming_SIMD_Extensions> for more information about these instructions.
/// This feature also implictily enables
///
/// # Example
///
/// This can be used to [`trampoline!`] into:
///
/// ```rust
/// #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
/// #[target_feature(enable = "sse")]
/// fn uses_sse() {
///     // ...
/// }
/// ```
///
#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub struct Sse {
    // We don't use non_exhaustive because we don't want this struct to be constructible.
    // in different modules in this crate.
    _private: (),
}

impl Debug for Sse {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, r#""sse" enabled."#)
    }
}

unsafe impl TargetFeatureToken for Sse {
    const FEATURES: &[&str] = &["sse"];

    #[inline(always)]
    fn vectorize<R>(self, f: impl FnOnce() -> R) -> R {
        trampoline!([Self = self] => "sse", <(R)> fn<(R)>(f: impl FnOnce() -> R = f) -> R { f() })
    }
}

impl Sse {
    #[cfg(feature = "std")]
    /// Create a new token if the `"sse"` target feature is detected as enabled.
    ///
    /// This does not do any caching internally, although note that the standard
    /// library does internally cache the features it detects.
    // TODO: Consider a manual override feature/env var?
    pub fn try_new() -> Option<Self> {
        // Feature flag required to make docs compile.
        // TODO: Extract into a (private) crate::x86::is_x86_feature_detected?
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if std::arch::is_x86_feature_detected!("sse") {
            // Safety: The required CPU feature was detected.
            unsafe { Some(Self::new()) }
        } else {
            None
        }
    }

    #[target_feature(enable = "sse")]
    /// Create a new token for the "sse" target feature is enabled.
    ///
    /// This method is useful to get a new token if you have an external proof that
    /// SSE is available. This could happen if you have a token for a target feature
    /// which [implicitly enables] `sse`.
    ///
    /// # Safety
    ///
    /// No conditions other than those inherited from the target feature attribute,
    /// i.e. that the "sse" target feature is available.
    ///
    /// [implicitly enables]: https://doc.rust-lang.org/beta/reference/attributes/codegen.html#r-attributes.codegen.target_feature.x86
    pub fn new() -> Self {
        Self { _private: () }
    }
}

const _: () = {
    assert!(
        core::mem::size_of::<Sse>() == 0,
        "Target feature tokens should be zero sized."
    );
};
