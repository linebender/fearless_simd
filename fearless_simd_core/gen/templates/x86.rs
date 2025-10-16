// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

// {AUTOGEN_COMMENT}

//! The {FEATURE_DOCS_NAME} target feature.

use crate::{TargetFeatureToken, trampoline};

use core::fmt::Debug;

/// {NEW_DOCS}
///
/// A token indicating that the current CPU has the `{FEATURE_ID}` target feature.
///
/// # Example
///
/// This can be used to [`trampoline!`] into functions like:
///
/// ```rust
/// #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
/// #[target_feature(enable = "{FEATURE_ID}")]
/// fn {EXAMPLE_FUNCTION_NAME}() {
///     // ...
/// }
/// ```
#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub struct FEATURE_STRUCT_NAME {
    // We don't use non_exhaustive because we don't want this struct to be constructible.
    // in different modules in this crate.
    _private: (),
}

impl Debug for FEATURE_STRUCT_NAME {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, r#""{FEATURE_ID}" enabled."#)
    }
}

// Safety: This token can only be constructed if you have proof that all the requisite
// target feature is enabled.
unsafe impl TargetFeatureToken for FEATURE_STRUCT_NAME {
    const FEATURES: &[&str] = &["{ENABLED_FEATURES_STR_LIST}"];

    #[inline(always)]
    fn vectorize<R>(self, f: impl FnOnce() -> R) -> R {
        // Because we need the safety check to be eagerly evaluated, it uses an constant item.
        // This means we can't use `Self = self` here, unfortunately.
        trampoline!([FEATURE_STRUCT_NAME = self] => "{FEATURE_ID}", <(R)> fn<(R)>(f: impl FnOnce() -> R = f) -> R { f() })
    }
}

impl FEATURE_STRUCT_NAME {
    #[cfg(feature = "std")]
    /// Create a new token if the `"{FEATURE_ID}"` target feature is detected as enabled.
    ///
    /// This does not do any caching internally, although note that the standard
    /// library does internally cache the features it detects.
    // TODO: Consider a manual override feature/env var?
    pub fn try_new() -> Option<Self> {
        // Feature flag required to make docs compile.
        // TODO: Extract into a (private) crate::x86::is_x86_feature_detected?
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if std::arch::is_x86_feature_detected!("{FEATURE_ID}") {
            // Safety: The required CPU feature was detected.
            unsafe { Some(Self::new()) }
        } else {
            None
        }
    }

    #[target_feature(enable = "{FEATURE_ID}")]
    /// Create a new token for the "{FEATURE_ID}" target feature.
    ///
    /// This method is useful to get a new token if you have an external proof that
    /// {FEATURE_DOCS_NAME} is available. This could happen if you are in a target feature
    /// function called by an external library user.
    ///
    /// # Safety
    ///
    /// No conditions other than those inherited from the target feature attribute,
    /// i.e. that the "{FEATURE_ID}" target feature is available.
    ///
    /// [implicitly enables]: https://doc.rust-lang.org/beta/reference/attributes/codegen.html?highlight=implicitly%20enabled#r-attributes.codegen.target_feature.safety-restrictions
    pub fn new() -> Self {
        Self { _private: () }
    }
}
/*{FROM_IMPLS}*/

const _: () = {
    assert!(
        core::mem::size_of::<FEATURE_STRUCT_NAME>() == 0,
        "Target feature tokens should be zero sized."
    );
};
