//! The x86-64-{LEVEL_ID} microarchitecture level.

use crate::{TargetFeatureToken, trampoline};

use core::fmt::Debug;

// TODO: Level specific docs?
/// A token indicating that the current CPU has the x86-64-{LEVEL_ID} microarchitecture level.
///
/// For more details on the microarchitecture levels, see
/// <https://en.wikipedia.org/wiki/X86-64#Microarchitecture_levels>.
///
/// # Example
///
/// This can be used to [`trampoline!`] into functions like:
///
/// ```rust
/// #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
/// #[target_feature(enable = "{LEVEL_FEATURE_LCD_CONTENTS}")]
/// fn uses_x86_64_{LEVEL_ID}() {
///     // ...
/// }
/// ```
///
/// This struct internally contains only the minimal features required to enable this level.
/// This is done to ensure that the fewest target features are checked.
/// However, it can be turned into any target feature it implies using the from impls.
#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub struct LEVEL_STRUCT_NAME {
    /*{LEVEL_FEATURE_LCD_FIELD_DEFINITIONS}*/
    // This struct explicitly is not non_exhaustive, because it is
    // completely safe to construct from the fields.
}

impl Debug for LEVEL_STRUCT_NAME {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, r#"x86-64-{LEVEL_ID} enabled."#)
    }
}

// Safety: This token can only be constructed if you have proofs that all the requisite
// target features are enabled.
unsafe impl TargetFeatureToken for LEVEL_STRUCT_NAME {
    const FEATURES: &[&str] = &["{LEVEL_FEATURE_SUPERSET_LIST}"];

    #[inline(always)]
    fn vectorize<R>(self, f: impl FnOnce() -> R) -> R {
        // We use the explicitly written out form here as validation that the set of
        // features we've created correctly mapes to the target feature string.
        trampoline!([{LEVEL_FEATURE_LCD_TRAMPOLINE}] => "{LEVEL_FEATURE_LCD_CONTENTS}", <(R)> fn<(R)>(f: impl FnOnce() -> R = f) -> R { f() })
    }
}

impl LEVEL_STRUCT_NAME {
    #[cfg(feature = "std")]
    /// Create a new token if the x86-64-{LEVEL_ID} target feature is detected as enabled.
    ///
    /// This does not do any caching internally, although note that the standard
    /// library does internally cache the features it detects.
    // TODO: Consider a manual override feature/env var?
    pub fn try_new() -> Option<Self> {
        Some(Self {
            /*{LEVEL_FEATURE_STRUCT_INITIALIZER_LCD_TRY_NEW}*/
        })
    }

    #[target_feature(enable = "{LEVEL_FEATURE_LCD_CONTENTS}")]
    /// Create a new token for the x86-64-{LEVEL_ID} microarchitecture level.
    ///
    /// This method is useful to get a new token if you have an external proof that
    /// x86-64-{LEVEL_ID} is available. This could happen if you are in a target feature
    /// function called by an external library user.
    ///
    /// # Safety
    ///
    /// No conditions other than those inherited from the target feature attribute,
    /// i.e. that the "{LEVEL_FEATURE_LCD_CONTENTS}" target feature is available.
    pub fn new() -> Self {
        Self {
            /*{LEVEL_FEATURE_STRUCT_INITIALIZER_LCD_NEW}*/
        }
    }
}
// TODO: From impls to convert into lower x86 versions.

/*{FROM_IMPLS}*/

const _: () = {
    assert!(
        core::mem::size_of::<LEVEL_STRUCT_NAME>() == 0,
        "Target feature tokens should be zero sized."
    );
};
