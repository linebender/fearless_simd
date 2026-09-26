//! Integration using only the production library's documented public API.
#![no_std]

pub use fearless_simd;

/// Select the detected backend, or the compilation baseline without detection.
/// The policy belongs to this optional facade, not to the SIMD core.
#[inline]
pub fn selected_level() -> fearless_simd::Level {
    fearless_simd::Level::try_detect().unwrap_or(fearless_simd::Level::baseline())
}
