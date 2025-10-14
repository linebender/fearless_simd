//! The "adx" target feature.

#[expect(
    clippy::module_inception,
    reason = "The inner module is automatically generated."
)]
mod adx;
pub use adx::Adx;
