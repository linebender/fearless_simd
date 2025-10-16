//! The "adx" target feature, used for arbitrary precision integer addition.

#[expect(
    clippy::module_inception,
    reason = "The inner module is automatically generated."
)]
mod adx;
pub use adx::Adx;
