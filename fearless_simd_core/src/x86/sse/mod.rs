//! Target features related to Streaming SIMD Extensions.
//!
//! These are the predecessors to the [AVX](crate::x86::avx) instructions.
//!
//! These are most commonly used through the [x86-64-v2](crate::x86::V2) microarchitecture level.
//! Some of these features are also included in [x86-64-v1](crate::x86::V1).
//!
//! These support SIMD registers of up to 128 bits.

mod fxsr;
pub use fxsr::Fxsr;

#[expect(
    clippy::module_inception,
    reason = "The inner module is automatically generated."
)]
mod sse;
pub use sse::Sse;

mod sse2;
pub use sse2::Sse2;

mod sse3;
pub use sse3::Sse3;

mod ssse3;
pub use ssse3::SupplementalSse3;

// These will be stabilised in 1.91.
// mod sse4a;
// pub use sse4a::Sse4a;

mod sse4_1;
pub use sse4_1::Sse4_1;

mod sse4_2;
pub use sse4_2::Sse4_2;
