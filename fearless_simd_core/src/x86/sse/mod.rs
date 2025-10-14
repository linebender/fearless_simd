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

mod sse4_1;
pub use sse4_1::Sse4_1;

mod sse4_2;
pub use sse4_2::Sse4_2;
