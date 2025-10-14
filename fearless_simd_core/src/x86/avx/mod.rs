#[expect(
    clippy::module_inception,
    reason = "The inner module is automatically generated."
)]
mod avx;
pub use avx::Avx;

mod avx2;
pub use avx2::Avx2;

mod avxifma;
pub use avxifma::Avxifma;

mod avxneconvert;
pub use avxneconvert::Avxneconvert;

mod avxvnni;
pub use avxvnni::Avxvnni;

mod avxvnniint8;
pub use avxvnniint8::Avxvnniint8;

mod avxvnniint16;
pub use avxvnniint16::Avxvnniint16;
