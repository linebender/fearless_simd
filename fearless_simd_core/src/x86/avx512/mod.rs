//! Target features related to the 512-bit extensions to [AVX](crate::x86::avx).
//!
//! Many of these are part of the [x86-64-v4](crate::x86::V4) microarchitecture level.
//!
//! These support SIMD registers of up to 512 bits.

mod avx512bf16;
pub use avx512bf16::Avx512bf16;

mod avx512bitalg;
pub use avx512bitalg::Avx512bitalg;

mod avx512bw;
pub use avx512bw::Avx512bw;

mod avx512cd;
pub use avx512cd::Avx512cd;

mod avx512dq;
pub use avx512dq::Avx512dq;

mod avx512f;
pub use avx512f::Avx512f;

mod avx512fp16;
pub use avx512fp16::Avx512fp16;

mod avx512ifma;
pub use avx512ifma::Avx512ifma;

mod avx512vbmi;
pub use avx512vbmi::Avx512vbmi;

mod avx512vbmi2;
pub use avx512vbmi2::Avx512vbmi2;

mod avx512vl;
pub use avx512vl::Avx512vl;

mod avx512vnni;
pub use avx512vnni::Avx512vnni;

mod avx512vp2intersect;
pub use avx512vp2intersect::Avx512vp2intersect;

mod avx512vpopcntdq;
pub use avx512vpopcntdq::Avx512vpopcntdq;
