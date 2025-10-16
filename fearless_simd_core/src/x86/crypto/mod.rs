//! Cryptogryphy related target features, including hashing, random number generation, and encryption.
//!
//! These are not generally part of the standardised microarchitecture levels.

mod aes;
pub use aes::Aes;

mod gfni;
pub use gfni::Gfni;

mod kl;
pub use kl::Keylocker;

mod pclmulqdq;
pub use pclmulqdq::Pclmulqdq;

mod rdrand;
pub use rdrand::Rdrand;

mod rdseed;
pub use rdseed::Rdseed;

mod sha;
pub use sha::Sha;

mod sha512;
pub use sha512::Sha512;

mod sm3;
pub use sm3::Sm3;

mod sm4;
pub use sm4::Sm4;

mod vaes;
pub use vaes::Vaes;

mod vpclmulqdq;
pub use vpclmulqdq::Vpclmulqdq;

mod widekl;
pub use widekl::WideKeylocker;
