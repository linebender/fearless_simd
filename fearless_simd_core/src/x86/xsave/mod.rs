//! Target features relating to saving processor state, as used to implement operating systems.

#[expect(
    clippy::module_inception,
    reason = "The inner module is automatically generated."
)]
mod xsave;
pub use xsave::Xsave;

mod xsavec;
pub use xsavec::Xsavec;

mod xsaveopt;
pub use xsaveopt::Xsaveopt;

pub use xsaves::Xsaves;
mod xsaves;
