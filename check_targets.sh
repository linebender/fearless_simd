# A script to run cargo check for fearless_simd on each supported platform.
# We currently don't run this in CI, as we expect it would take too long.
# Before using, you must run:
# rustup target add aarch64-linux-android x86_64-unknown-linux-gnu i686-pc-windows-msvc wasm32-unknown-unknown riscv64gc-unknown-linux-gnu

# Run using `sh ./check_targets.sh`
# TODO: Make into an xtask like thing so that windows users can use easily.

# aarch64, both with and without neon support.
# Note that doing `-neon` is not sound due to the standard library's ABI, but the
# binary is never executed (nor indeed is it even created), and it's still a useful sanity check.
RUSTFLAGS=-Ctarget-feature=-neon cargo check -p fearless_simd --target aarch64-linux-android
RUSTFLAGS=-Ctarget-feature=-neon cargo check -p fearless_simd --target aarch64-linux-android --features force_support_fallback
cargo check -p fearless_simd --target aarch64-linux-android  --features force_support_fallback
cargo check -p fearless_simd --target aarch64-linux-android

# x86_64, at all supported static SIMD levels.
RUSTFLAGS=-Ctarget-cpu=icelake-server cargo check -p fearless_simd --target x86_64-unknown-linux-gnu
RUSTFLAGS=-Ctarget-cpu=icelake-server cargo check -p fearless_simd --target x86_64-unknown-linux-gnu --features force_support_fallback
RUSTFLAGS=-Ctarget-cpu=x86-64-v3 cargo check -p fearless_simd --target x86_64-unknown-linux-gnu
RUSTFLAGS=-Ctarget-cpu=x86-64-v3 cargo check -p fearless_simd --target x86_64-unknown-linux-gnu --features force_support_fallback
RUSTFLAGS=-Ctarget-cpu=x86-64-v2 cargo check -p fearless_simd --target x86_64-unknown-linux-gnu
RUSTFLAGS=-Ctarget-cpu=x86-64-v2 cargo check -p fearless_simd --target x86_64-unknown-linux-gnu  --features force_support_fallback
cargo check -p fearless_simd --target x86_64-unknown-linux-gnu
cargo check -p fearless_simd --target x86_64-unknown-linux-gnu  --features force_support_fallback
cargo check -p fearless_simd --target x86_64-unknown-linux-gnu --no-default-features --features std

# x86 (i.e. 32 bit) at all supported static SIMD levels.
RUSTFLAGS=-Ctarget-cpu=x86-64-v3 cargo check -p fearless_simd --target i686-pc-windows-msvc
RUSTFLAGS=-Ctarget-cpu=x86-64-v3 cargo check -p fearless_simd --target i686-pc-windows-msvc --features force_support_fallback
RUSTFLAGS=-Ctarget-cpu=x86-64-v2 cargo check -p fearless_simd --target i686-pc-windows-msvc
RUSTFLAGS=-Ctarget-cpu=x86-64-v2 cargo check -p fearless_simd --target i686-pc-windows-msvc  --features force_support_fallback
cargo check -p fearless_simd --target i686-pc-windows-msvc
cargo check -p fearless_simd --target i686-pc-windows-msvc  --features force_support_fallback
cargo check -p fearless_simd --target i686-pc-windows-msvc --no-default-features --features std

# Wasm, both with and without SIMD.
cargo check -p fearless_simd --target wasm32-unknown-unknown
cargo check -p fearless_simd --target wasm32-unknown-unknown --features force_support_fallback
RUSTFLAGS=-Ctarget-feature=+simd128 cargo check -p fearless_simd --target wasm32-unknown-unknown
RUSTFLAGS=-Ctarget-feature=+simd128 cargo check -p fearless_simd --target wasm32-unknown-unknown --features force_support_fallback

# riscv64, which is importantly a target we don't support any SIMD levels for.
cargo check -p fearless_simd --target riscv64gc-unknown-linux-gnu
cargo check -p fearless_simd --target riscv64gc-unknown-linux-gnu --features force_support_fallback
