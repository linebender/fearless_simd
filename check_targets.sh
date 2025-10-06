# A script to run cargo check for fearless_simd on each supported platform.
# We currently don't run this in CI, as we expect it would take too long.
# Before using, you must run:
# rustup target add aarch64-linux-android x86_64-unknown-linux-gnu i686-pc-windows-msvc wasm32-unknown-unknown riscv64gc-unknown-linux-gnu

# Run using `sh ./check_targets.sh`
# TODO: Make into an xtask like thing so that windows users can use easily.

# Note that doing `-neon` is not sound, but we never run this so YOLO.
RUSTFLAGS=-Ctarget-feature=-neon cargo check -p fearless_simd --target aarch64-linux-android
RUSTFLAGS=-Ctarget-feature=-neon cargo check -p fearless_simd --target aarch64-linux-android --features force_support_fallback
cargo check -p fearless_simd --target aarch64-linux-android  --features force_support_fallback
cargo check -p fearless_simd --target aarch64-linux-android

RUSTFLAGS=-Ctarget-feature=+avx2,+fma cargo check -p fearless_simd --target x86_64-unknown-linux-gnu
RUSTFLAGS=-Ctarget-feature=+avx2,+fma cargo check -p fearless_simd --target x86_64-unknown-linux-gnu --features force_support_fallback
RUSTFLAGS=-Ctarget-feature=+sse4.2 cargo check -p fearless_simd --target x86_64-unknown-linux-gnu
RUSTFLAGS=-Ctarget-feature=+sse4.2 cargo check -p fearless_simd --target x86_64-unknown-linux-gnu  --features force_support_fallback
cargo check -p fearless_simd --target x86_64-unknown-linux-gnu
cargo check -p fearless_simd --target x86_64-unknown-linux-gnu  --features force_support_fallback

RUSTFLAGS=-Ctarget-feature=+avx2,+fma cargo check -p fearless_simd --target i686-pc-windows-msvc
RUSTFLAGS=-Ctarget-feature=+avx2,+fma cargo check -p fearless_simd --target i686-pc-windows-msvc --features force_support_fallback
RUSTFLAGS=-Ctarget-feature=+sse4.2 cargo check -p fearless_simd --target i686-pc-windows-msvc
RUSTFLAGS=-Ctarget-feature=+sse4.2 cargo check -p fearless_simd --target i686-pc-windows-msvc  --features force_support_fallback
cargo check -p fearless_simd --target i686-pc-windows-msvc
cargo check -p fearless_simd --target i686-pc-windows-msvc  --features force_support_fallback

cargo check -p fearless_simd --target wasm32-unknown-unknown
cargo check -p fearless_simd --target wasm32-unknown-unknown --features force_support_fallback
RUSTFLAGS=-Ctarget-feature=+simd128 cargo check -p fearless_simd --target wasm32-unknown-unknown
RUSTFLAGS=-Ctarget-feature=+simd128 cargo check -p fearless_simd --target wasm32-unknown-unknown --features force_support_fallback

cargo check -p fearless_simd --target riscv64gc-unknown-linux-gnu
cargo check -p fearless_simd --target riscv64gc-unknown-linux-gnu --features force_support_fallback
