#![no_std]

use fearless_simd::{dispatch, Level, Simd};

#[cfg(feature = "std")]
pub fn process(xs: &mut [u32]) {
    process_at_level(Level::new(), xs);
}

pub fn process_at_level(level: Level, xs: &mut [u32]) {
    dispatch!(level, simd => process::__simd(simd, xs));
}

pub mod process {
    use fearless_simd::Simd;

    #[inline(always)]
    pub fn __simd<S: Simd>(simd: S, xs: &mut [u32]) {
        simd.vectorize(#[inline(always)] move || {
            for x in xs {
                *x = bridge_kernels::mix::__simd(simd, *x);
            }
        });
    }
}

pub fn explicit_vectors_at_level(level: Level, xs: &mut [u32]) {
    dispatch!(level, simd => explicit_vectors(simd, xs));
}

#[inline(always)]
pub fn explicit_vectors<S: Simd>(simd: S, xs: &mut [u32]) {
    simd.vectorize(#[inline(always)] move || {
        bridge_kernels::double::__simd(simd, xs);
        process::__simd(simd, xs);
    });
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[no_mangle]
pub fn automatic_avx2(simd: fearless_simd::Avx2, xs: &mut [u32]) {
    process::__simd(simd, xs);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[no_mangle]
pub fn manual_avx2(simd: fearless_simd::Avx2, xs: &mut [u32]) {
    simd.vectorize(#[inline(always)] move || {
        for x in xs {
            *x = x.wrapping_mul(3).wrapping_add(7);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cross_crate_helper_and_mutable_slice_work() {
        let mut xs = [0, 1, 2, u32::MAX];
        process_at_level(Level::new(), &mut xs);
        assert_eq!(xs, [7, 10, 13, 4]);
    }

    #[test]
    fn explicit_native_width_vectors_work() {
        let mut xs = [42; 67];
        explicit_vectors_at_level(Level::new(), &mut xs);
        assert_eq!(xs, [259; 67]);
    }

    #[test]
    fn explicit_vectors_wrap_consistently() {
        let mut xs = [u32::MAX; 67];
        explicit_vectors_at_level(Level::new(), &mut xs);
        assert_eq!(xs, [1; 67]);
    }
}
