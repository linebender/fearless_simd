// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Explicit prototype checks; the operation harness also runs on the detected SVE2 level.

use fearless_simd::*;

#[test]
fn sve2_detection_and_dispatch() {
    #[inline(always)]
    fn native_bits<S: Simd>(_: S) -> usize {
        S::u8s::LEN * 8
    }

    let level = Level::new();
    let bits = match level {
        Level::Sve2_128(_) => 128,
        Level::Sve2_256(_) => 256,
        Level::Sve2_512(_) => 512,
        _ => 0,
    };
    // Used by the documented QEMU commands so missing support cannot silently pass.
    if let Ok(expected) = std::env::var("SVE_TEST_BITS") {
        assert_eq!(bits, expected.parse::<usize>().unwrap());
    }
    if bits != 0 {
        assert_eq!(dispatch!(level, simd => native_bits(simd)), bits);
        assert!(level.as_neon().is_some());
        assert!(!level.is_fallback());
        assert_eq!(
            format!("{level:?}"),
            format!("{:?}", Level::detect_sve2().unwrap())
        );
    } else {
        assert!(Level::detect_sve2().is_none());
    }
}

#[test]
fn sve2_array_layout() {
    assert_eq!(size_of::<f32x4<Sve2_128>>(), 16);
    assert_eq!(align_of::<f32x4<Sve2_128>>(), 16);
    assert_eq!(size_of::<f32x8<Sve2_256>>(), 32);
    assert_eq!(align_of::<f32x8<Sve2_256>>(), 32);
    assert_eq!(size_of::<f32x16<Sve2_512>>(), 64);
    assert_eq!(align_of::<f32x16<Sve2_512>>(), 64);
    assert_eq!(size_of::<u8x64<Sve2_128>>(), 64);
    assert_eq!(size_of::<u8x16<Sve2_512>>(), 16);
    assert_eq!(size_of::<Sve2_128>(), 0);
    assert_eq!(size_of::<Sve2_256>(), 0);
    assert_eq!(size_of::<Sve2_512>(), 0);
}

#[test]
fn sve2_128_operations_outside_vectorize() {
    if let Level::Sve2_128(simd) = Level::new() {
        let a = f32x4::from_slice(simd, &[1.0, 2.0, 3.0, 4.0]);
        let b = (a + a) * a;
        assert_eq!(*b, [2.0, 8.0, 18.0, 32.0]);
        assert_eq!(*b.reverse(), [32.0, 18.0, 8.0, 2.0]);
    }
}

#[test]
fn sve2_256_operations_outside_vectorize() {
    if let Level::Sve2_256(simd) = Level::new() {
        let a = f32x8::from_slice(simd, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let b = (a + a) * a;
        assert_eq!(*b, [2.0, 8.0, 18.0, 32.0, 50.0, 72.0, 98.0, 128.0]);
        assert_eq!(*a.zip_high(a), [5.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0]);
    }
}

#[test]
fn sve2_512_operations_outside_vectorize() {
    if let Level::Sve2_512(simd) = Level::new() {
        let a = u32x16::from_slice(
            simd,
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        );
        assert_eq!(
            *(a * a),
            [
                1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256
            ]
        );
        assert_eq!(
            *a.unzip_high(a),
            [2, 4, 6, 8, 10, 12, 14, 16, 2, 4, 6, 8, 10, 12, 14, 16]
        );
        assert_eq!(a.reduce_sum(), 136);
    }
}
