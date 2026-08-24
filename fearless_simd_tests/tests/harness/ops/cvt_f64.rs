// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

#[simd_test]
fn cvt_f64_i64x2<S: Simd>(simd: S) {
    let values = [i64::MIN, i64::MAX];
    let a = i64x2::from_slice(simd, &values);
    assert_eq!(*a.to_float::<f64x2<_>>(), values.map(|x| x as f64));
}

#[simd_test]
fn cvt_f64_u64x2<S: Simd>(simd: S) {
    let values = [0, u64::MAX];
    let a = u64x2::from_slice(simd, &values);
    assert_eq!(*a.to_float::<f64x2<_>>(), values.map(|x| x as f64));
}

#[simd_test]
fn cvt_f64_i64x4<S: Simd>(simd: S) {
    let values = [
        i64::MIN,
        -9_007_199_254_740_993,
        9_007_199_254_740_993,
        i64::MAX,
    ];
    let a = i64x4::from_slice(simd, &values);
    assert_eq!(*a.to_float::<f64x4<_>>(), values.map(|x| x as f64));
}

#[simd_test]
fn cvt_f64_u64x4<S: Simd>(simd: S) {
    let values = [0, 9_007_199_254_740_993, 1 << 63, u64::MAX];
    let a = u64x4::from_slice(simd, &values);
    assert_eq!(*a.to_float::<f64x4<_>>(), values.map(|x| x as f64));
}

#[simd_test]
fn cvt_f64_i64x8<S: Simd>(simd: S) {
    use fearless_simd::SimdCvtFloat;
    let values = [
        i64::MIN,
        -9_007_199_254_740_993,
        -42,
        -1,
        0,
        1,
        9_007_199_254_740_993,
        i64::MAX,
    ];
    let a = i64x8::from_slice(simd, &values);
    let result = f64x8::float_from(a);
    assert_eq!(*result, values.map(|x| x as f64));
}

#[simd_test]
fn cvt_f64_u64x8<S: Simd>(simd: S) {
    use fearless_simd::SimdCvtFloat;
    let values = [
        0,
        1,
        42,
        1 << 53,
        (1 << 53) + 1,
        1 << 63,
        u64::MAX - 1,
        u64::MAX,
    ];
    let a = u64x8::from_slice(simd, &values);
    let result = f64x8::float_from(a);
    assert_eq!(*result, values.map(|x| x as f64));
}

#[simd_test]
#[ignore = "randomly checks 10 million signed and unsigned 64-bit integers"]
// Run with: cargo test --release cvt_f64_i64_u64_random -- --ignored
fn cvt_f64_i64_u64_random<S: Simd>(simd: S) {
    simd.vectorize(
        #[inline(always)]
        || {
            let mut rng = fastrand::Rng::with_seed(0x243f_6a88_85a3_08d3);

            for iteration in 0..2_500_000 {
                let signed = [rng.i64(..), rng.i64(..), rng.i64(..), rng.i64(..)];
                let expected_signed = signed.map(|value| (value as f64).to_bits());
                let signed_x4 = i64x4::from_slice(simd, &signed).to_float::<f64x4<_>>();
                assert_eq!(
                    (*signed_x4).map(f64::to_bits),
                    expected_signed,
                    "signed x4 iteration {iteration}",
                );
                let signed_x2 = i64x2::from_slice(simd, &signed[..2]).to_float::<f64x2<_>>();
                assert_eq!(
                    (*signed_x2).map(f64::to_bits),
                    [expected_signed[0], expected_signed[1]],
                    "signed x2 iteration {iteration}",
                );

                let unsigned = [rng.u64(..), rng.u64(..), rng.u64(..), rng.u64(..)];
                let expected_unsigned = unsigned.map(|value| (value as f64).to_bits());
                let unsigned_x4 = u64x4::from_slice(simd, &unsigned).to_float::<f64x4<_>>();
                assert_eq!(
                    (*unsigned_x4).map(f64::to_bits),
                    expected_unsigned,
                    "unsigned x4 iteration {iteration}",
                );
                let unsigned_x2 = u64x2::from_slice(simd, &unsigned[..2]).to_float::<f64x2<_>>();
                assert_eq!(
                    (*unsigned_x2).map(f64::to_bits),
                    [expected_unsigned[0], expected_unsigned[1]],
                    "unsigned x2 iteration {iteration}",
                );
            }
        },
    );
}
