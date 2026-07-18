// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// These are direct moves of the pre-existing op tests. Coverage gaps are intentionally left unchanged.

fn mask_lane(value: bool) -> i64 {
    if value { -1 } else { 0 }
}

#[simd_test]
fn native_width_i64_u64<S: Simd>(simd: S) {
    let mask_vals: Vec<i64> = (0..S::mask64s::N).map(|i| mask_lane(i % 2 == 0)).collect();
    let mask = S::mask64s::from_slice(simd, &mask_vals);

    let u_true: Vec<u64> = (0..S::u64s::N).map(|i| (1_u64 << 63) + i as u64).collect();
    let u_false: Vec<u64> = (0..S::u64s::N).map(|i| i as u64).collect();
    let u_selected = mask.select(
        S::u64s::from_slice(simd, &u_true),
        S::u64s::from_slice(simd, &u_false),
    );
    let u_expected: Vec<u64> = (0..S::u64s::N)
        .map(|i| if i % 2 == 0 { u_true[i] } else { u_false[i] })
        .collect();
    assert_eq!(u_selected.as_slice(), u_expected);
    assert_eq!(
        (S::u64s::splat(simd, 3) * 7).as_slice(),
        vec![21; S::u64s::N]
    );

    let i_true: Vec<i64> = (0..S::i64s::N)
        .map(|i| -i64::try_from(i).expect("native vector length fits in i64") - 1)
        .collect();
    let i_false: Vec<i64> = (0..S::i64s::N)
        .map(|i| i64::try_from(i).expect("native vector length fits in i64") + 1)
        .collect();
    let i_selected = mask.select(
        S::i64s::from_slice(simd, &i_true),
        S::i64s::from_slice(simd, &i_false),
    );
    let i_expected: Vec<i64> = (0..S::i64s::N)
        .map(|i| if i % 2 == 0 { i_true[i] } else { i_false[i] })
        .collect();
    assert_eq!(i_selected.as_slice(), i_expected);
    assert_eq!(
        (S::i64s::block_splat(i64x2::from_slice(simd, &[11, -12]))).as_slice(),
        [11, -12].repeat(S::i64s::N / 2)
    );
    assert_eq!(
        (S::u64s::block_splat(u64x2::from_slice(simd, &[13, 14]))).as_slice(),
        [13, 14].repeat(S::u64s::N / 2)
    );
}
