// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

fn mul_q15(a: i16, b: i16) -> i16 {
    ((i32::from(a) * i32::from(b) + (1 << 14)) >> 15).min(i32::from(i16::MAX)) as i16
}

fn check<const N: usize>(a: [i16; N], b: [i16; N], fast: [i16; N], precise: [i16; N]) {
    for i in 0..N {
        let expected = mul_q15(a[i], b[i]);
        assert_eq!(precise[i], expected, "mul_q15_precise({}, {})", a[i], b[i]);

        if (a[i], b[i]) != (i16::MIN, i16::MIN) {
            assert_eq!(fast[i], expected, "mul_q15({}, {})", a[i], b[i]);
        }
    }
}

/// Pairs of the extremes, of ties that round upwards from both signs, and of ordinary values.
const LHS: [i16; 32] = [
    i16::MIN,
    i16::MIN,
    i16::MAX,
    i16::MAX,
    i16::MIN,
    0,
    1,
    -1,
    3,
    -3,
    2,
    -2,
    16384,
    -16384,
    100,
    -100,
    12345,
    -12345,
    1,
    -1,
    7,
    8191,
    -8192,
    20000,
    -20000,
    30000,
    255,
    16320,
    -32767,
    32767,
    4,
    -5,
];
const RHS: [i16; 32] = [
    i16::MIN,
    i16::MAX,
    i16::MAX,
    i16::MIN,
    1,
    i16::MIN,
    16384,
    16384,
    16384,
    16384,
    8192,
    8192,
    16384,
    16384,
    16384,
    16384,
    23456,
    23456,
    16383,
    16383,
    -7,
    4,
    4,
    -30000,
    -30000,
    30000,
    32767,
    4681,
    -32767,
    -1,
    i16::MIN,
    i16::MIN,
];

const SPLAT_RHS: [i16; 8] = [i16::MIN, i16::MAX, 16384, -16384, 1, -1, 12345, 0];

fn cases<const N: usize>() -> impl Iterator<Item = ([i16; N], [i16; N])> {
    let pairs = LHS.as_chunks::<N>().0.iter().zip(RHS.as_chunks::<N>().0);
    let every_lhs = SPLAT_RHS.into_iter().flat_map(|b| {
        (i16::MIN..=i16::MAX)
            .step_by(N)
            .map(move |start| (core::array::from_fn(|i| start + i as i16), [b; N]))
    });

    pairs.map(|(a, b)| (*a, *b)).chain(every_lhs)
}

#[simd_test]
fn mul_q15_i16x8<S: Simd>(simd: S) {
    for (a, b) in cases::<8>() {
        let (lhs, rhs) = (i16x8::from_slice(simd, &a), i16x8::from_slice(simd, &b));
        let fast = *simd.mul_q15_i16x8(lhs, rhs);
        let precise = *simd.mul_q15_precise_i16x8(lhs, rhs);
        check(a, b, fast, precise);
    }
}

#[simd_test]
fn mul_q15_i16x16<S: Simd>(simd: S) {
    for (a, b) in cases::<16>() {
        let (lhs, rhs) = (i16x16::from_slice(simd, &a), i16x16::from_slice(simd, &b));
        let fast = *simd.mul_q15_i16x16(lhs, rhs);
        let precise = *simd.mul_q15_precise_i16x16(lhs, rhs);
        check(a, b, fast, precise);
    }
}

#[simd_test]
fn mul_q15_i16x32<S: Simd>(simd: S) {
    for (a, b) in cases::<32>() {
        let (lhs, rhs) = (i16x32::from_slice(simd, &a), i16x32::from_slice(simd, &b));
        let fast = *simd.mul_q15_i16x32(lhs, rhs);
        let precise = *simd.mul_q15_precise_i16x32(lhs, rhs);
        check(a, b, fast, precise);
    }
}
