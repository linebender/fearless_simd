// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

fn mask_lane(value: bool) -> i64 {
    if value { -1 } else { 0 }
}

#[simd_test]
fn construct_i64x2<S: Simd>(simd: S) {
    let a = i64x2::from_slice(simd, &[-9, 18]);
    let mut stored = [0_i64; 2];
    a.store_slice(&mut stored);
    assert_eq!(stored, [-9, 18]);
    assert_eq!(*i64x2::splat(simd, -9), [-9, -9]);
    assert_eq!(*i64x2::simd_from(simd, [-9, 18]), [-9, 18]);
    assert_eq!(*i64x2::from_fn(simd, |i| [-9, 18][i]), [-9, 18]);
    assert_eq!(*i64x2::from_bytes(a.to_bytes()), [-9, 18]);
}

#[simd_test]
fn construct_i64x4<S: Simd>(simd: S) {
    let vals = [-9, 18, i64::MAX - 7, i64::MIN + 9];
    let a = i64x4::from_slice(simd, &vals);
    let mut stored = [0_i64; 4];
    a.store_slice(&mut stored);
    assert_eq!(stored, vals);
    assert_eq!(*i64x4::splat(simd, -9), [-9, -9, -9, -9]);
    assert_eq!(*i64x4::simd_from(simd, vals), vals);
    assert_eq!(*i64x4::from_fn(simd, |i| vals[i]), vals);
    assert_eq!(*i64x4::from_bytes(a.to_bytes()), vals);
}

#[simd_test]
fn construct_i64x8<S: Simd>(simd: S) {
    let vals = [-9, 18, i64::MAX - 7, i64::MIN + 9, 123, -456, 789, -1024];
    let a = i64x8::from_slice(simd, &vals);
    let mut stored = [0_i64; 8];
    a.store_slice(&mut stored);
    assert_eq!(stored, vals);
    assert_eq!(*i64x8::splat(simd, -9), [-9, -9, -9, -9, -9, -9, -9, -9]);
    assert_eq!(*i64x8::simd_from(simd, vals), vals);
    assert_eq!(*i64x8::from_fn(simd, |i| vals[i]), vals);
    assert_eq!(*i64x8::from_bytes(a.to_bytes()), vals);
}

#[simd_test]
fn construct_u64x2<S: Simd>(simd: S) {
    let vals = [0, 1_u64 << 63];
    let a = u64x2::from_slice(simd, &vals);
    let mut stored = [0_u64; 2];
    a.store_slice(&mut stored);
    assert_eq!(stored, vals);
    assert_eq!(*u64x2::splat(simd, vals[0]), [0, 0]);
    assert_eq!(*u64x2::simd_from(simd, vals), vals);
    assert_eq!(*u64x2::from_fn(simd, |i| vals[i]), vals);
    assert_eq!(*u64x2::from_bytes(a.to_bytes()), vals);
}

#[simd_test]
fn construct_u64x4<S: Simd>(simd: S) {
    let vals = [0, 1_u64 << 63, u64::MAX - 3, 42];
    let a = u64x4::from_slice(simd, &vals);
    let mut stored = [0_u64; 4];
    a.store_slice(&mut stored);
    assert_eq!(stored, vals);
    assert_eq!(*u64x4::splat(simd, vals[0]), [0, 0, 0, 0]);
    assert_eq!(*u64x4::simd_from(simd, vals), vals);
    assert_eq!(*u64x4::from_fn(simd, |i| vals[i]), vals);
    assert_eq!(*u64x4::from_bytes(a.to_bytes()), vals);
}

#[simd_test]
fn construct_u64x8<S: Simd>(simd: S) {
    let vals = [
        0,
        1_u64 << 63,
        u64::MAX - 3,
        42,
        17,
        99,
        123456789,
        u64::MAX,
    ];
    let a = u64x8::from_slice(simd, &vals);
    let mut stored = [0_u64; 8];
    a.store_slice(&mut stored);
    assert_eq!(stored, vals);
    assert_eq!(*u64x8::splat(simd, vals[0]), [0, 0, 0, 0, 0, 0, 0, 0]);
    assert_eq!(*u64x8::simd_from(simd, vals), vals);
    assert_eq!(*u64x8::from_fn(simd, |i| vals[i]), vals);
    assert_eq!(*u64x8::from_bytes(a.to_bytes()), vals);
}

#[simd_test]
fn arithmetic_i64x2<S: Simd>(simd: S) {
    let a = i64x2::from_slice(simd, &[-9, 18]);
    let b = i64x2::from_slice(simd, &[3, -6]);
    assert_eq!(*(a + b), [-6, 12]);
    assert_eq!(*(a - b), [-12, 24]);
    assert_eq!(*(a * b), [-27, -108]);
}

#[simd_test]
fn arithmetic_i64x4<S: Simd>(simd: S) {
    let a = i64x4::from_slice(simd, &[-9, 18, i64::MAX - 7, i64::MIN + 9]);
    let b = i64x4::from_slice(simd, &[3, -6, 5, -7]);
    assert_eq!(
        *(a + b),
        [-6, 12, 9223372036854775805, -9223372036854775806]
    );
    assert_eq!(
        *(a - b),
        [-12, 24, 9223372036854775795, -9223372036854775792]
    );
    assert_eq!(
        *(a * b),
        [-27, -108, 9223372036854775768, 9223372036854775745]
    );
}

#[simd_test]
fn arithmetic_i64x8<S: Simd>(simd: S) {
    let a = i64x8::from_slice(
        simd,
        &[-9, 18, i64::MAX - 7, i64::MIN + 9, 123, -456, 789, -1024],
    );
    let b = i64x8::from_slice(simd, &[3, -6, 5, -7, -11, 13, -17, 19]);
    assert_eq!(
        *(a + b),
        [
            -6,
            12,
            9223372036854775805,
            -9223372036854775806,
            112,
            -443,
            772,
            -1005
        ]
    );
    assert_eq!(
        *(a - b),
        [
            -12,
            24,
            9223372036854775795,
            -9223372036854775792,
            134,
            -469,
            806,
            -1043
        ]
    );
    assert_eq!(
        *(a * b),
        [
            -27,
            -108,
            9223372036854775768,
            9223372036854775745,
            -1353,
            -5928,
            -13413,
            -19456
        ]
    );
}

#[simd_test]
fn arithmetic_u64x2<S: Simd>(simd: S) {
    let a = u64x2::from_slice(simd, &[0, 1_u64 << 63]);
    let b = u64x2::from_slice(simd, &[u64::MAX, 7]);
    assert_eq!(*(a + b), [u64::MAX, 9223372036854775815]);
    assert_eq!(*(a - b), [1, 9223372036854775801]);
    assert_eq!(*(a * b), [0, 1_u64 << 63]);
}

#[simd_test]
fn arithmetic_u64x4<S: Simd>(simd: S) {
    let a = u64x4::from_slice(simd, &[0, 1_u64 << 63, u64::MAX - 3, 42]);
    let b = u64x4::from_slice(simd, &[u64::MAX, 7, 13, 999]);
    assert_eq!(*(a + b), [u64::MAX, 9223372036854775815, 9, 1041]);
    assert_eq!(
        *(a - b),
        [
            1,
            9223372036854775801,
            18446744073709551599,
            18446744073709550659
        ]
    );
    assert_eq!(*(a * b), [0, 1_u64 << 63, 18446744073709551564, 41958]);
}

#[simd_test]
fn arithmetic_u64x8<S: Simd>(simd: S) {
    let a = u64x8::from_slice(
        simd,
        &[
            0,
            1_u64 << 63,
            u64::MAX - 3,
            42,
            17,
            99,
            123456789,
            u64::MAX,
        ],
    );
    let b = u64x8::from_slice(simd, &[u64::MAX, 7, 13, 999, 29, 11, 987654321, 1]);
    assert_eq!(
        *(a + b),
        [
            u64::MAX,
            9223372036854775815,
            9,
            1041,
            46,
            110,
            1111111110,
            0
        ]
    );
    assert_eq!(
        *(a - b),
        [
            1,
            9223372036854775801,
            18446744073709551599,
            18446744073709550659,
            18446744073709551604,
            88,
            18446744072845354084,
            18446744073709551614
        ]
    );
    assert_eq!(
        *(a * b),
        [
            0,
            1_u64 << 63,
            18446744073709551564,
            41958,
            493,
            1089,
            121932631112635269,
            u64::MAX
        ]
    );
}

#[simd_test]
fn neg_i64x8<S: Simd>(simd: S) {
    let a = i64x8::from_slice(simd, &[-1, 2, -3, 4, -5, 6, -7, 8]);
    assert_eq!(*(-a), [1, -2, 3, -4, 5, -6, 7, -8]);
}

#[simd_test]
fn bitwise_i64x2<S: Simd>(simd: S) {
    let a = i64x2::from_slice(simd, &[-9, 18]);
    let b = i64x2::from_slice(simd, &[3, -6]);
    assert_eq!(*(a & b), [3, 18]);
    assert_eq!(*(a | b), [-9, -6]);
    assert_eq!(*(a ^ b), [-12, -24]);
    assert_eq!(*(!a), [8, -19]);
}

#[simd_test]
fn bitwise_i64x4<S: Simd>(simd: S) {
    let a = i64x4::from_slice(simd, &[-9, 18, i64::MAX - 7, i64::MIN + 9]);
    let b = i64x4::from_slice(simd, &[3, -6, 5, -7]);
    assert_eq!(*(a & b), [3, 18, 0, -9223372036854775799]);
    assert_eq!(*(a | b), [-9, -6, 9223372036854775805, -7]);
    assert_eq!(
        *(a ^ b),
        [-12, -24, 9223372036854775805, 9223372036854775792]
    );
    assert_eq!(*(!a), [8, -19, -9223372036854775801, 9223372036854775798]);
}

#[simd_test]
fn bitwise_i64x8<S: Simd>(simd: S) {
    let a = i64x8::from_slice(
        simd,
        &[-9, 18, i64::MAX - 7, i64::MIN + 9, 123, -456, 789, -1024],
    );
    let b = i64x8::from_slice(simd, &[3, -6, 5, -7, -11, 13, -17, 19]);
    assert_eq!(*(a & b), [3, 18, 0, -9223372036854775799, 113, 8, 773, 0]);
    assert_eq!(
        *(a | b),
        [-9, -6, 9223372036854775805, -7, -1, -451, -1, -1005]
    );
    assert_eq!(
        *(a ^ b),
        [
            -12,
            -24,
            9223372036854775805,
            9223372036854775792,
            -114,
            -459,
            -774,
            -1005
        ]
    );
    assert_eq!(
        *(!a),
        [
            8,
            -19,
            -9223372036854775801,
            9223372036854775798,
            -124,
            455,
            -790,
            1023
        ]
    );
}

#[simd_test]
fn bitwise_u64x2<S: Simd>(simd: S) {
    let a = u64x2::from_slice(simd, &[0, 1_u64 << 63]);
    let b = u64x2::from_slice(simd, &[u64::MAX, 7]);
    assert_eq!(*(a & b), [0, 0]);
    assert_eq!(*(a | b), [u64::MAX, 9223372036854775815]);
    assert_eq!(*(a ^ b), [u64::MAX, 9223372036854775815]);
    assert_eq!(*(!a), [u64::MAX, 9223372036854775807]);
}

#[simd_test]
fn bitwise_u64x4<S: Simd>(simd: S) {
    let a = u64x4::from_slice(simd, &[0, 1_u64 << 63, u64::MAX - 3, 42]);
    let b = u64x4::from_slice(simd, &[u64::MAX, 7, 13, 999]);
    assert_eq!(*(a & b), [0, 0, 12, 34]);
    assert_eq!(
        *(a | b),
        [u64::MAX, 9223372036854775815, 18446744073709551613, 1007]
    );
    assert_eq!(
        *(a ^ b),
        [u64::MAX, 9223372036854775815, 18446744073709551601, 973]
    );
    assert_eq!(
        *(!a),
        [u64::MAX, 9223372036854775807, 3, 18446744073709551573]
    );
}

#[simd_test]
fn bitwise_u64x8<S: Simd>(simd: S) {
    let a = u64x8::from_slice(
        simd,
        &[
            0,
            1_u64 << 63,
            u64::MAX - 3,
            42,
            17,
            99,
            123456789,
            u64::MAX,
        ],
    );
    let b = u64x8::from_slice(simd, &[u64::MAX, 7, 13, 999, 29, 11, 987654321, 1]);
    assert_eq!(*(a & b), [0, 0, 12, 34, 17, 3, 39471121, 1]);
    assert_eq!(
        *(a | b),
        [
            u64::MAX,
            9223372036854775815,
            18446744073709551613,
            1007,
            29,
            107,
            1071639989,
            u64::MAX
        ]
    );
    assert_eq!(
        *(a ^ b),
        [
            u64::MAX,
            9223372036854775815,
            18446744073709551601,
            973,
            12,
            104,
            1032168868,
            18446744073709551614
        ]
    );
    assert_eq!(
        *(!a),
        [
            u64::MAX,
            9223372036854775807,
            3,
            18446744073709551573,
            18446744073709551598,
            18446744073709551516,
            18446744073586094826,
            0
        ]
    );
}

#[simd_test]
fn shifts_i64x2<S: Simd>(simd: S) {
    let a = i64x2::from_slice(simd, &[-9, 18]);
    let shifts = i64x2::from_slice(simd, &[0, 1]);
    assert_eq!(*(a << 2_u32), [-36, 72]);
    assert_eq!(*(a >> 2_u32), [-3, 4]);
    assert_eq!(*(a << shifts), [-9, 36]);
    assert_eq!(*(a >> shifts), [-9, 9]);
}

#[simd_test]
fn shifts_i64x4<S: Simd>(simd: S) {
    let a = i64x4::from_slice(simd, &[-9, 18, i64::MAX - 7, i64::MIN + 9]);
    let shifts = i64x4::from_slice(simd, &[0, 1, 2, 3]);
    assert_eq!(*(a << 2_u32), [-36, 72, -32, 36]);
    assert_eq!(
        *(a >> 2_u32),
        [-3, 4, 2305843009213693950, -2305843009213693950]
    );
    assert_eq!(*(a << shifts), [-9, 36, -32, 72]);
    assert_eq!(
        *(a >> shifts),
        [-9, 9, 2305843009213693950, -1152921504606846975]
    );
}

#[simd_test]
fn shifts_i64x8<S: Simd>(simd: S) {
    let a = i64x8::from_slice(
        simd,
        &[-9, 18, i64::MAX - 7, i64::MIN + 9, 123, -456, 789, -1024],
    );
    let shifts = i64x8::from_slice(simd, &[0, 1, 2, 3, 0, 1, 2, 3]);
    assert_eq!(*(a << 2_u32), [-36, 72, -32, 36, 492, -1824, 3156, -4096]);
    assert_eq!(
        *(a >> 2_u32),
        [
            -3,
            4,
            2305843009213693950,
            -2305843009213693950,
            30,
            -114,
            197,
            -256
        ]
    );
    assert_eq!(*(a << shifts), [-9, 36, -32, 72, 123, -912, 3156, -8192]);
    assert_eq!(
        *(a >> shifts),
        [
            -9,
            9,
            2305843009213693950,
            -1152921504606846975,
            123,
            -228,
            197,
            -128
        ]
    );
}

#[simd_test]
fn shifts_u64x2<S: Simd>(simd: S) {
    let a = u64x2::from_slice(simd, &[0, 1_u64 << 63]);
    let shifts = u64x2::from_slice(simd, &[0, 1]);
    assert_eq!(*(a << 2_u32), [0, 0]);
    assert_eq!(*(a >> 2_u32), [0, 2305843009213693952]);
    assert_eq!(*(a << shifts), [0, 0]);
    assert_eq!(*(a >> shifts), [0, 4611686018427387904]);
}

#[simd_test]
fn shifts_u64x4<S: Simd>(simd: S) {
    let a = u64x4::from_slice(simd, &[0, 1_u64 << 63, u64::MAX - 3, 42]);
    let shifts = u64x4::from_slice(simd, &[0, 1, 2, 3]);
    assert_eq!(*(a << 2_u32), [0, 0, 18446744073709551600, 168]);
    assert_eq!(
        *(a >> 2_u32),
        [0, 2305843009213693952, 4611686018427387903, 10]
    );
    assert_eq!(*(a << shifts), [0, 0, 18446744073709551600, 336]);
    assert_eq!(
        *(a >> shifts),
        [0, 4611686018427387904, 4611686018427387903, 5]
    );
}

#[simd_test]
fn shifts_u64x8<S: Simd>(simd: S) {
    let a = u64x8::from_slice(
        simd,
        &[
            0,
            1_u64 << 63,
            u64::MAX - 3,
            42,
            17,
            99,
            123456789,
            u64::MAX,
        ],
    );
    let shifts = u64x8::from_slice(simd, &[0, 1, 2, 3, 0, 1, 2, 3]);
    assert_eq!(
        *(a << 2_u32),
        [
            0,
            0,
            18446744073709551600,
            168,
            68,
            396,
            493827156,
            18446744073709551612
        ]
    );
    assert_eq!(
        *(a >> 2_u32),
        [
            0,
            2305843009213693952,
            4611686018427387903,
            10,
            4,
            24,
            30864197,
            4611686018427387903
        ]
    );
    assert_eq!(
        *(a << shifts),
        [
            0,
            0,
            18446744073709551600,
            336,
            17,
            198,
            493827156,
            18446744073709551608
        ]
    );
    assert_eq!(
        *(a >> shifts),
        [
            0,
            4611686018427387904,
            4611686018427387903,
            5,
            17,
            49,
            30864197,
            2305843009213693951
        ]
    );
}

#[simd_test]
fn compare_i64x2<S: Simd>(simd: S) {
    let a = i64x2::from_slice(simd, &[-9, 18]);
    let b = i64x2::from_slice(simd, &[3, -6]);
    assert_eq!(<[i64; 2]>::from(a.simd_eq(b)), [0, 0]);
    assert_eq!(<[i64; 2]>::from(a.simd_lt(b)), [-1, 0]);
    assert_eq!(<[i64; 2]>::from(a.simd_le(b)), [-1, 0]);
    assert_eq!(<[i64; 2]>::from(a.simd_ge(b)), [0, -1]);
    assert_eq!(<[i64; 2]>::from(a.simd_gt(b)), [0, -1]);
}

#[simd_test]
fn compare_i64x4<S: Simd>(simd: S) {
    let a = i64x4::from_slice(simd, &[-9, 18, i64::MAX - 7, i64::MIN + 9]);
    let b = i64x4::from_slice(simd, &[3, -6, 5, -7]);
    assert_eq!(<[i64; 4]>::from(a.simd_eq(b)), [0, 0, 0, 0]);
    assert_eq!(<[i64; 4]>::from(a.simd_lt(b)), [-1, 0, 0, -1]);
    assert_eq!(<[i64; 4]>::from(a.simd_le(b)), [-1, 0, 0, -1]);
    assert_eq!(<[i64; 4]>::from(a.simd_ge(b)), [0, -1, -1, 0]);
    assert_eq!(<[i64; 4]>::from(a.simd_gt(b)), [0, -1, -1, 0]);
}

#[simd_test]
fn compare_i64x8<S: Simd>(simd: S) {
    let a = i64x8::from_slice(
        simd,
        &[-9, 18, i64::MAX - 7, i64::MIN + 9, 123, -456, 789, -1024],
    );
    let b = i64x8::from_slice(simd, &[3, -6, 5, -7, -11, 13, -17, 19]);
    assert_eq!(<[i64; 8]>::from(a.simd_eq(b)), [0, 0, 0, 0, 0, 0, 0, 0]);
    assert_eq!(<[i64; 8]>::from(a.simd_lt(b)), [-1, 0, 0, -1, 0, -1, 0, -1]);
    assert_eq!(<[i64; 8]>::from(a.simd_le(b)), [-1, 0, 0, -1, 0, -1, 0, -1]);
    assert_eq!(<[i64; 8]>::from(a.simd_ge(b)), [0, -1, -1, 0, -1, 0, -1, 0]);
    assert_eq!(<[i64; 8]>::from(a.simd_gt(b)), [0, -1, -1, 0, -1, 0, -1, 0]);
}

#[simd_test]
fn compare_u64x2<S: Simd>(simd: S) {
    let a = u64x2::from_slice(simd, &[0, 1_u64 << 63]);
    let b = u64x2::from_slice(simd, &[u64::MAX, 7]);
    assert_eq!(<[i64; 2]>::from(a.simd_eq(b)), [0, 0]);
    assert_eq!(<[i64; 2]>::from(a.simd_lt(b)), [-1, 0]);
    assert_eq!(<[i64; 2]>::from(a.simd_le(b)), [-1, 0]);
    assert_eq!(<[i64; 2]>::from(a.simd_ge(b)), [0, -1]);
    assert_eq!(<[i64; 2]>::from(a.simd_gt(b)), [0, -1]);
}

#[simd_test]
fn compare_u64x4<S: Simd>(simd: S) {
    let a = u64x4::from_slice(simd, &[0, 1_u64 << 63, u64::MAX - 3, 42]);
    let b = u64x4::from_slice(simd, &[u64::MAX, 7, 13, 999]);
    assert_eq!(<[i64; 4]>::from(a.simd_eq(b)), [0, 0, 0, 0]);
    assert_eq!(<[i64; 4]>::from(a.simd_lt(b)), [-1, 0, 0, -1]);
    assert_eq!(<[i64; 4]>::from(a.simd_le(b)), [-1, 0, 0, -1]);
    assert_eq!(<[i64; 4]>::from(a.simd_ge(b)), [0, -1, -1, 0]);
    assert_eq!(<[i64; 4]>::from(a.simd_gt(b)), [0, -1, -1, 0]);
}

#[simd_test]
fn compare_u64x8<S: Simd>(simd: S) {
    let a = u64x8::from_slice(
        simd,
        &[
            0,
            1_u64 << 63,
            u64::MAX - 3,
            42,
            17,
            99,
            123456789,
            u64::MAX,
        ],
    );
    let b = u64x8::from_slice(simd, &[u64::MAX, 7, 13, 999, 29, 11, 987654321, 1]);
    assert_eq!(<[i64; 8]>::from(a.simd_eq(b)), [0, 0, 0, 0, 0, 0, 0, 0]);
    assert_eq!(<[i64; 8]>::from(a.simd_lt(b)), [-1, 0, 0, -1, -1, 0, -1, 0]);
    assert_eq!(<[i64; 8]>::from(a.simd_le(b)), [-1, 0, 0, -1, -1, 0, -1, 0]);
    assert_eq!(<[i64; 8]>::from(a.simd_ge(b)), [0, -1, -1, 0, 0, -1, 0, -1]);
    assert_eq!(<[i64; 8]>::from(a.simd_gt(b)), [0, -1, -1, 0, 0, -1, 0, -1]);
}

#[simd_test]
fn min_max_i64x2<S: Simd>(simd: S) {
    let a = i64x2::from_slice(simd, &[-9, 18]);
    let b = i64x2::from_slice(simd, &[3, -6]);
    assert_eq!(*a.min(b), [-9, -6]);
    assert_eq!(*a.max(b), [3, 18]);
}

#[simd_test]
fn min_max_i64x4<S: Simd>(simd: S) {
    let a = i64x4::from_slice(simd, &[-9, 18, i64::MAX - 7, i64::MIN + 9]);
    let b = i64x4::from_slice(simd, &[3, -6, 5, -7]);
    assert_eq!(*a.min(b), [-9, -6, 5, -9223372036854775799]);
    assert_eq!(*a.max(b), [3, 18, 9223372036854775800, -7]);
}

#[simd_test]
fn min_max_i64x8<S: Simd>(simd: S) {
    let a = i64x8::from_slice(
        simd,
        &[-9, 18, i64::MAX - 7, i64::MIN + 9, 123, -456, 789, -1024],
    );
    let b = i64x8::from_slice(simd, &[3, -6, 5, -7, -11, 13, -17, 19]);
    assert_eq!(
        *a.min(b),
        [-9, -6, 5, -9223372036854775799, -11, -456, -17, -1024]
    );
    assert_eq!(
        *a.max(b),
        [3, 18, 9223372036854775800, -7, 123, 13, 789, 19]
    );
}

#[simd_test]
fn min_max_u64x2<S: Simd>(simd: S) {
    let a = u64x2::from_slice(simd, &[0, 1_u64 << 63]);
    let b = u64x2::from_slice(simd, &[u64::MAX, 7]);
    assert_eq!(*a.min(b), [0, 7]);
    assert_eq!(*a.max(b), [u64::MAX, 1_u64 << 63]);
}

#[simd_test]
fn min_max_u64x4<S: Simd>(simd: S) {
    let a = u64x4::from_slice(simd, &[0, 1_u64 << 63, u64::MAX - 3, 42]);
    let b = u64x4::from_slice(simd, &[u64::MAX, 7, 13, 999]);
    assert_eq!(*a.min(b), [0, 7, 13, 42]);
    assert_eq!(*a.max(b), [u64::MAX, 1_u64 << 63, u64::MAX - 3, 999]);
}

#[simd_test]
fn min_max_u64x8<S: Simd>(simd: S) {
    let a = u64x8::from_slice(
        simd,
        &[
            0,
            1_u64 << 63,
            u64::MAX - 3,
            42,
            17,
            99,
            123456789,
            u64::MAX,
        ],
    );
    let b = u64x8::from_slice(simd, &[u64::MAX, 7, 13, 999, 29, 11, 987654321, 1]);
    assert_eq!(*a.min(b), [0, 7, 13, 42, 17, 11, 123456789, 1]);
    assert_eq!(
        *a.max(b),
        [
            u64::MAX,
            1_u64 << 63,
            u64::MAX - 3,
            999,
            29,
            99,
            987654321,
            u64::MAX
        ]
    );
}

#[simd_test]
fn select_i64x2<S: Simd>(simd: S) {
    let mask = mask64x2::from_slice(simd, &[-1, 0]);
    let a = i64x2::from_slice(simd, &[-9, 18]);
    let b = i64x2::from_slice(simd, &[3, -6]);
    assert_eq!(*mask.select(a, b), [-9, -6]);
}

#[simd_test]
fn select_i64x4<S: Simd>(simd: S) {
    let mask = mask64x4::from_slice(simd, &[-1, 0, -1, 0]);
    let a = i64x4::from_slice(simd, &[-9, 18, i64::MAX - 7, i64::MIN + 9]);
    let b = i64x4::from_slice(simd, &[3, -6, 5, -7]);
    assert_eq!(*mask.select(a, b), [-9, -6, 9223372036854775800, -7]);
}

#[simd_test]
fn select_i64x8<S: Simd>(simd: S) {
    let mask = mask64x8::from_slice(simd, &[-1, 0, -1, 0, -1, 0, -1, 0]);
    let a = i64x8::from_slice(
        simd,
        &[-9, 18, i64::MAX - 7, i64::MIN + 9, 123, -456, 789, -1024],
    );
    let b = i64x8::from_slice(simd, &[3, -6, 5, -7, -11, 13, -17, 19]);
    assert_eq!(
        *mask.select(a, b),
        [-9, -6, 9223372036854775800, -7, 123, 13, 789, 19]
    );
}

#[simd_test]
fn select_u64x2<S: Simd>(simd: S) {
    let mask = mask64x2::from_slice(simd, &[-1, 0]);
    let a = u64x2::from_slice(simd, &[0, 1_u64 << 63]);
    let b = u64x2::from_slice(simd, &[u64::MAX, 7]);
    assert_eq!(*mask.select(a, b), [0, 7]);
}

#[simd_test]
fn select_u64x4<S: Simd>(simd: S) {
    let mask = mask64x4::from_slice(simd, &[-1, 0, -1, 0]);
    let a = u64x4::from_slice(simd, &[0, 1_u64 << 63, u64::MAX - 3, 42]);
    let b = u64x4::from_slice(simd, &[u64::MAX, 7, 13, 999]);
    assert_eq!(*mask.select(a, b), [0, 7, u64::MAX - 3, 999]);
}

#[simd_test]
fn select_u64x8<S: Simd>(simd: S) {
    let mask = mask64x8::from_slice(simd, &[-1, 0, -1, 0, -1, 0, -1, 0]);
    let a = u64x8::from_slice(
        simd,
        &[
            0,
            1_u64 << 63,
            u64::MAX - 3,
            42,
            17,
            99,
            123456789,
            u64::MAX,
        ],
    );
    let b = u64x8::from_slice(simd, &[u64::MAX, 7, 13, 999, 29, 11, 987654321, 1]);
    assert_eq!(
        *mask.select(a, b),
        [0, 7, u64::MAX - 3, 999, 17, 11, 123456789, 1]
    );
}

#[simd_test]
fn slide_i64x2<S: Simd>(simd: S) {
    let a = i64x2::from_slice(simd, &[1, 2]);
    let b = i64x2::from_slice(simd, &[3, 4]);
    assert_eq!(*a.slide::<1>(b), [2, 3]);
}

#[simd_test]
fn slide_i64x4<S: Simd>(simd: S) {
    let a = i64x4::from_slice(simd, &[1, 2, 3, 4]);
    let b = i64x4::from_slice(simd, &[5, 6, 7, 8]);
    assert_eq!(*a.slide::<1>(b), [2, 3, 4, 5]);
}

#[simd_test]
fn slide_i64x8<S: Simd>(simd: S) {
    let a = i64x8::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let b = i64x8::from_slice(simd, &[9, 10, 11, 12, 13, 14, 15, 16]);
    assert_eq!(*a.slide::<1>(b), [2, 3, 4, 5, 6, 7, 8, 9]);
}

#[simd_test]
fn slide_u64x2<S: Simd>(simd: S) {
    let a = u64x2::from_slice(simd, &[1, 2]);
    let b = u64x2::from_slice(simd, &[3, 4]);
    assert_eq!(*a.slide::<1>(b), [2, 3]);
}

#[simd_test]
fn slide_u64x4<S: Simd>(simd: S) {
    let a = u64x4::from_slice(simd, &[1, 2, 3, 4]);
    let b = u64x4::from_slice(simd, &[5, 6, 7, 8]);
    assert_eq!(*a.slide::<1>(b), [2, 3, 4, 5]);
}

#[simd_test]
fn slide_u64x8<S: Simd>(simd: S) {
    let a = u64x8::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let b = u64x8::from_slice(simd, &[9, 10, 11, 12, 13, 14, 15, 16]);
    assert_eq!(*a.slide::<1>(b), [2, 3, 4, 5, 6, 7, 8, 9]);
}

#[simd_test]
fn i64_split_combine<S: Simd>(simd: S) {
    let lo = i64x2::from_slice(simd, &[1, 2]);
    let hi = i64x2::from_slice(simd, &[3, 4]);
    let combined = lo.combine(hi);
    assert_eq!(*combined, [1, 2, 3, 4]);

    let (lo, hi) = combined.split();
    assert_eq!(*lo, [1, 2]);
    assert_eq!(*hi, [3, 4]);

    let tail = i64x4::from_slice(simd, &[5, 6, 7, 8]);
    let wide = combined.combine(tail);
    assert_eq!(*wide, [1, 2, 3, 4, 5, 6, 7, 8]);

    let (lo, hi) = wide.split();
    assert_eq!(*lo, [1, 2, 3, 4]);
    assert_eq!(*hi, [5, 6, 7, 8]);
}

#[simd_test]
fn u64_split_combine<S: Simd>(simd: S) {
    let lo = u64x2::from_slice(simd, &[1, 2]);
    let hi = u64x2::from_slice(simd, &[3, 4]);
    let combined = lo.combine(hi);
    assert_eq!(*combined, [1, 2, 3, 4]);

    let (lo, hi) = combined.split();
    assert_eq!(*lo, [1, 2]);
    assert_eq!(*hi, [3, 4]);

    let tail = u64x4::from_slice(simd, &[5, 6, 7, 8]);
    let wide = combined.combine(tail);
    assert_eq!(*wide, [1, 2, 3, 4, 5, 6, 7, 8]);

    let (lo, hi) = wide.split();
    assert_eq!(*lo, [1, 2, 3, 4]);
    assert_eq!(*hi, [5, 6, 7, 8]);
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

#[simd_test]
fn array_methods_i64x2<S: Simd>(simd: S) {
    let a = simd.load_array_i64x2([1, 2]);
    assert_eq!(simd.as_array_i64x2(a), [1, 2]);

    let b_vals = [3, 4];
    let mut b = simd.load_array_ref_i64x2(&b_vals);
    assert_eq!(simd.as_array_ref_i64x2(&b), &[3, 4]);

    simd.as_array_mut_i64x2(&mut b)[1] = 9;
    assert_eq!(*b, [3, 9]);

    let mut dest = [0_i64; 2];
    simd.store_array_i64x2(b, &mut dest);
    assert_eq!(dest, [3, 9]);
}

#[simd_test]
fn array_methods_i64x4<S: Simd>(simd: S) {
    let a = simd.load_array_i64x4([1, 2, 3, 4]);
    assert_eq!(simd.as_array_i64x4(a), [1, 2, 3, 4]);

    let b_vals = [5, 6, 7, 8];
    let mut b = simd.load_array_ref_i64x4(&b_vals);
    assert_eq!(simd.as_array_ref_i64x4(&b), &[5, 6, 7, 8]);

    simd.as_array_mut_i64x4(&mut b)[2] = 99;
    assert_eq!(*b, [5, 6, 99, 8]);

    let mut dest = [0_i64; 4];
    simd.store_array_i64x4(b, &mut dest);
    assert_eq!(dest, [5, 6, 99, 8]);
}

#[simd_test]
fn array_methods_i64x8<S: Simd>(simd: S) {
    let a = simd.load_array_i64x8([1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(simd.as_array_i64x8(a), [1, 2, 3, 4, 5, 6, 7, 8]);

    let b_vals = [9, 10, 11, 12, 13, 14, 15, 16];
    let mut b = simd.load_array_ref_i64x8(&b_vals);
    assert_eq!(
        simd.as_array_ref_i64x8(&b),
        &[9, 10, 11, 12, 13, 14, 15, 16]
    );

    simd.as_array_mut_i64x8(&mut b)[4] = 99;
    assert_eq!(*b, [9, 10, 11, 12, 99, 14, 15, 16]);

    let mut dest = [0_i64; 8];
    simd.store_array_i64x8(b, &mut dest);
    assert_eq!(dest, [9, 10, 11, 12, 99, 14, 15, 16]);
}

#[simd_test]
fn array_methods_u64x2<S: Simd>(simd: S) {
    let a = simd.load_array_u64x2([1, 2]);
    assert_eq!(simd.as_array_u64x2(a), [1, 2]);

    let b_vals = [3, 4];
    let mut b = simd.load_array_ref_u64x2(&b_vals);
    assert_eq!(simd.as_array_ref_u64x2(&b), &[3, 4]);

    simd.as_array_mut_u64x2(&mut b)[1] = 9;
    assert_eq!(*b, [3, 9]);

    let mut dest = [0_u64; 2];
    simd.store_array_u64x2(b, &mut dest);
    assert_eq!(dest, [3, 9]);
}

#[simd_test]
fn array_methods_u64x4<S: Simd>(simd: S) {
    let a = simd.load_array_u64x4([1, 2, 3, 4]);
    assert_eq!(simd.as_array_u64x4(a), [1, 2, 3, 4]);

    let b_vals = [5, 6, 7, 8];
    let mut b = simd.load_array_ref_u64x4(&b_vals);
    assert_eq!(simd.as_array_ref_u64x4(&b), &[5, 6, 7, 8]);

    simd.as_array_mut_u64x4(&mut b)[2] = 99;
    assert_eq!(*b, [5, 6, 99, 8]);

    let mut dest = [0_u64; 4];
    simd.store_array_u64x4(b, &mut dest);
    assert_eq!(dest, [5, 6, 99, 8]);
}

#[simd_test]
fn array_methods_u64x8<S: Simd>(simd: S) {
    let a = simd.load_array_u64x8([1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(simd.as_array_u64x8(a), [1, 2, 3, 4, 5, 6, 7, 8]);

    let b_vals = [9, 10, 11, 12, 13, 14, 15, 16];
    let mut b = simd.load_array_ref_u64x8(&b_vals);
    assert_eq!(
        simd.as_array_ref_u64x8(&b),
        &[9, 10, 11, 12, 13, 14, 15, 16]
    );

    simd.as_array_mut_u64x8(&mut b)[4] = 99;
    assert_eq!(*b, [9, 10, 11, 12, 99, 14, 15, 16]);

    let mut dest = [0_u64; 8];
    simd.store_array_u64x8(b, &mut dest);
    assert_eq!(dest, [9, 10, 11, 12, 99, 14, 15, 16]);
}

#[simd_test]
fn neg_i64x2<S: Simd>(simd: S) {
    let a = i64x2::from_slice(simd, &[-1, 2]);
    assert_eq!(*(-a), [1, -2]);
}

#[simd_test]
fn neg_i64x4<S: Simd>(simd: S) {
    let a = i64x4::from_slice(simd, &[-1, 2, -3, 4]);
    assert_eq!(*(-a), [1, -2, 3, -4]);
}

#[simd_test]
fn slide_within_blocks_i64x2<S: Simd>(simd: S) {
    let a = i64x2::from_slice(simd, &[1, 2]);
    let b = i64x2::from_slice(simd, &[3, 4]);
    assert_eq!(*a.slide_within_blocks::<0>(b), [1, 2]);
    assert_eq!(*a.slide_within_blocks::<1>(b), [2, 3]);
    assert_eq!(*a.slide_within_blocks::<2>(b), [3, 4]);
}

#[simd_test]
fn slide_within_blocks_i64x4<S: Simd>(simd: S) {
    let a = i64x4::from_slice(simd, &[1, 2, 3, 4]);
    let b = i64x4::from_slice(simd, &[5, 6, 7, 8]);
    assert_eq!(*a.slide_within_blocks::<0>(b), [1, 2, 3, 4]);
    assert_eq!(*a.slide_within_blocks::<1>(b), [2, 5, 4, 7]);
    assert_eq!(*a.slide_within_blocks::<2>(b), [5, 6, 7, 8]);
}

#[simd_test]
fn slide_within_blocks_i64x8<S: Simd>(simd: S) {
    let a = i64x8::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let b = i64x8::from_slice(simd, &[9, 10, 11, 12, 13, 14, 15, 16]);
    assert_eq!(*a.slide_within_blocks::<0>(b), [1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(*a.slide_within_blocks::<1>(b), [2, 9, 4, 11, 6, 13, 8, 15]);
    assert_eq!(
        *a.slide_within_blocks::<2>(b),
        [9, 10, 11, 12, 13, 14, 15, 16]
    );
}

#[simd_test]
fn slide_within_blocks_u64x2<S: Simd>(simd: S) {
    let a = u64x2::from_slice(simd, &[1, 2]);
    let b = u64x2::from_slice(simd, &[3, 4]);
    assert_eq!(*a.slide_within_blocks::<0>(b), [1, 2]);
    assert_eq!(*a.slide_within_blocks::<1>(b), [2, 3]);
    assert_eq!(*a.slide_within_blocks::<2>(b), [3, 4]);
}

#[simd_test]
fn slide_within_blocks_u64x4<S: Simd>(simd: S) {
    let a = u64x4::from_slice(simd, &[1, 2, 3, 4]);
    let b = u64x4::from_slice(simd, &[5, 6, 7, 8]);
    assert_eq!(*a.slide_within_blocks::<0>(b), [1, 2, 3, 4]);
    assert_eq!(*a.slide_within_blocks::<1>(b), [2, 5, 4, 7]);
    assert_eq!(*a.slide_within_blocks::<2>(b), [5, 6, 7, 8]);
}

#[simd_test]
fn slide_within_blocks_u64x8<S: Simd>(simd: S) {
    let a = u64x8::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let b = u64x8::from_slice(simd, &[9, 10, 11, 12, 13, 14, 15, 16]);
    assert_eq!(*a.slide_within_blocks::<0>(b), [1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(*a.slide_within_blocks::<1>(b), [2, 9, 4, 11, 6, 13, 8, 15]);
    assert_eq!(
        *a.slide_within_blocks::<2>(b),
        [9, 10, 11, 12, 13, 14, 15, 16]
    );
}

#[simd_test]
fn zip_unzip_i64x2<S: Simd>(simd: S) {
    let a = i64x2::from_slice(simd, &[1, 2]);
    let b = i64x2::from_slice(simd, &[3, 4]);
    assert_eq!(*simd.zip_low_i64x2(a, b), [1, 3]);
    assert_eq!(*simd.zip_high_i64x2(a, b), [2, 4]);
    assert_eq!(*simd.unzip_low_i64x2(a, b), [1, 3]);
    assert_eq!(*simd.unzip_high_i64x2(a, b), [2, 4]);
}

#[simd_test]
fn zip_unzip_i64x4<S: Simd>(simd: S) {
    let a = i64x4::from_slice(simd, &[1, 2, 3, 4]);
    let b = i64x4::from_slice(simd, &[5, 6, 7, 8]);
    assert_eq!(*simd.zip_low_i64x4(a, b), [1, 5, 2, 6]);
    assert_eq!(*simd.zip_high_i64x4(a, b), [3, 7, 4, 8]);
    assert_eq!(*simd.unzip_low_i64x4(a, b), [1, 3, 5, 7]);
    assert_eq!(*simd.unzip_high_i64x4(a, b), [2, 4, 6, 8]);
}

#[simd_test]
fn zip_unzip_i64x8<S: Simd>(simd: S) {
    let a = i64x8::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let b = i64x8::from_slice(simd, &[9, 10, 11, 12, 13, 14, 15, 16]);
    assert_eq!(*simd.zip_low_i64x8(a, b), [1, 9, 2, 10, 3, 11, 4, 12]);
    assert_eq!(*simd.zip_high_i64x8(a, b), [5, 13, 6, 14, 7, 15, 8, 16]);
    assert_eq!(*simd.unzip_low_i64x8(a, b), [1, 3, 5, 7, 9, 11, 13, 15]);
    assert_eq!(*simd.unzip_high_i64x8(a, b), [2, 4, 6, 8, 10, 12, 14, 16]);
}

#[simd_test]
fn zip_unzip_u64x2<S: Simd>(simd: S) {
    let a = u64x2::from_slice(simd, &[1, 2]);
    let b = u64x2::from_slice(simd, &[3, 4]);
    assert_eq!(*simd.zip_low_u64x2(a, b), [1, 3]);
    assert_eq!(*simd.zip_high_u64x2(a, b), [2, 4]);
    assert_eq!(*simd.unzip_low_u64x2(a, b), [1, 3]);
    assert_eq!(*simd.unzip_high_u64x2(a, b), [2, 4]);
}

#[simd_test]
fn zip_unzip_u64x4<S: Simd>(simd: S) {
    let a = u64x4::from_slice(simd, &[1, 2, 3, 4]);
    let b = u64x4::from_slice(simd, &[5, 6, 7, 8]);
    assert_eq!(*simd.zip_low_u64x4(a, b), [1, 5, 2, 6]);
    assert_eq!(*simd.zip_high_u64x4(a, b), [3, 7, 4, 8]);
    assert_eq!(*simd.unzip_low_u64x4(a, b), [1, 3, 5, 7]);
    assert_eq!(*simd.unzip_high_u64x4(a, b), [2, 4, 6, 8]);
}

#[simd_test]
fn zip_unzip_u64x8<S: Simd>(simd: S) {
    let a = u64x8::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let b = u64x8::from_slice(simd, &[9, 10, 11, 12, 13, 14, 15, 16]);
    assert_eq!(*simd.zip_low_u64x8(a, b), [1, 9, 2, 10, 3, 11, 4, 12]);
    assert_eq!(*simd.zip_high_u64x8(a, b), [5, 13, 6, 14, 7, 15, 8, 16]);
    assert_eq!(*simd.unzip_low_u64x8(a, b), [1, 3, 5, 7, 9, 11, 13, 15]);
    assert_eq!(*simd.unzip_high_u64x8(a, b), [2, 4, 6, 8, 10, 12, 14, 16]);
}

#[simd_test]
fn interleave_deinterleave_i64x2<S: Simd>(simd: S) {
    let a = i64x2::from_slice(simd, &[1, 2]);
    let b = i64x2::from_slice(simd, &[3, 4]);
    let (lo, hi) = simd.interleave_i64x2(a, b);
    assert_eq!(*lo, [1, 3]);
    assert_eq!(*hi, [2, 4]);
    let (a_roundtrip, b_roundtrip) = simd.deinterleave_i64x2(lo, hi);
    assert_eq!(*a_roundtrip, [1, 2]);
    assert_eq!(*b_roundtrip, [3, 4]);
}

#[simd_test]
fn interleave_deinterleave_i64x4<S: Simd>(simd: S) {
    let a = i64x4::from_slice(simd, &[1, 2, 3, 4]);
    let b = i64x4::from_slice(simd, &[5, 6, 7, 8]);
    let (lo, hi) = simd.interleave_i64x4(a, b);
    assert_eq!(*lo, [1, 5, 2, 6]);
    assert_eq!(*hi, [3, 7, 4, 8]);
    let (a_roundtrip, b_roundtrip) = simd.deinterleave_i64x4(lo, hi);
    assert_eq!(*a_roundtrip, [1, 2, 3, 4]);
    assert_eq!(*b_roundtrip, [5, 6, 7, 8]);

    let (lo, hi) = a.interleave(b);
    let (a_roundtrip, b_roundtrip) = lo.deinterleave(hi);
    assert_eq!(*a_roundtrip, [1, 2, 3, 4]);
    assert_eq!(*b_roundtrip, [5, 6, 7, 8]);
}

#[simd_test]
fn interleave_deinterleave_i64x8<S: Simd>(simd: S) {
    let a = i64x8::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let b = i64x8::from_slice(simd, &[9, 10, 11, 12, 13, 14, 15, 16]);
    let (lo, hi) = simd.interleave_i64x8(a, b);
    assert_eq!(*lo, [1, 9, 2, 10, 3, 11, 4, 12]);
    assert_eq!(*hi, [5, 13, 6, 14, 7, 15, 8, 16]);
    let (a_roundtrip, b_roundtrip) = simd.deinterleave_i64x8(lo, hi);
    assert_eq!(*a_roundtrip, [1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(*b_roundtrip, [9, 10, 11, 12, 13, 14, 15, 16]);
}

#[simd_test]
fn interleave_deinterleave_u64x2<S: Simd>(simd: S) {
    let a = u64x2::from_slice(simd, &[1, 2]);
    let b = u64x2::from_slice(simd, &[3, 4]);
    let (lo, hi) = simd.interleave_u64x2(a, b);
    assert_eq!(*lo, [1, 3]);
    assert_eq!(*hi, [2, 4]);
    let (a_roundtrip, b_roundtrip) = simd.deinterleave_u64x2(lo, hi);
    assert_eq!(*a_roundtrip, [1, 2]);
    assert_eq!(*b_roundtrip, [3, 4]);
}

#[simd_test]
fn interleave_deinterleave_u64x4<S: Simd>(simd: S) {
    let a = u64x4::from_slice(simd, &[1, 2, 3, 4]);
    let b = u64x4::from_slice(simd, &[5, 6, 7, 8]);
    let (lo, hi) = simd.interleave_u64x4(a, b);
    assert_eq!(*lo, [1, 5, 2, 6]);
    assert_eq!(*hi, [3, 7, 4, 8]);
    let (a_roundtrip, b_roundtrip) = simd.deinterleave_u64x4(lo, hi);
    assert_eq!(*a_roundtrip, [1, 2, 3, 4]);
    assert_eq!(*b_roundtrip, [5, 6, 7, 8]);

    let (lo, hi) = a.interleave(b);
    let (a_roundtrip, b_roundtrip) = lo.deinterleave(hi);
    assert_eq!(*a_roundtrip, [1, 2, 3, 4]);
    assert_eq!(*b_roundtrip, [5, 6, 7, 8]);
}

#[simd_test]
fn interleave_deinterleave_u64x8<S: Simd>(simd: S) {
    let a = u64x8::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let b = u64x8::from_slice(simd, &[9, 10, 11, 12, 13, 14, 15, 16]);
    let (lo, hi) = simd.interleave_u64x8(a, b);
    assert_eq!(*lo, [1, 9, 2, 10, 3, 11, 4, 12]);
    assert_eq!(*hi, [5, 13, 6, 14, 7, 15, 8, 16]);
    let (a_roundtrip, b_roundtrip) = simd.deinterleave_u64x8(lo, hi);
    assert_eq!(*a_roundtrip, [1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(*b_roundtrip, [9, 10, 11, 12, 13, 14, 15, 16]);
}

#[simd_test]
fn load_store_interleaved_128_u64x8<S: Simd>(simd: S) {
    let data = [1, 2, 101, 102, 201, 202, 301, 302];
    let loaded = simd.load_interleaved_128_u64x8(&data);
    assert_eq!(*loaded, [1, 201, 2, 202, 101, 301, 102, 302]);

    let a = u64x8::from_slice(simd, &[1, 201, 2, 202, 101, 301, 102, 302]);
    let mut dest = [0_u64; 8];
    simd.store_interleaved_128_u64x8(a, &mut dest);
    assert_eq!(dest, data);
}

#[simd_test]
fn reinterpret_i64x2<S: Simd>(simd: S) {
    let a = i64x2::from_slice(simd, &[1, -2]);
    let bytes: u8x16<S> = a.bitcast();
    let words: u32x4<S> = a.bitcast();
    assert_eq!(simd.reinterpret_u8_i64x2(a).as_slice(), bytes.as_slice());
    assert_eq!(simd.reinterpret_u32_i64x2(a).as_slice(), words.as_slice());
}

#[simd_test]
fn reinterpret_i64x4<S: Simd>(simd: S) {
    let a = i64x4::from_slice(simd, &[1, -2, 3, -4]);
    let bytes: u8x32<S> = a.bitcast();
    let words: u32x8<S> = a.bitcast();
    assert_eq!(simd.reinterpret_u8_i64x4(a).as_slice(), bytes.as_slice());
    assert_eq!(simd.reinterpret_u32_i64x4(a).as_slice(), words.as_slice());
}

#[simd_test]
fn reinterpret_i64x8<S: Simd>(simd: S) {
    let a = i64x8::from_slice(simd, &[1, -2, 3, -4, 5, -6, 7, -8]);
    let bytes: u8x64<S> = a.bitcast();
    let words: u32x16<S> = a.bitcast();
    assert_eq!(simd.reinterpret_u8_i64x8(a).as_slice(), bytes.as_slice());
    assert_eq!(simd.reinterpret_u32_i64x8(a).as_slice(), words.as_slice());
}

#[simd_test]
fn reinterpret_u64x2<S: Simd>(simd: S) {
    let a = u64x2::from_slice(simd, &[1, u64::MAX - 1]);
    let bytes: u8x16<S> = a.bitcast();
    let words: u32x4<S> = a.bitcast();
    assert_eq!(simd.reinterpret_u8_u64x2(a).as_slice(), bytes.as_slice());
    assert_eq!(simd.reinterpret_u32_u64x2(a).as_slice(), words.as_slice());
}

#[simd_test]
fn reinterpret_u64x4<S: Simd>(simd: S) {
    let a = u64x4::from_slice(simd, &[1, u64::MAX - 1, 3, u64::MAX - 3]);
    let bytes: u8x32<S> = a.bitcast();
    let words: u32x8<S> = a.bitcast();
    assert_eq!(simd.reinterpret_u8_u64x4(a).as_slice(), bytes.as_slice());
    assert_eq!(simd.reinterpret_u32_u64x4(a).as_slice(), words.as_slice());
}

#[simd_test]
fn reinterpret_u64x8<S: Simd>(simd: S) {
    let a = u64x8::from_slice(simd, &[1, u64::MAX - 1, 3, u64::MAX - 3, 5, 6, 7, 8]);
    let bytes: u8x64<S> = a.bitcast();
    let words: u32x16<S> = a.bitcast();
    assert_eq!(simd.reinterpret_u8_u64x8(a).as_slice(), bytes.as_slice());
    assert_eq!(simd.reinterpret_u32_u64x8(a).as_slice(), words.as_slice());
}

#[simd_test]
fn mask64x2_ops<S: Simd>(simd: S) {
    let t = simd.splat_mask64x2(true);
    let f = simd.splat_mask64x2(false);
    assert_eq!(simd.as_array_mask64x2(t), [-1, -1]);
    assert_eq!(simd.as_array_mask64x2(f), [0, 0]);

    let a = simd.load_array_mask64x2([-1, 0]);
    let b = simd.load_array_mask64x2([0, -1]);
    assert_eq!(simd.as_array_mask64x2(a), [-1, 0]);
    assert_eq!(simd.as_array_mask64x2(simd.and_mask64x2(a, b)), [0, 0]);
    assert_eq!(simd.as_array_mask64x2(simd.or_mask64x2(a, b)), [-1, -1]);
    assert_eq!(simd.as_array_mask64x2(simd.xor_mask64x2(a, b)), [-1, -1]);
    assert_eq!(simd.as_array_mask64x2(simd.not_mask64x2(a)), [0, -1]);
    assert_eq!(
        simd.as_array_mask64x2(simd.select_mask64x2(a, t, f)),
        [-1, 0]
    );
    assert_eq!(
        simd.as_array_mask64x2(simd.simd_eq_mask64x2(a, a)),
        [-1, -1]
    );
    assert_eq!(simd.as_array_mask64x2(simd.simd_eq_mask64x2(a, b)), [0, 0]);

    let mut bitmask = simd.from_bitmask_mask64x2(0b01);
    assert_eq!(simd.as_array_mask64x2(bitmask), [-1, 0]);
    assert_eq!(simd.to_bitmask_mask64x2(bitmask), 0b01);
    simd.set_mask64x2(&mut bitmask, 1, true);
    assert_eq!(simd.to_bitmask_mask64x2(bitmask), 0b11);

    assert!(simd.any_true_mask64x2(a));
    assert!(!simd.all_true_mask64x2(a));
    assert!(simd.any_false_mask64x2(a));
    assert!(!simd.all_false_mask64x2(a));
    assert!(simd.all_true_mask64x2(t));
    assert!(simd.all_false_mask64x2(f));
}

#[simd_test]
fn mask64x4_ops<S: Simd>(simd: S) {
    let t = simd.splat_mask64x4(true);
    let f = simd.splat_mask64x4(false);
    assert_eq!(simd.as_array_mask64x4(t), [-1, -1, -1, -1]);
    assert_eq!(simd.as_array_mask64x4(f), [0, 0, 0, 0]);

    let a = simd.load_array_mask64x4([-1, 0, -1, 0]);
    let b = simd.load_array_mask64x4([0, -1, -1, 0]);
    assert_eq!(
        simd.as_array_mask64x4(simd.and_mask64x4(a, b)),
        [0, 0, -1, 0]
    );
    assert_eq!(
        simd.as_array_mask64x4(simd.or_mask64x4(a, b)),
        [-1, -1, -1, 0]
    );
    assert_eq!(
        simd.as_array_mask64x4(simd.xor_mask64x4(a, b)),
        [-1, -1, 0, 0]
    );
    assert_eq!(simd.as_array_mask64x4(simd.not_mask64x4(a)), [0, -1, 0, -1]);
    assert_eq!(
        simd.as_array_mask64x4(simd.select_mask64x4(a, t, f)),
        [-1, 0, -1, 0]
    );
    assert_eq!(
        simd.as_array_mask64x4(simd.simd_eq_mask64x4(a, b)),
        [0, 0, -1, -1]
    );

    let mut bitmask = simd.from_bitmask_mask64x4(0b1010);
    assert_eq!(simd.as_array_mask64x4(bitmask), [0, -1, 0, -1]);
    assert_eq!(simd.to_bitmask_mask64x4(bitmask), 0b1010);
    simd.set_mask64x4(&mut bitmask, 0, true);
    assert_eq!(simd.to_bitmask_mask64x4(bitmask), 0b1011);

    let combined = simd.combine_mask64x2(
        simd.load_array_mask64x2([-1, 0]),
        simd.load_array_mask64x2([0, -1]),
    );
    assert_eq!(simd.as_array_mask64x4(combined), [-1, 0, 0, -1]);
    let (lo, hi) = simd.split_mask64x4(combined);
    assert_eq!(simd.as_array_mask64x2(lo), [-1, 0]);
    assert_eq!(simd.as_array_mask64x2(hi), [0, -1]);

    assert!(simd.any_true_mask64x4(a));
    assert!(!simd.all_true_mask64x4(a));
    assert!(simd.any_false_mask64x4(a));
    assert!(!simd.all_false_mask64x4(a));
    assert!(simd.all_true_mask64x4(t));
    assert!(simd.all_false_mask64x4(f));
}

#[simd_test]
fn mask64x8_ops<S: Simd>(simd: S) {
    let t = simd.splat_mask64x8(true);
    let f = simd.splat_mask64x8(false);
    assert_eq!(simd.as_array_mask64x8(t), [-1, -1, -1, -1, -1, -1, -1, -1]);
    assert_eq!(simd.as_array_mask64x8(f), [0, 0, 0, 0, 0, 0, 0, 0]);

    let a = simd.load_array_mask64x8([-1, 0, -1, 0, -1, 0, -1, 0]);
    let b = simd.load_array_mask64x8([0, -1, -1, 0, 0, -1, -1, 0]);
    assert_eq!(
        simd.as_array_mask64x8(simd.and_mask64x8(a, b)),
        [0, 0, -1, 0, 0, 0, -1, 0]
    );
    assert_eq!(
        simd.as_array_mask64x8(simd.or_mask64x8(a, b)),
        [-1, -1, -1, 0, -1, -1, -1, 0]
    );
    assert_eq!(
        simd.as_array_mask64x8(simd.xor_mask64x8(a, b)),
        [-1, -1, 0, 0, -1, -1, 0, 0]
    );
    assert_eq!(
        simd.as_array_mask64x8(simd.not_mask64x8(a)),
        [0, -1, 0, -1, 0, -1, 0, -1]
    );
    assert_eq!(
        simd.as_array_mask64x8(simd.select_mask64x8(a, t, f)),
        [-1, 0, -1, 0, -1, 0, -1, 0]
    );
    assert_eq!(
        simd.as_array_mask64x8(simd.simd_eq_mask64x8(a, b)),
        [0, 0, -1, -1, 0, 0, -1, -1]
    );

    let mut bitmask = simd.from_bitmask_mask64x8(0b1010_0101);
    assert_eq!(
        simd.as_array_mask64x8(bitmask),
        [-1, 0, -1, 0, 0, -1, 0, -1]
    );
    assert_eq!(simd.to_bitmask_mask64x8(bitmask), 0b1010_0101);
    simd.set_mask64x8(&mut bitmask, 1, true);
    assert_eq!(simd.to_bitmask_mask64x8(bitmask), 0b1010_0111);

    let combined = simd.combine_mask64x4(
        simd.load_array_mask64x4([-1, 0, -1, 0]),
        simd.load_array_mask64x4([0, -1, 0, -1]),
    );
    assert_eq!(
        simd.as_array_mask64x8(combined),
        [-1, 0, -1, 0, 0, -1, 0, -1]
    );
    let (lo, hi) = simd.split_mask64x8(combined);
    assert_eq!(simd.as_array_mask64x4(lo), [-1, 0, -1, 0]);
    assert_eq!(simd.as_array_mask64x4(hi), [0, -1, 0, -1]);

    assert!(simd.any_true_mask64x8(a));
    assert!(!simd.all_true_mask64x8(a));
    assert!(simd.any_false_mask64x8(a));
    assert!(!simd.all_false_mask64x8(a));
    assert!(simd.all_true_mask64x8(t));
    assert!(simd.all_false_mask64x8(f));
}
