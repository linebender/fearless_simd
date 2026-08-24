// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// One concrete test row per supported integer vector type.

#[simd_test]
fn reduce_and_i8x16<S: Simd>(simd: S) {
    let value = i8x16::simd_from(
        simd,
        [
            -5, 127, -17, -3, -65, -9, -2, -33, 123, 127, -17, -3, -65, -9, -2, -49,
        ],
    );
    assert_eq!(value.reduce_and(), 0);
}

#[simd_test]
fn reduce_and_u8x16<S: Simd>(simd: S) {
    let value = u8x16::simd_from(
        simd,
        [
            251, 127, 239, 253, 191, 247, 254, 223, 123, 127, 239, 253, 191, 247, 254, 207,
        ],
    );
    assert_eq!(value.reduce_and(), 0);
}

#[simd_test]
fn reduce_and_i8x32<S: Simd>(simd: S) {
    let value = i8x32::simd_from(
        simd,
        [
            -5, 127, -17, -3, -65, -9, -2, -33, -5, 127, -17, -3, -65, -9, -2, -33, 123, 127, -17,
            -3, -65, -9, -2, -33, -5, 127, -17, -3, -65, -9, -2, -49,
        ],
    );
    assert_eq!(value.reduce_and(), 0);
}

#[simd_test]
fn reduce_and_u8x32<S: Simd>(simd: S) {
    let value = u8x32::simd_from(
        simd,
        [
            251, 127, 239, 253, 191, 247, 254, 223, 251, 127, 239, 253, 191, 247, 254, 223, 123,
            127, 239, 253, 191, 247, 254, 223, 251, 127, 239, 253, 191, 247, 254, 207,
        ],
    );
    assert_eq!(value.reduce_and(), 0);
}

#[simd_test]
fn reduce_and_i8x64<S: Simd>(simd: S) {
    let value = i8x64::simd_from(
        simd,
        [
            -5, 127, -17, -3, -65, -9, -2, -33, -5, 127, -17, -3, -65, -9, -2, -33, -5, 127, -17,
            -3, -65, -9, -2, -33, -5, 127, -17, -3, -65, -9, -2, -33, 123, 127, -17, -3, -65, -9,
            -2, -33, -5, 127, -17, -3, -65, -9, -2, -33, -5, 127, -17, -3, -65, -9, -2, -33, -5,
            127, -17, -3, -65, -9, -2, -49,
        ],
    );
    assert_eq!(value.reduce_and(), 0);
}

#[simd_test]
fn reduce_and_u8x64<S: Simd>(simd: S) {
    let value = u8x64::simd_from(
        simd,
        [
            251, 127, 239, 253, 191, 247, 254, 223, 251, 127, 239, 253, 191, 247, 254, 223, 251,
            127, 239, 253, 191, 247, 254, 223, 251, 127, 239, 253, 191, 247, 254, 223, 123, 127,
            239, 253, 191, 247, 254, 223, 251, 127, 239, 253, 191, 247, 254, 223, 251, 127, 239,
            253, 191, 247, 254, 223, 251, 127, 239, 253, 191, 247, 254, 207,
        ],
    );
    assert_eq!(value.reduce_and(), 0);
}

#[simd_test]
fn reduce_and_i16x8<S: Simd>(simd: S) {
    let value = i16x8::simd_from(simd, [-5, -129, -4097, -3, 32703, -2049, -2, -289]);
    assert_eq!(value.reduce_and(), 26136);
}

#[simd_test]
fn reduce_and_u16x8<S: Simd>(simd: S) {
    let value = u16x8::simd_from(
        simd,
        [65531, 65407, 61439, 65533, 32703, 63487, 65534, 65247],
    );
    assert_eq!(value.reduce_and(), 26136);
}

#[simd_test]
fn reduce_and_i16x16<S: Simd>(simd: S) {
    let value = i16x16::simd_from(
        simd,
        [
            -5, -129, -4097, -3, -65, -2049, -2, -33, 31743, 32767, -17, -513, -16385, -9, -257,
            -8449,
        ],
    );
    assert_eq!(value.reduce_and(), 0);
}

#[simd_test]
fn reduce_and_u16x16<S: Simd>(simd: S) {
    let value = u16x16::simd_from(
        simd,
        [
            65531, 65407, 61439, 65533, 65471, 63487, 65534, 65503, 31743, 32767, 65519, 65023,
            49151, 65527, 65279, 57087,
        ],
    );
    assert_eq!(value.reduce_and(), 0);
}

#[simd_test]
fn reduce_and_i16x32<S: Simd>(simd: S) {
    let value = i16x32::simd_from(
        simd,
        [
            -5, -129, -4097, -3, -65, -2049, -2, -33, -1025, 32767, -17, -513, -16385, -9, -257,
            -8193, 32763, -129, -4097, -3, -65, -2049, -2, -33, -1025, 32767, -17, -513, -16385,
            -9, -257, -8449,
        ],
    );
    assert_eq!(value.reduce_and(), 0);
}

#[simd_test]
fn reduce_and_u16x32<S: Simd>(simd: S) {
    let value = u16x32::simd_from(
        simd,
        [
            65531, 65407, 61439, 65533, 65471, 63487, 65534, 65503, 64511, 32767, 65519, 65023,
            49151, 65527, 65279, 57343, 32763, 65407, 61439, 65533, 65471, 63487, 65534, 65503,
            64511, 32767, 65519, 65023, 49151, 65527, 65279, 57087,
        ],
    );
    assert_eq!(value.reduce_and(), 0);
}

#[simd_test]
fn reduce_and_i32x4<S: Simd>(simd: S) {
    let value = i32x4::simd_from(simd, [-5, -129, 2147479551, -196609]);
    assert_eq!(value.reduce_and(), 2147282811);
}

#[simd_test]
fn reduce_and_u32x4<S: Simd>(simd: S) {
    let value = u32x4::simd_from(simd, [4294967291, 4294967167, 2147479551, 4294770687]);
    assert_eq!(value.reduce_and(), 2147282811);
}

#[simd_test]
fn reduce_and_i32x8<S: Simd>(simd: S) {
    let value = i32x8::simd_from(
        simd,
        [-5, -129, -4097, -131073, 2143289343, -134217729, -2, -65569],
    );
    assert_eq!(value.reduce_and(), 2008870746);
}

#[simd_test]
fn reduce_and_u32x8<S: Simd>(simd: S) {
    let value = u32x8::simd_from(
        simd,
        [
            4294967291, 4294967167, 4294963199, 4294836223, 2143289343, 4160749567, 4294967294,
            4294901727,
        ],
    );
    assert_eq!(value.reduce_and(), 2008870746);
}

#[simd_test]
fn reduce_and_i32x16<S: Simd>(simd: S) {
    let value = i32x16::simd_from(
        simd,
        [
            -5,
            -129,
            -4097,
            -131073,
            -4194305,
            -134217729,
            -2,
            -33,
            2147482623,
            -32769,
            -1048577,
            -33554433,
            -1073741825,
            -9,
            -257,
            -73729,
        ],
    );
    assert_eq!(value.reduce_and(), 900483666);
}

#[simd_test]
fn reduce_and_u32x16<S: Simd>(simd: S) {
    let value = u32x16::simd_from(
        simd,
        [
            4294967291, 4294967167, 4294963199, 4294836223, 4290772991, 4160749567, 4294967294,
            4294967263, 2147482623, 4294934527, 4293918719, 4261412863, 3221225471, 4294967287,
            4294967039, 4294893567,
        ],
    );
    assert_eq!(value.reduce_and(), 900483666);
}

#[simd_test]
fn reduce_and_i64x2<S: Simd>(simd: S) {
    let value = i64x2::simd_from(simd, [-5, 9223372032559808383]);
    assert_eq!(value.reduce_and(), 9223372032559808379);
}

#[simd_test]
fn reduce_and_u64x2<S: Simd>(simd: S) {
    let value = u64x2::simd_from(simd, [18446744073709551611, 9223372032559808383]);
    assert_eq!(value.reduce_and(), 9223372032559808379);
}

#[simd_test]
fn reduce_and_i64x4<S: Simd>(simd: S) {
    let value = i64x4::simd_from(simd, [-5, -129, 9223372036854771711, -4295098369]);
    assert_eq!(value.reduce_and(), 9223372032559673211);
}

#[simd_test]
fn reduce_and_u64x4<S: Simd>(simd: S) {
    let value = u64x4::simd_from(
        simd,
        [
            18446744073709551611,
            18446744073709551487,
            9223372036854771711,
            18446744069414453247,
        ],
    );
    assert_eq!(value.reduce_and(), 9223372032559673211);
}

#[simd_test]
fn reduce_and_i64x8<S: Simd>(simd: S) {
    let value = i64x8::simd_from(
        simd,
        [
            -5,
            -129,
            -4097,
            -131073,
            9223372036850581503,
            -134217729,
            -4294967297,
            -141733920769,
        ],
    );
    assert_eq!(value.reduce_and(), 9223371894982307707);
}

#[simd_test]
fn reduce_and_u64x8<S: Simd>(simd: S) {
    let value = u64x8::simd_from(
        simd,
        [
            18446744073709551611,
            18446744073709551487,
            18446744073709547519,
            18446744073709420543,
            9223372036850581503,
            18446744073575333887,
            18446744069414584319,
            18446743931975630847,
        ],
    );
    assert_eq!(value.reduce_and(), 9223371894982307707);
}
