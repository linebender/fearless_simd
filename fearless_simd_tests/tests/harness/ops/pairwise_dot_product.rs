// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

const I8_MIN: i8 = i8::MIN;
const I8_MAX: i8 = i8::MAX;
const I16_MIN: i16 = i16::MIN;
const I16_MAX: i16 = i16::MAX;
const I32_MIN: i32 = i32::MIN;
const I32_MAX: i32 = i32::MAX;
const U8_MAX: u8 = u8::MAX;
const U8_MAX_M1: u8 = U8_MAX - 1;
const U16_MAX: u16 = u16::MAX;
const U16_MAX_M1: u16 = U16_MAX - 1;
const U32_MAX: u32 = u32::MAX;
const U32_MAX_M1: u32 = U32_MAX - 1;

#[simd_test]
fn pairwise_dot_product_i8x16<S: Simd>(simd: S) {
    let cases: [([i8; 16], [i8; 16]); 2] = [
        (
            [
                I8_MIN, I8_MIN, I8_MAX, -I8_MAX, 4, -5, 6, -7, 8, -9, 10, -11, 12, -13, 14, -15,
            ],
            [
                I8_MIN, I8_MIN, I8_MAX, I8_MAX, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
            ],
        ),
        (
            [0, 1, -1, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            [
                I8_MAX, I8_MIN, I8_MAX, I8_MAX, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14,
            ],
        ),
    ];
    for (a, b) in cases {
        let lhs = i8x16::from_slice(simd, &a);
        let rhs = i8x16::from_slice(simd, &b);
        let result = lhs.pairwise_dot_product(rhs);
        for i in 0..8 {
            let even = (a[2 * i] as i16) * (b[2 * i] as i16);
            let odd = (a[2 * i + 1] as i16) * (b[2 * i + 1] as i16);
            assert_eq!(result[i], even.wrapping_add(odd));
        }
    }
}

#[simd_test]
fn pairwise_dot_product_i8x32<S: Simd>(simd: S) {
    let cases: [([i8; 32], [i8; 32]); 2] = [
        (
            [
                I8_MIN, I8_MIN, I8_MAX, -I8_MAX, 4, -5, 6, -7, 8, -9, 10, -11, 12, -13, 14, -15,
                16, -17, 18, -19, 20, -21, 22, -23, 24, -25, 26, -27, 28, -29, 30, -31,
            ],
            [
                I8_MIN, I8_MIN, I8_MAX, I8_MAX, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33,
                35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63,
            ],
        ),
        (
            [
                0, 1, -1, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25, 26, 27, 28, 29, 30, 31,
            ],
            [
                I8_MAX, I8_MIN, I8_MAX, I8_MAX, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13,
                -14, -15, -16, -17, -18, -19, -20, -21, -22, -23, -24, -25, -26, -27, -28, -29,
                -30,
            ],
        ),
    ];
    for (a, b) in cases {
        let lhs = i8x32::from_slice(simd, &a);
        let rhs = i8x32::from_slice(simd, &b);
        let result = lhs.pairwise_dot_product(rhs);
        for i in 0..16 {
            let even = (a[2 * i] as i16) * (b[2 * i] as i16);
            let odd = (a[2 * i + 1] as i16) * (b[2 * i + 1] as i16);
            assert_eq!(result[i], even.wrapping_add(odd));
        }
    }
}

#[simd_test]
fn pairwise_dot_product_i8x64<S: Simd>(simd: S) {
    let cases: [([i8; 64], [i8; 64]); 2] = [
        (
            [
                I8_MIN, I8_MIN, I8_MAX, -I8_MAX, 4, -5, 6, -7, 8, -9, 10, -11, 12, -13, 14, -15,
                16, -17, 18, -19, 20, -21, 22, -23, 24, -25, 26, -27, 28, -29, 30, -31, 32, -33,
                34, -35, 36, -37, 38, -39, 40, -41, 42, -43, 44, -45, 46, -47, 48, -49, 50, -51,
                52, -53, 54, -55, 56, -57, 58, -59, 60, -61, 62, -63,
            ],
            [
                I8_MIN, I8_MIN, I8_MAX, I8_MAX, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33,
                35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75,
                77, 79, 81, 83, 85, 87, 89, 91, 93, 95, 97, 99, 101, 103, 105, 107, 109, 111, 113,
                115, 117, 119, 121, 123, 125, 127,
            ],
        ),
        (
            [
                0, 1, -1, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
            ],
            [
                I8_MAX, I8_MIN, I8_MAX, I8_MAX, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13,
                -14, -15, -16, -17, -18, -19, -20, -21, -22, -23, -24, -25, -26, -27, -28, -29,
                -30, -31, -32, -33, -34, -35, -36, -37, -38, -39, -40, -41, -42, -43, -44, -45,
                -46, -47, -48, -49, -50, -51, -52, -53, -54, -55, -56, -57, -58, -59, -60, -61,
                -62,
            ],
        ),
    ];
    for (a, b) in cases {
        let lhs = i8x64::from_slice(simd, &a);
        let rhs = i8x64::from_slice(simd, &b);
        let result = lhs.pairwise_dot_product(rhs);
        for i in 0..32 {
            let even = (a[2 * i] as i16) * (b[2 * i] as i16);
            let odd = (a[2 * i + 1] as i16) * (b[2 * i + 1] as i16);
            assert_eq!(result[i], even.wrapping_add(odd));
        }
    }
}

#[simd_test]
fn pairwise_dot_product_i16x8<S: Simd>(simd: S) {
    let cases: [([i16; 8], [i16; 8]); 2] = [
        (
            [I16_MIN, I16_MIN, I16_MAX, -I16_MAX, 4, -5, 6, -7],
            [I16_MIN, I16_MIN, I16_MAX, I16_MAX, 9, 11, 13, 15],
        ),
        (
            [0, 1, -1, 1, 4, 5, 6, 7],
            [I16_MAX, I16_MIN, I16_MAX, I16_MAX, -3, -4, -5, -6],
        ),
    ];
    for (a, b) in cases {
        let lhs = i16x8::from_slice(simd, &a);
        let rhs = i16x8::from_slice(simd, &b);
        let result = lhs.pairwise_dot_product(rhs);
        for i in 0..4 {
            let even = (a[2 * i] as i32) * (b[2 * i] as i32);
            let odd = (a[2 * i + 1] as i32) * (b[2 * i + 1] as i32);
            assert_eq!(result[i], even.wrapping_add(odd));
        }
    }
}

#[simd_test]
fn pairwise_dot_product_i16x16<S: Simd>(simd: S) {
    let cases: [([i16; 16], [i16; 16]); 2] = [
        (
            [
                I16_MIN, I16_MIN, I16_MAX, -I16_MAX, 4, -5, 6, -7, 8, -9, 10, -11, 12, -13, 14, -15,
            ],
            [
                I16_MIN, I16_MIN, I16_MAX, I16_MAX, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
            ],
        ),
        (
            [0, 1, -1, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            [
                I16_MAX, I16_MIN, I16_MAX, I16_MAX, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13,
                -14,
            ],
        ),
    ];
    for (a, b) in cases {
        let lhs = i16x16::from_slice(simd, &a);
        let rhs = i16x16::from_slice(simd, &b);
        let result = lhs.pairwise_dot_product(rhs);
        for i in 0..8 {
            let even = (a[2 * i] as i32) * (b[2 * i] as i32);
            let odd = (a[2 * i + 1] as i32) * (b[2 * i + 1] as i32);
            assert_eq!(result[i], even.wrapping_add(odd));
        }
    }
}

#[simd_test]
fn pairwise_dot_product_i16x32<S: Simd>(simd: S) {
    let cases: [([i16; 32], [i16; 32]); 2] = [
        (
            [
                I16_MIN, I16_MIN, I16_MAX, -I16_MAX, 4, -5, 6, -7, 8, -9, 10, -11, 12, -13, 14,
                -15, 16, -17, 18, -19, 20, -21, 22, -23, 24, -25, 26, -27, 28, -29, 30, -31,
            ],
            [
                I16_MIN, I16_MIN, I16_MAX, I16_MAX, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
                33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63,
            ],
        ),
        (
            [
                0, 1, -1, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25, 26, 27, 28, 29, 30, 31,
            ],
            [
                I16_MAX, I16_MIN, I16_MAX, I16_MAX, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13,
                -14, -15, -16, -17, -18, -19, -20, -21, -22, -23, -24, -25, -26, -27, -28, -29,
                -30,
            ],
        ),
    ];
    for (a, b) in cases {
        let lhs = i16x32::from_slice(simd, &a);
        let rhs = i16x32::from_slice(simd, &b);
        let result = lhs.pairwise_dot_product(rhs);
        for i in 0..16 {
            let even = (a[2 * i] as i32) * (b[2 * i] as i32);
            let odd = (a[2 * i + 1] as i32) * (b[2 * i + 1] as i32);
            assert_eq!(result[i], even.wrapping_add(odd));
        }
    }
}

#[simd_test]
fn pairwise_dot_product_i32x4<S: Simd>(simd: S) {
    let cases: [([i32; 4], [i32; 4]); 2] = [
        (
            [I32_MIN, I32_MIN, I32_MAX, -I32_MAX],
            [I32_MIN, I32_MIN, I32_MAX, I32_MAX],
        ),
        ([0, 1, -1, 1], [I32_MAX, I32_MIN, I32_MAX, I32_MAX]),
    ];
    for (a, b) in cases {
        let lhs = i32x4::from_slice(simd, &a);
        let rhs = i32x4::from_slice(simd, &b);
        let result = lhs.pairwise_dot_product(rhs);
        for i in 0..2 {
            let even = (a[2 * i] as i64) * (b[2 * i] as i64);
            let odd = (a[2 * i + 1] as i64) * (b[2 * i + 1] as i64);
            assert_eq!(result[i], even.wrapping_add(odd));
        }
    }
}

#[simd_test]
fn pairwise_dot_product_i32x8<S: Simd>(simd: S) {
    let cases: [([i32; 8], [i32; 8]); 2] = [
        (
            [I32_MIN, I32_MIN, I32_MAX, -I32_MAX, 4, -5, 6, -7],
            [I32_MIN, I32_MIN, I32_MAX, I32_MAX, 9, 11, 13, 15],
        ),
        (
            [0, 1, -1, 1, 4, 5, 6, 7],
            [I32_MAX, I32_MIN, I32_MAX, I32_MAX, -3, -4, -5, -6],
        ),
    ];
    for (a, b) in cases {
        let lhs = i32x8::from_slice(simd, &a);
        let rhs = i32x8::from_slice(simd, &b);
        let result = lhs.pairwise_dot_product(rhs);
        for i in 0..4 {
            let even = (a[2 * i] as i64) * (b[2 * i] as i64);
            let odd = (a[2 * i + 1] as i64) * (b[2 * i + 1] as i64);
            assert_eq!(result[i], even.wrapping_add(odd));
        }
    }
}

#[simd_test]
fn pairwise_dot_product_i32x16<S: Simd>(simd: S) {
    let cases: [([i32; 16], [i32; 16]); 2] = [
        (
            [
                I32_MIN, I32_MIN, I32_MAX, -I32_MAX, 4, -5, 6, -7, 8, -9, 10, -11, 12, -13, 14, -15,
            ],
            [
                I32_MIN, I32_MIN, I32_MAX, I32_MAX, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
            ],
        ),
        (
            [0, 1, -1, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            [
                I32_MAX, I32_MIN, I32_MAX, I32_MAX, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13,
                -14,
            ],
        ),
    ];
    for (a, b) in cases {
        let lhs = i32x16::from_slice(simd, &a);
        let rhs = i32x16::from_slice(simd, &b);
        let result = lhs.pairwise_dot_product(rhs);
        for i in 0..8 {
            let even = (a[2 * i] as i64) * (b[2 * i] as i64);
            let odd = (a[2 * i + 1] as i64) * (b[2 * i + 1] as i64);
            assert_eq!(result[i], even.wrapping_add(odd));
        }
    }
}

#[simd_test]
fn pairwise_dot_product_u8x16<S: Simd>(simd: S) {
    let cases: [([u8; 16], [u8; 16]); 2] = [
        (
            [
                U8_MAX, U8_MAX, 0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            ],
            [
                U8_MAX, U8_MAX, U8_MAX, U8_MAX, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
            ],
        ),
        (
            [
                1, U8_MAX, 2, U8_MAX_M1, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
            ],
            [U8_MAX, 1, 1, 0, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2],
        ),
    ];
    for (a, b) in cases {
        let lhs = u8x16::from_slice(simd, &a);
        let rhs = u8x16::from_slice(simd, &b);
        let result = lhs.pairwise_dot_product(rhs);
        for i in 0..8 {
            let even = (a[2 * i] as u16) * (b[2 * i] as u16);
            let odd = (a[2 * i + 1] as u16) * (b[2 * i + 1] as u16);
            assert_eq!(result[i], even.wrapping_add(odd));
        }
    }
}

#[simd_test]
fn pairwise_dot_product_u8x32<S: Simd>(simd: S) {
    let cases: [([u8; 32], [u8; 32]); 2] = [
        (
            [
                U8_MAX, U8_MAX, 0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
            ],
            [
                U8_MAX, U8_MAX, U8_MAX, U8_MAX, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33,
                35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63,
            ],
        ),
        (
            [
                1, U8_MAX, 2, U8_MAX_M1, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15,
                14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
            ],
            [
                U8_MAX, 1, 1, 0, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14,
                13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2,
            ],
        ),
    ];
    for (a, b) in cases {
        let lhs = u8x32::from_slice(simd, &a);
        let rhs = u8x32::from_slice(simd, &b);
        let result = lhs.pairwise_dot_product(rhs);
        for i in 0..16 {
            let even = (a[2 * i] as u16) * (b[2 * i] as u16);
            let odd = (a[2 * i + 1] as u16) * (b[2 * i + 1] as u16);
            assert_eq!(result[i], even.wrapping_add(odd));
        }
    }
}

#[simd_test]
fn pairwise_dot_product_u8x64<S: Simd>(simd: S) {
    let cases: [([u8; 64], [u8; 64]); 2] = [
        (
            [
                U8_MAX, U8_MAX, 0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
                42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
                63,
            ],
            [
                U8_MAX, U8_MAX, U8_MAX, U8_MAX, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33,
                35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75,
                77, 79, 81, 83, 85, 87, 89, 91, 93, 95, 97, 99, 101, 103, 105, 107, 109, 111, 113,
                115, 117, 119, 121, 123, 125, 127,
            ],
        ),
        (
            [
                1, U8_MAX, 2, U8_MAX_M1, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47,
                46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26,
                25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4,
                3, 2, 1,
            ],
            [
                U8_MAX, 1, 1, 0, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46,
                45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25,
                24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2,
            ],
        ),
    ];
    for (a, b) in cases {
        let lhs = u8x64::from_slice(simd, &a);
        let rhs = u8x64::from_slice(simd, &b);
        let result = lhs.pairwise_dot_product(rhs);
        for i in 0..32 {
            let even = (a[2 * i] as u16) * (b[2 * i] as u16);
            let odd = (a[2 * i + 1] as u16) * (b[2 * i + 1] as u16);
            assert_eq!(result[i], even.wrapping_add(odd));
        }
    }
}

#[simd_test]
fn pairwise_dot_product_u16x8<S: Simd>(simd: S) {
    let cases: [([u16; 8], [u16; 8]); 2] = [
        (
            [U16_MAX, U16_MAX, 0, 1, 4, 5, 6, 7],
            [U16_MAX, U16_MAX, U16_MAX, U16_MAX, 9, 11, 13, 15],
        ),
        (
            [1, U16_MAX, 2, U16_MAX_M1, 4, 3, 2, 1],
            [U16_MAX, 1, 1, 0, 5, 4, 3, 2],
        ),
    ];
    for (a, b) in cases {
        let lhs = u16x8::from_slice(simd, &a);
        let rhs = u16x8::from_slice(simd, &b);
        let result = lhs.pairwise_dot_product(rhs);
        for i in 0..4 {
            let even = (a[2 * i] as u32) * (b[2 * i] as u32);
            let odd = (a[2 * i + 1] as u32) * (b[2 * i + 1] as u32);
            assert_eq!(result[i], even.wrapping_add(odd));
        }
    }
}

#[simd_test]
fn pairwise_dot_product_u16x16<S: Simd>(simd: S) {
    let cases: [([u16; 16], [u16; 16]); 2] = [
        (
            [
                U16_MAX, U16_MAX, 0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            ],
            [
                U16_MAX, U16_MAX, U16_MAX, U16_MAX, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
            ],
        ),
        (
            [
                1, U16_MAX, 2, U16_MAX_M1, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
            ],
            [U16_MAX, 1, 1, 0, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2],
        ),
    ];
    for (a, b) in cases {
        let lhs = u16x16::from_slice(simd, &a);
        let rhs = u16x16::from_slice(simd, &b);
        let result = lhs.pairwise_dot_product(rhs);
        for i in 0..8 {
            let even = (a[2 * i] as u32) * (b[2 * i] as u32);
            let odd = (a[2 * i + 1] as u32) * (b[2 * i + 1] as u32);
            assert_eq!(result[i], even.wrapping_add(odd));
        }
    }
}

#[simd_test]
fn pairwise_dot_product_u16x32<S: Simd>(simd: S) {
    let cases: [([u16; 32], [u16; 32]); 2] = [
        (
            [
                U16_MAX, U16_MAX, 0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
            ],
            [
                U16_MAX, U16_MAX, U16_MAX, U16_MAX, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
                33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63,
            ],
        ),
        (
            [
                1, U16_MAX, 2, U16_MAX_M1, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15,
                14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
            ],
            [
                U16_MAX, 1, 1, 0, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14,
                13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2,
            ],
        ),
    ];
    for (a, b) in cases {
        let lhs = u16x32::from_slice(simd, &a);
        let rhs = u16x32::from_slice(simd, &b);
        let result = lhs.pairwise_dot_product(rhs);
        for i in 0..16 {
            let even = (a[2 * i] as u32) * (b[2 * i] as u32);
            let odd = (a[2 * i + 1] as u32) * (b[2 * i + 1] as u32);
            assert_eq!(result[i], even.wrapping_add(odd));
        }
    }
}

#[simd_test]
fn pairwise_dot_product_u32x4<S: Simd>(simd: S) {
    let cases: [([u32; 4], [u32; 4]); 2] = [
        (
            [U32_MAX, U32_MAX, 0, 1],
            [U32_MAX, U32_MAX, U32_MAX, U32_MAX],
        ),
        ([1, U32_MAX, 2, U32_MAX_M1], [U32_MAX, 1, 1, 0]),
    ];
    for (a, b) in cases {
        let lhs = u32x4::from_slice(simd, &a);
        let rhs = u32x4::from_slice(simd, &b);
        let result = lhs.pairwise_dot_product(rhs);
        for i in 0..2 {
            let even = (a[2 * i] as u64) * (b[2 * i] as u64);
            let odd = (a[2 * i + 1] as u64) * (b[2 * i + 1] as u64);
            assert_eq!(result[i], even.wrapping_add(odd));
        }
    }
}

#[simd_test]
fn pairwise_dot_product_u32x8<S: Simd>(simd: S) {
    let cases: [([u32; 8], [u32; 8]); 2] = [
        (
            [U32_MAX, U32_MAX, 0, 1, 4, 5, 6, 7],
            [U32_MAX, U32_MAX, U32_MAX, U32_MAX, 9, 11, 13, 15],
        ),
        (
            [1, U32_MAX, 2, U32_MAX_M1, 4, 3, 2, 1],
            [U32_MAX, 1, 1, 0, 5, 4, 3, 2],
        ),
    ];
    for (a, b) in cases {
        let lhs = u32x8::from_slice(simd, &a);
        let rhs = u32x8::from_slice(simd, &b);
        let result = lhs.pairwise_dot_product(rhs);
        for i in 0..4 {
            let even = (a[2 * i] as u64) * (b[2 * i] as u64);
            let odd = (a[2 * i + 1] as u64) * (b[2 * i + 1] as u64);
            assert_eq!(result[i], even.wrapping_add(odd));
        }
    }
}

#[simd_test]
fn pairwise_dot_product_u32x16<S: Simd>(simd: S) {
    let cases: [([u32; 16], [u32; 16]); 2] = [
        (
            [
                U32_MAX, U32_MAX, 0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            ],
            [
                U32_MAX, U32_MAX, U32_MAX, U32_MAX, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
            ],
        ),
        (
            [
                1, U32_MAX, 2, U32_MAX_M1, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
            ],
            [U32_MAX, 1, 1, 0, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2],
        ),
    ];
    for (a, b) in cases {
        let lhs = u32x16::from_slice(simd, &a);
        let rhs = u32x16::from_slice(simd, &b);
        let result = lhs.pairwise_dot_product(rhs);
        for i in 0..8 {
            let even = (a[2 * i] as u64) * (b[2 * i] as u64);
            let odd = (a[2 * i + 1] as u64) * (b[2 * i + 1] as u64);
            assert_eq!(result[i], even.wrapping_add(odd));
        }
    }
}

#[simd_test]
fn pairwise_dot_product_generic<S: Simd>(simd: S) {
    fn dot<S: Simd, V: SimdIntWiden<S>>(a: V, b: V) -> V::Widened {
        a.pairwise_dot_product(b)
    }

    let a = i16x8::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let b = i16x8::from_slice(simd, &[10, 20, 30, 40, 50, 60, 70, 80]);
    let result: i32x4<S> = dot(a, b);
    assert_eq!(*result, [50, 250, 610, 1130]);

    let min = i16x8::splat(simd, I16_MIN);
    assert_eq!(*dot(min, min), [I32_MIN; 4]);
}

#[simd_test]
fn pairwise_dot_product_native_i8<S: Simd>(simd: S) {
    let a = S::i8s::splat(simd, I8_MIN);
    let result: S::i16s = a.pairwise_dot_product(a);
    let product = (I8_MIN as i16) * (I8_MIN as i16);
    let expected = product.wrapping_add(product);
    assert!(result.as_slice().iter().all(|&lane| lane == expected));
}

#[simd_test]
fn pairwise_dot_product_native_i16<S: Simd>(simd: S) {
    let a = S::i16s::splat(simd, I16_MIN);
    let result: S::i32s = a.pairwise_dot_product(a);
    let product = (I16_MIN as i32) * (I16_MIN as i32);
    let expected = product.wrapping_add(product);
    assert!(result.as_slice().iter().all(|&lane| lane == expected));
}

#[simd_test]
fn pairwise_dot_product_native_i32<S: Simd>(simd: S) {
    let a = S::i32s::splat(simd, I32_MIN);
    let result: S::i64s = a.pairwise_dot_product(a);
    let product = (I32_MIN as i64) * (I32_MIN as i64);
    let expected = product.wrapping_add(product);
    assert!(result.as_slice().iter().all(|&lane| lane == expected));
}

#[simd_test]
fn pairwise_dot_product_native_u8<S: Simd>(simd: S) {
    let a = S::u8s::splat(simd, U8_MAX);
    let result: S::u16s = a.pairwise_dot_product(a);
    let product = (U8_MAX as u16) * (U8_MAX as u16);
    let expected = product.wrapping_add(product);
    assert!(result.as_slice().iter().all(|&lane| lane == expected));
}

#[simd_test]
fn pairwise_dot_product_native_u16<S: Simd>(simd: S) {
    let a = S::u16s::splat(simd, U16_MAX);
    let result: S::u32s = a.pairwise_dot_product(a);
    let product = (U16_MAX as u32) * (U16_MAX as u32);
    let expected = product.wrapping_add(product);
    assert!(result.as_slice().iter().all(|&lane| lane == expected));
}

#[simd_test]
fn pairwise_dot_product_native_u32<S: Simd>(simd: S) {
    let a = S::u32s::splat(simd, U32_MAX);
    let result: S::u64s = a.pairwise_dot_product(a);
    let product = (U32_MAX as u64) * (U32_MAX as u64);
    let expected = product.wrapping_add(product);
    assert!(result.as_slice().iter().all(|&lane| lane == expected));
}
