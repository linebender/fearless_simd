// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

#[simd_test]
fn pairwise_widening_add_i8x16<S: Simd>(simd: S) {
    let cases: [[i8; 16]; 6] = [
        [0; 16],
        [i8::MIN; 16],
        [i8::MAX; 16],
        [
            i8::MIN,
            i8::MAX,
            i8::MAX,
            i8::MIN,
            i8::MIN,
            i8::MAX,
            i8::MAX,
            i8::MIN,
            i8::MIN,
            i8::MAX,
            i8::MAX,
            i8::MIN,
            i8::MIN,
            i8::MAX,
            i8::MAX,
            i8::MIN,
        ],
        [-1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1],
        [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7],
    ];
    for a in cases {
        let input = i8x16::from_slice(simd, &a);
        let result: i16x8<S> = input.pairwise_widening_add();
        for i in 0..8 {
            let expected = (a[2 * i] as i16) + (a[2 * i + 1] as i16);
            assert_eq!(result[i], expected);
        }
    }
}

#[simd_test]
fn pairwise_widening_add_i8x32<S: Simd>(simd: S) {
    let cases: [[i8; 32]; 6] = [
        [0; 32],
        [i8::MIN; 32],
        [i8::MAX; 32],
        [
            i8::MIN,
            i8::MAX,
            i8::MAX,
            i8::MIN,
            i8::MIN,
            i8::MAX,
            i8::MAX,
            i8::MIN,
            i8::MIN,
            i8::MAX,
            i8::MAX,
            i8::MIN,
            i8::MIN,
            i8::MAX,
            i8::MAX,
            i8::MIN,
            i8::MIN,
            i8::MAX,
            i8::MAX,
            i8::MIN,
            i8::MIN,
            i8::MAX,
            i8::MAX,
            i8::MIN,
            i8::MIN,
            i8::MAX,
            i8::MAX,
            i8::MIN,
            i8::MIN,
            i8::MAX,
            i8::MAX,
            i8::MIN,
        ],
        [
            -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1,
            1, 1, -1, -1, 1, 1, -1,
        ],
        [
            -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4,
            5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        ],
    ];
    for a in cases {
        let input = i8x32::from_slice(simd, &a);
        let result: i16x16<S> = input.pairwise_widening_add();
        for i in 0..16 {
            let expected = (a[2 * i] as i16) + (a[2 * i + 1] as i16);
            assert_eq!(result[i], expected);
        }
    }
}

#[simd_test]
fn pairwise_widening_add_i8x64<S: Simd>(simd: S) {
    let cases: [[i8; 64]; 6] = [
        [0; 64],
        [i8::MIN; 64],
        [i8::MAX; 64],
        [
            i8::MIN,
            i8::MAX,
            i8::MAX,
            i8::MIN,
            i8::MIN,
            i8::MAX,
            i8::MAX,
            i8::MIN,
            i8::MIN,
            i8::MAX,
            i8::MAX,
            i8::MIN,
            i8::MIN,
            i8::MAX,
            i8::MAX,
            i8::MIN,
            i8::MIN,
            i8::MAX,
            i8::MAX,
            i8::MIN,
            i8::MIN,
            i8::MAX,
            i8::MAX,
            i8::MIN,
            i8::MIN,
            i8::MAX,
            i8::MAX,
            i8::MIN,
            i8::MIN,
            i8::MAX,
            i8::MAX,
            i8::MIN,
            i8::MIN,
            i8::MAX,
            i8::MAX,
            i8::MIN,
            i8::MIN,
            i8::MAX,
            i8::MAX,
            i8::MIN,
            i8::MIN,
            i8::MAX,
            i8::MAX,
            i8::MIN,
            i8::MIN,
            i8::MAX,
            i8::MAX,
            i8::MIN,
            i8::MIN,
            i8::MAX,
            i8::MAX,
            i8::MIN,
            i8::MIN,
            i8::MAX,
            i8::MAX,
            i8::MIN,
            i8::MIN,
            i8::MAX,
            i8::MAX,
            i8::MIN,
            i8::MIN,
            i8::MAX,
            i8::MAX,
            i8::MIN,
        ],
        [
            -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1,
            1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1,
            1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1,
        ],
        [
            -32, -31, -30, -29, -28, -27, -26, -25, -24, -23, -22, -21, -20, -19, -18, -17, -16,
            -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6,
            7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
            29, 30, 31,
        ],
    ];
    for a in cases {
        let input = i8x64::from_slice(simd, &a);
        let result: i16x32<S> = input.pairwise_widening_add();
        for i in 0..32 {
            let expected = (a[2 * i] as i16) + (a[2 * i + 1] as i16);
            assert_eq!(result[i], expected);
        }
    }
}

#[simd_test]
fn pairwise_widening_add_i16x8<S: Simd>(simd: S) {
    let cases: [[i16; 8]; 6] = [
        [0; 8],
        [i16::MIN; 8],
        [i16::MAX; 8],
        [
            i16::MIN,
            i16::MAX,
            i16::MAX,
            i16::MIN,
            i16::MIN,
            i16::MAX,
            i16::MAX,
            i16::MIN,
        ],
        [-1, 1, 1, -1, -1, 1, 1, -1],
        [-4, -3, -2, -1, 0, 1, 2, 3],
    ];
    for a in cases {
        let input = i16x8::from_slice(simd, &a);
        let result: i32x4<S> = input.pairwise_widening_add();
        for i in 0..4 {
            let expected = (a[2 * i] as i32) + (a[2 * i + 1] as i32);
            assert_eq!(result[i], expected);
        }
    }
}

#[simd_test]
fn pairwise_widening_add_i16x16<S: Simd>(simd: S) {
    let cases: [[i16; 16]; 6] = [
        [0; 16],
        [i16::MIN; 16],
        [i16::MAX; 16],
        [
            i16::MIN,
            i16::MAX,
            i16::MAX,
            i16::MIN,
            i16::MIN,
            i16::MAX,
            i16::MAX,
            i16::MIN,
            i16::MIN,
            i16::MAX,
            i16::MAX,
            i16::MIN,
            i16::MIN,
            i16::MAX,
            i16::MAX,
            i16::MIN,
        ],
        [-1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1],
        [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7],
    ];
    for a in cases {
        let input = i16x16::from_slice(simd, &a);
        let result: i32x8<S> = input.pairwise_widening_add();
        for i in 0..8 {
            let expected = (a[2 * i] as i32) + (a[2 * i + 1] as i32);
            assert_eq!(result[i], expected);
        }
    }
}

#[simd_test]
fn pairwise_widening_add_i16x32<S: Simd>(simd: S) {
    let cases: [[i16; 32]; 6] = [
        [0; 32],
        [i16::MIN; 32],
        [i16::MAX; 32],
        [
            i16::MIN,
            i16::MAX,
            i16::MAX,
            i16::MIN,
            i16::MIN,
            i16::MAX,
            i16::MAX,
            i16::MIN,
            i16::MIN,
            i16::MAX,
            i16::MAX,
            i16::MIN,
            i16::MIN,
            i16::MAX,
            i16::MAX,
            i16::MIN,
            i16::MIN,
            i16::MAX,
            i16::MAX,
            i16::MIN,
            i16::MIN,
            i16::MAX,
            i16::MAX,
            i16::MIN,
            i16::MIN,
            i16::MAX,
            i16::MAX,
            i16::MIN,
            i16::MIN,
            i16::MAX,
            i16::MAX,
            i16::MIN,
        ],
        [
            -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1,
            1, 1, -1, -1, 1, 1, -1,
        ],
        [
            -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4,
            5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        ],
    ];
    for a in cases {
        let input = i16x32::from_slice(simd, &a);
        let result: i32x16<S> = input.pairwise_widening_add();
        for i in 0..16 {
            let expected = (a[2 * i] as i32) + (a[2 * i + 1] as i32);
            assert_eq!(result[i], expected);
        }
    }
}

#[simd_test]
fn pairwise_widening_add_i32x4<S: Simd>(simd: S) {
    let cases: [[i32; 4]; 6] = [
        [0; 4],
        [i32::MIN; 4],
        [i32::MAX; 4],
        [i32::MIN, i32::MAX, i32::MAX, i32::MIN],
        [-1, 1, 1, -1],
        [-2, -1, 0, 1],
    ];
    for a in cases {
        let input = i32x4::from_slice(simd, &a);
        let result: i64x2<S> = input.pairwise_widening_add();
        for i in 0..2 {
            let expected = (a[2 * i] as i64) + (a[2 * i + 1] as i64);
            assert_eq!(result[i], expected);
        }
    }
}

#[simd_test]
fn pairwise_widening_add_i32x8<S: Simd>(simd: S) {
    let cases: [[i32; 8]; 6] = [
        [0; 8],
        [i32::MIN; 8],
        [i32::MAX; 8],
        [
            i32::MIN,
            i32::MAX,
            i32::MAX,
            i32::MIN,
            i32::MIN,
            i32::MAX,
            i32::MAX,
            i32::MIN,
        ],
        [-1, 1, 1, -1, -1, 1, 1, -1],
        [-4, -3, -2, -1, 0, 1, 2, 3],
    ];
    for a in cases {
        let input = i32x8::from_slice(simd, &a);
        let result: i64x4<S> = input.pairwise_widening_add();
        for i in 0..4 {
            let expected = (a[2 * i] as i64) + (a[2 * i + 1] as i64);
            assert_eq!(result[i], expected);
        }
    }
}

#[simd_test]
fn pairwise_widening_add_i32x16<S: Simd>(simd: S) {
    let cases: [[i32; 16]; 6] = [
        [0; 16],
        [i32::MIN; 16],
        [i32::MAX; 16],
        [
            i32::MIN,
            i32::MAX,
            i32::MAX,
            i32::MIN,
            i32::MIN,
            i32::MAX,
            i32::MAX,
            i32::MIN,
            i32::MIN,
            i32::MAX,
            i32::MAX,
            i32::MIN,
            i32::MIN,
            i32::MAX,
            i32::MAX,
            i32::MIN,
        ],
        [-1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1],
        [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7],
    ];
    for a in cases {
        let input = i32x16::from_slice(simd, &a);
        let result: i64x8<S> = input.pairwise_widening_add();
        for i in 0..8 {
            let expected = (a[2 * i] as i64) + (a[2 * i + 1] as i64);
            assert_eq!(result[i], expected);
        }
    }
}

#[simd_test]
fn pairwise_widening_add_u8x16<S: Simd>(simd: S) {
    let cases: [[u8; 16]; 5] = [
        [0; 16],
        [u8::MAX - 1; 16],
        [u8::MAX; 16],
        [
            u8::MIN,
            u8::MAX,
            u8::MAX,
            u8::MIN,
            u8::MIN,
            u8::MAX,
            u8::MAX,
            u8::MIN,
            u8::MIN,
            u8::MAX,
            u8::MAX,
            u8::MIN,
            u8::MIN,
            u8::MAX,
            u8::MAX,
            u8::MIN,
        ],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    ];
    for a in cases {
        let input = u8x16::from_slice(simd, &a);
        let result: u16x8<S> = input.pairwise_widening_add();
        for i in 0..8 {
            let expected = (a[2 * i] as u16) + (a[2 * i + 1] as u16);
            assert_eq!(result[i], expected);
        }
    }
}

#[simd_test]
fn pairwise_widening_add_u8x32<S: Simd>(simd: S) {
    let cases: [[u8; 32]; 5] = [
        [0; 32],
        [u8::MAX - 1; 32],
        [u8::MAX; 32],
        [
            u8::MIN,
            u8::MAX,
            u8::MAX,
            u8::MIN,
            u8::MIN,
            u8::MAX,
            u8::MAX,
            u8::MIN,
            u8::MIN,
            u8::MAX,
            u8::MAX,
            u8::MIN,
            u8::MIN,
            u8::MAX,
            u8::MAX,
            u8::MIN,
            u8::MIN,
            u8::MAX,
            u8::MAX,
            u8::MIN,
            u8::MIN,
            u8::MAX,
            u8::MAX,
            u8::MIN,
            u8::MIN,
            u8::MAX,
            u8::MAX,
            u8::MIN,
            u8::MIN,
            u8::MAX,
            u8::MAX,
            u8::MIN,
        ],
        [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31,
        ],
    ];
    for a in cases {
        let input = u8x32::from_slice(simd, &a);
        let result: u16x16<S> = input.pairwise_widening_add();
        for i in 0..16 {
            let expected = (a[2 * i] as u16) + (a[2 * i + 1] as u16);
            assert_eq!(result[i], expected);
        }
    }
}

#[simd_test]
fn pairwise_widening_add_u8x64<S: Simd>(simd: S) {
    let cases: [[u8; 64]; 5] = [
        [0; 64],
        [u8::MAX - 1; 64],
        [u8::MAX; 64],
        [
            u8::MIN,
            u8::MAX,
            u8::MAX,
            u8::MIN,
            u8::MIN,
            u8::MAX,
            u8::MAX,
            u8::MIN,
            u8::MIN,
            u8::MAX,
            u8::MAX,
            u8::MIN,
            u8::MIN,
            u8::MAX,
            u8::MAX,
            u8::MIN,
            u8::MIN,
            u8::MAX,
            u8::MAX,
            u8::MIN,
            u8::MIN,
            u8::MAX,
            u8::MAX,
            u8::MIN,
            u8::MIN,
            u8::MAX,
            u8::MAX,
            u8::MIN,
            u8::MIN,
            u8::MAX,
            u8::MAX,
            u8::MIN,
            u8::MIN,
            u8::MAX,
            u8::MAX,
            u8::MIN,
            u8::MIN,
            u8::MAX,
            u8::MAX,
            u8::MIN,
            u8::MIN,
            u8::MAX,
            u8::MAX,
            u8::MIN,
            u8::MIN,
            u8::MAX,
            u8::MAX,
            u8::MIN,
            u8::MIN,
            u8::MAX,
            u8::MAX,
            u8::MIN,
            u8::MIN,
            u8::MAX,
            u8::MAX,
            u8::MIN,
            u8::MIN,
            u8::MAX,
            u8::MAX,
            u8::MIN,
            u8::MIN,
            u8::MAX,
            u8::MAX,
            u8::MIN,
        ],
        [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
            46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
        ],
    ];
    for a in cases {
        let input = u8x64::from_slice(simd, &a);
        let result: u16x32<S> = input.pairwise_widening_add();
        for i in 0..32 {
            let expected = (a[2 * i] as u16) + (a[2 * i + 1] as u16);
            assert_eq!(result[i], expected);
        }
    }
}

#[simd_test]
fn pairwise_widening_add_u16x8<S: Simd>(simd: S) {
    let cases: [[u16; 8]; 5] = [
        [0; 8],
        [u16::MAX - 1; 8],
        [u16::MAX; 8],
        [
            u16::MIN,
            u16::MAX,
            u16::MAX,
            u16::MIN,
            u16::MIN,
            u16::MAX,
            u16::MAX,
            u16::MIN,
        ],
        [0, 1, 2, 3, 4, 5, 6, 7],
    ];
    for a in cases {
        let input = u16x8::from_slice(simd, &a);
        let result: u32x4<S> = input.pairwise_widening_add();
        for i in 0..4 {
            let expected = (a[2 * i] as u32) + (a[2 * i + 1] as u32);
            assert_eq!(result[i], expected);
        }
    }
}

#[simd_test]
fn pairwise_widening_add_u16x16<S: Simd>(simd: S) {
    let cases: [[u16; 16]; 5] = [
        [0; 16],
        [u16::MAX - 1; 16],
        [u16::MAX; 16],
        [
            u16::MIN,
            u16::MAX,
            u16::MAX,
            u16::MIN,
            u16::MIN,
            u16::MAX,
            u16::MAX,
            u16::MIN,
            u16::MIN,
            u16::MAX,
            u16::MAX,
            u16::MIN,
            u16::MIN,
            u16::MAX,
            u16::MAX,
            u16::MIN,
        ],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    ];
    for a in cases {
        let input = u16x16::from_slice(simd, &a);
        let result: u32x8<S> = input.pairwise_widening_add();
        for i in 0..8 {
            let expected = (a[2 * i] as u32) + (a[2 * i + 1] as u32);
            assert_eq!(result[i], expected);
        }
    }
}

#[simd_test]
fn pairwise_widening_add_u16x32<S: Simd>(simd: S) {
    let cases: [[u16; 32]; 5] = [
        [0; 32],
        [u16::MAX - 1; 32],
        [u16::MAX; 32],
        [
            u16::MIN,
            u16::MAX,
            u16::MAX,
            u16::MIN,
            u16::MIN,
            u16::MAX,
            u16::MAX,
            u16::MIN,
            u16::MIN,
            u16::MAX,
            u16::MAX,
            u16::MIN,
            u16::MIN,
            u16::MAX,
            u16::MAX,
            u16::MIN,
            u16::MIN,
            u16::MAX,
            u16::MAX,
            u16::MIN,
            u16::MIN,
            u16::MAX,
            u16::MAX,
            u16::MIN,
            u16::MIN,
            u16::MAX,
            u16::MAX,
            u16::MIN,
            u16::MIN,
            u16::MAX,
            u16::MAX,
            u16::MIN,
        ],
        [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31,
        ],
    ];
    for a in cases {
        let input = u16x32::from_slice(simd, &a);
        let result: u32x16<S> = input.pairwise_widening_add();
        for i in 0..16 {
            let expected = (a[2 * i] as u32) + (a[2 * i + 1] as u32);
            assert_eq!(result[i], expected);
        }
    }
}

#[simd_test]
fn pairwise_widening_add_u32x4<S: Simd>(simd: S) {
    let cases: [[u32; 4]; 5] = [
        [0; 4],
        [u32::MAX - 1; 4],
        [u32::MAX; 4],
        [u32::MIN, u32::MAX, u32::MAX, u32::MIN],
        [0, 1, 2, 3],
    ];
    for a in cases {
        let input = u32x4::from_slice(simd, &a);
        let result: u64x2<S> = input.pairwise_widening_add();
        for i in 0..2 {
            let expected = (a[2 * i] as u64) + (a[2 * i + 1] as u64);
            assert_eq!(result[i], expected);
        }
    }
}

#[simd_test]
fn pairwise_widening_add_u32x8<S: Simd>(simd: S) {
    let cases: [[u32; 8]; 5] = [
        [0; 8],
        [u32::MAX - 1; 8],
        [u32::MAX; 8],
        [
            u32::MIN,
            u32::MAX,
            u32::MAX,
            u32::MIN,
            u32::MIN,
            u32::MAX,
            u32::MAX,
            u32::MIN,
        ],
        [0, 1, 2, 3, 4, 5, 6, 7],
    ];
    for a in cases {
        let input = u32x8::from_slice(simd, &a);
        let result: u64x4<S> = input.pairwise_widening_add();
        for i in 0..4 {
            let expected = (a[2 * i] as u64) + (a[2 * i + 1] as u64);
            assert_eq!(result[i], expected);
        }
    }
}

#[simd_test]
fn pairwise_widening_add_u32x16<S: Simd>(simd: S) {
    let cases: [[u32; 16]; 5] = [
        [0; 16],
        [u32::MAX - 1; 16],
        [u32::MAX; 16],
        [
            u32::MIN,
            u32::MAX,
            u32::MAX,
            u32::MIN,
            u32::MIN,
            u32::MAX,
            u32::MAX,
            u32::MIN,
            u32::MIN,
            u32::MAX,
            u32::MAX,
            u32::MIN,
            u32::MIN,
            u32::MAX,
            u32::MAX,
            u32::MIN,
        ],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    ];
    for a in cases {
        let input = u32x16::from_slice(simd, &a);
        let result: u64x8<S> = input.pairwise_widening_add();
        for i in 0..8 {
            let expected = (a[2 * i] as u64) + (a[2 * i + 1] as u64);
            assert_eq!(result[i], expected);
        }
    }
}
