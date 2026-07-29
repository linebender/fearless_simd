// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// One concrete test row per supported vector type.

#[simd_test]
fn narrow_u16x16<S: Simd>(simd: S) {
    let a = u16x16::from_slice(
        simd,
        &[
            0, 1, 127, 128, 255, 256, 300, 1000, 128, 192, 224, 240, 248, 252, 254, 65535,
        ],
    );
    assert_eq!(
        *simd.narrow_u16x16(a),
        [
            0, 1, 127, 128, 255, 0, 44, 232, 128, 192, 224, 240, 248, 252, 254, 255
        ]
    );
}

#[simd_test]
fn narrow_u16x32<S: Simd>(simd: S) {
    let a = u16x32::from_slice(
        simd,
        &[
            0, 1, 127, 128, 255, 256, 300, 1000, 128, 192, 224, 240, 248, 252, 254, 255, 100, 200,
            255, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65535, 0, 1, 2, 3,
        ],
    );
    assert_eq!(
        *simd.narrow_u16x32(a),
        [
            0, 1, 127, 128, 255, 0, 44, 232, 128, 192, 224, 240, 248, 252, 254, 255, 100, 200, 255,
            0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 1, 2, 3
        ]
    );
}

#[simd_test]
fn narrow_u16x16_all_low_bytes<S: Simd>(simd: S) {
    for low_base in (0_u16..=240).step_by(16) {
        let values: [u16; 16] = core::array::from_fn(|i| {
            let low = low_base + i as u16;
            let high = (15 - i) as u16;
            (high << 8) | low
        });
        let expected = values.map(|value| value as u8);
        let result = simd.narrow_u16x16(u16x16::from_slice(simd, &values));

        assert_eq!(
            result.as_slice(),
            expected.as_slice(),
            "low-byte base {low_base}",
        );
    }
}

#[simd_test]
fn narrow_u16x32_all_low_bytes<S: Simd>(simd: S) {
    for low_base in (0_u16..=224).step_by(32) {
        let values: [u16; 32] = core::array::from_fn(|i| {
            let low = low_base + i as u16;
            let high = (31 - i) as u16;
            (high << 8) | low
        });
        let expected = values.map(|value| value as u8);
        let result = simd.narrow_u16x32(u16x32::from_slice(simd, &values));

        assert_eq!(
            result.as_slice(),
            expected.as_slice(),
            "low-byte base {low_base}",
        );
    }
}
