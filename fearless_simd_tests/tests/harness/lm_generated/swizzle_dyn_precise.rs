// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

fn expected_swizzle_precise<const N: usize>(bytes: [u8; N], indices: [u8; N]) -> [u8; N] {
    core::array::from_fn(|i| {
        let index = indices[i] as usize;
        bytes.get(index).copied().unwrap_or(0)
    })
}

#[simd_test]
fn swizzle_dyn_precise_u8x16<S: Simd>(simd: S) {
    let bytes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let indices = [15, 14, 0, 1, 16, 17, 31, 127, 128, 255, 8, 7, 6, 5, 4, 3];
    let expected = expected_swizzle_precise(bytes, indices);

    let value = u8x16::simd_from(simd, bytes);
    let index_vec = u8x16::simd_from(simd, indices);
    let result = value.swizzle_dyn_precise(index_vec);

    assert_eq!(*result, expected);
}

#[simd_test]
fn swizzle_dyn_precise_u8x32_crosses_blocks<S: Simd>(simd: S) {
    let bytes: [u8; 32] = core::array::from_fn(|i| u8::try_from(i + 1).unwrap());
    let indices = [
        31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 0, 1, 2, 3, 4, 5, 6, 7, 32,
        33, 127, 128, 255, 15, 16, 31,
    ];
    let expected = expected_swizzle_precise(bytes, indices);

    let value = u8x32::simd_from(simd, bytes);
    let index_vec = u8x32::simd_from(simd, indices);
    let result = value.swizzle_dyn_precise(index_vec);

    assert_eq!(*result, expected);
}

#[simd_test]
fn swizzle_dyn_precise_u8x64_crosses_blocks<S: Simd>(simd: S) {
    let bytes: [u8; 64] = core::array::from_fn(|i| u8::try_from(i + 1).unwrap());
    let indices = [
        63, 48, 32, 16, 0, 15, 31, 47, 62, 49, 33, 17, 1, 14, 30, 46, 45, 29, 13, 2, 18, 34, 50,
        61, 60, 44, 28, 12, 3, 19, 35, 51, 52, 36, 20, 4, 11, 27, 43, 59, 58, 42, 26, 10, 5, 21,
        37, 53, 64, 65, 127, 128, 255, 54, 38, 22, 6, 9, 25, 41, 57, 56, 40, 24,
    ];
    let expected = expected_swizzle_precise(bytes, indices);

    let value = u8x64::simd_from(simd, bytes);
    let index_vec = u8x64::simd_from(simd, indices);
    let result = value.swizzle_dyn_precise(index_vec);

    assert_eq!(*result, expected);
}

#[simd_test]
fn swizzle_dyn_precise_bitcast_f32x8<S: Simd>(simd: S) {
    let bytes: [u8; 32] = core::array::from_fn(|i| u8::try_from(i * 3 + 1).unwrap());
    let indices = [
        16, 17, 18, 19, 0, 1, 2, 3, 31, 30, 29, 28, 32, 33, 128, 255, 4, 5, 6, 7, 20, 21, 22, 23,
        27, 26, 25, 24, 15, 14, 13, 12,
    ];
    let expected = expected_swizzle_precise(bytes, indices);

    let byte_vec = u8x32::simd_from(simd, bytes);
    let value: f32x8<S> = byte_vec.bitcast();
    let index_vec = u8x32::simd_from(simd, indices);
    let result_bytes: u8x32<S> = value.swizzle_dyn_precise(index_vec).bitcast();

    assert_eq!(*result_bytes, expected);
}

#[simd_test]
fn swizzle_dyn_precise_generic_indices<S: Simd>(simd: S) {
    #[inline(always)]
    fn do_swizzle<S: Simd, V: SimdBase<S>>(value: V, indices: V::Bytes) -> V {
        value.swizzle_dyn_precise(indices)
    }

    let bytes: [u8; 32] = core::array::from_fn(|i| u8::try_from(i + 1).unwrap());
    let indices = [
        16, 17, 18, 19, 20, 21, 22, 23, 31, 30, 29, 28, 27, 26, 25, 24, 0, 1, 2, 3, 4, 5, 6, 7, 32,
        64, 127, 128, 255, 8, 9, 10,
    ];
    let expected = expected_swizzle_precise(bytes, indices);

    let value = u8x32::simd_from(simd, bytes);
    let index_vec = u8x32::simd_from(simd, indices);
    let result = do_swizzle::<S, u8x32<S>>(value, index_vec);

    assert_eq!(*result, expected);
}

#[simd_test]
fn swizzle_dyn_precise_random_u8_widths<S: Simd>(simd: S) {
    let mut rng = fastrand::Rng::with_seed(0x5eed_5eed_cafe_f00d);

    for iteration in 0..100_000 {
        let mut bytes = [0u8; 64];
        let mut indices = [0u8; 64];
        rng.fill(&mut bytes);
        rng.fill(&mut indices);

        let bytes16 = &bytes[..16];
        let indices16 = &indices[..16];
        let expected16 = expected_swizzle_precise_from_slice::<16>(bytes16, indices16);
        let value16 = u8x16::from_slice(simd, bytes16);
        let index_vec16 = u8x16::from_slice(simd, indices16);
        let result16 = value16.swizzle_dyn_precise(index_vec16);
        assert_eq!(*result16, expected16, "iteration {iteration}, width 16");

        let bytes32 = &bytes[..32];
        let indices32 = &indices[..32];
        let expected32 = expected_swizzle_precise_from_slice::<32>(bytes32, indices32);
        let value32 = u8x32::from_slice(simd, bytes32);
        let index_vec32 = u8x32::from_slice(simd, indices32);
        let result32 = value32.swizzle_dyn_precise(index_vec32);
        assert_eq!(*result32, expected32, "iteration {iteration}, width 32");

        let bytes64 = &bytes[..64];
        let indices64 = &indices[..64];
        let expected64 = expected_swizzle_precise_from_slice::<64>(bytes64, indices64);
        let value64 = u8x64::from_slice(simd, bytes64);
        let index_vec64 = u8x64::from_slice(simd, indices64);
        let result64 = value64.swizzle_dyn_precise(index_vec64);
        assert_eq!(*result64, expected64, "iteration {iteration}, width 64");
    }
}

fn expected_swizzle_precise_from_slice<const N: usize>(bytes: &[u8], indices: &[u8]) -> [u8; N] {
    assert_eq!(bytes.len(), N);
    assert_eq!(indices.len(), N);
    core::array::from_fn(|i| {
        let index = indices[i] as usize;
        bytes.get(index).copied().unwrap_or(0)
    })
}
