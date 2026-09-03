// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

fn assert_swizzle_dyn<const N: usize>(bytes: [u8; N], indices: [u8; N], result: [u8; N]) {
    for i in 0..N {
        let index = indices[i] as usize;
        if index < N {
            assert_eq!(result[i], bytes[index], "output lane {i}, index {index}");
        }
    }
}

#[simd_test]
fn swizzle_dyn_u8x16<S: Simd>(simd: S) {
    let bytes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let indices = [15, 14, 0, 1, 16, 17, 31, 127, 128, 255, 8, 7, 6, 5, 4, 3];

    let value = u8x16::simd_from(simd, bytes);
    let index_vec = u8x16::simd_from(simd, indices);
    let result = value.swizzle_dyn(index_vec);

    assert_swizzle_dyn(bytes, indices, *result);
}

#[simd_test]
fn swizzle_dyn_u8x32_crosses_blocks<S: Simd>(simd: S) {
    let bytes: [u8; 32] = core::array::from_fn(|i| u8::try_from(i + 1).unwrap());
    let indices = [
        31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 0, 1, 2, 3, 4, 5, 6, 7, 32,
        33, 127, 128, 255, 15, 16, 31,
    ];

    let value = u8x32::simd_from(simd, bytes);
    let index_vec = u8x32::simd_from(simd, indices);
    let result = value.swizzle_dyn(index_vec);

    assert_swizzle_dyn(bytes, indices, *result);
}

#[simd_test]
fn swizzle_dyn_u8x32_every_valid_index_in_both_lanes<S: Simd>(simd: S) {
    // Every value has four bits set, so ORing two distinct table bytes cannot
    // accidentally produce either input byte.
    let bytes = [
        15, 23, 27, 29, 30, 39, 43, 45, 46, 51, 53, 54, 57, 58, 60, 71, 75, 77, 78, 83, 85, 86, 89,
        90, 92, 99, 101, 102, 105, 106, 108, 113,
    ];
    let value = u8x32::simd_from(simd, bytes);

    for index in 0_u8..32 {
        let index_vec = u8x32::simd_from(simd, [index; 32]);
        let result = value.swizzle_dyn(index_vec);

        assert_eq!(*result, [bytes[usize::from(index)]; 32], "index {index}");
    }
}

#[simd_test]
fn swizzle_dyn_u8x64_crosses_blocks<S: Simd>(simd: S) {
    let bytes: [u8; 64] = core::array::from_fn(|i| u8::try_from(i + 1).unwrap());
    let indices: [u8; 64] = core::array::from_fn(|i| {
        if i % 7 == 0 {
            u8::try_from(64 + i).unwrap()
        } else {
            u8::try_from((i * 17) % 64).unwrap()
        }
    });

    let value = u8x64::simd_from(simd, bytes);
    let index_vec = u8x64::simd_from(simd, indices);
    let result = value.swizzle_dyn(index_vec);

    assert_swizzle_dyn(bytes, indices, *result);
}

#[simd_test]
fn swizzle_dyn_bitcast_f32x8<S: Simd>(simd: S) {
    let bytes: [u8; 32] = core::array::from_fn(|i| u8::try_from(i * 3 + 1).unwrap());
    let indices = [
        16, 17, 18, 19, 0, 1, 2, 3, 31, 30, 29, 28, 32, 33, 128, 255, 4, 5, 6, 7, 20, 21, 22, 23,
        27, 26, 25, 24, 15, 14, 13, 12,
    ];

    let byte_vec = u8x32::simd_from(simd, bytes);
    let value: f32x8<S> = byte_vec.bitcast();
    let index_vec = u8x32::simd_from(simd, indices);
    let result_bytes: u8x32<S> = value.swizzle_dyn(index_vec).bitcast();

    assert_swizzle_dyn(bytes, indices, *result_bytes);
}

#[simd_test]
fn swizzle_dyn_generic_indices<S: Simd>(simd: S) {
    #[inline(always)]
    fn do_swizzle<S: Simd, V: SimdBase<S>>(value: V, indices: V::Bytes) -> V {
        value.swizzle_dyn(indices)
    }

    let bytes: [u8; 32] = core::array::from_fn(|i| u8::try_from(i + 1).unwrap());
    let indices = [
        16, 17, 18, 19, 20, 21, 22, 23, 31, 30, 29, 28, 27, 26, 25, 24, 0, 1, 2, 3, 4, 5, 6, 7, 32,
        64, 127, 128, 255, 8, 9, 10,
    ];

    let value = u8x32::simd_from(simd, bytes);
    let index_vec = u8x32::simd_from(simd, indices);
    let result = do_swizzle::<S, u8x32<S>>(value, index_vec);

    assert_swizzle_dyn(bytes, indices, *result);
}

#[simd_test]
#[ignore = "this test is slow"]
// run with: cargo test --release swizzle_dyn_random_u8_all_widths -- --ignored
fn swizzle_dyn_random_u8_all_widths<S: Simd>(simd: S) {
    let mut rng = fastrand::Rng::with_seed(0x5eed_5eed_cafe_f00d);

    for iteration in 0..100_000 {
        let mut bytes: [u8; 64] = [0; 64];
        let mut indices: [u8; 64] = [0; 64];
        rng.fill(&mut bytes);
        rng.fill(&mut indices);

        let bytes16 = &bytes[..16];
        let indices16 = &indices[..16];
        let value16 = u8x16::from_slice(simd, bytes16);
        let index_vec16 = u8x16::from_slice(simd, indices16);
        let result16 = value16.swizzle_dyn(index_vec16);
        assert_random_swizzle_dyn_from_slice(bytes16, indices16, *result16, iteration);

        let bytes32 = &bytes[..32];
        let indices32 = &indices[..32];
        let value32 = u8x32::from_slice(simd, bytes32);
        let index_vec32 = u8x32::from_slice(simd, indices32);
        let result32 = value32.swizzle_dyn(index_vec32);
        assert_random_swizzle_dyn_from_slice(bytes32, indices32, *result32, iteration);

        let bytes64 = &bytes[..64];
        let indices64 = &indices[..64];
        let value64 = u8x64::from_slice(simd, bytes64);
        let index_vec64 = u8x64::from_slice(simd, indices64);
        let result64 = value64.swizzle_dyn(index_vec64);
        assert_random_swizzle_dyn_from_slice(bytes64, indices64, *result64, iteration);
    }
}

fn assert_random_swizzle_dyn_from_slice<const N: usize>(
    bytes: &[u8],
    indices: &[u8],
    result: [u8; N],
    iteration: usize,
) {
    assert_eq!(bytes.len(), N);
    assert_eq!(indices.len(), N);
    for i in 0..N {
        let index = indices[i] as usize;
        if index < N {
            assert_eq!(
                result[i], bytes[index],
                "iteration {iteration}, width {N}, output lane {i}, index {index}"
            );
        }
    }
}
