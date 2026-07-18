// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// One concrete test row per supported vector type.

fn expected_swizzle_within_blocks<const N: usize>(bytes: [u8; N], indices: [u8; N]) -> [u8; N] {
    assert_eq!(N % 16, 0);
    core::array::from_fn(|i| {
        let block_start = (i / 16) * 16;
        bytes[block_start + usize::from(indices[i])]
    })
}

#[expect(clippy::cast_possible_truncation, reason = "truncation is deliberate")]
fn swizzle_test_byte(i: usize) -> u8 {
    (i * 37 + 11) as u8
}

fn swizzle_test_index(i: usize) -> u8 {
    u8::try_from(15 - (i % 16)).unwrap()
}

#[simd_test]
fn swizzle_dyn_within_blocks_f32x4<S: Simd>(simd: S) {
    let bytes: [u8; 16] = core::array::from_fn(swizzle_test_byte);
    let indices: [u8; 16] = core::array::from_fn(swizzle_test_index);
    let expected = expected_swizzle_within_blocks(bytes, indices);

    let byte_vec = u8x16::simd_from(simd, bytes);
    let index_vec = u8x16::simd_from(simd, indices);
    let value: f32x4<S> = byte_vec.bitcast();
    let result_bytes: u8x16<S> = value.swizzle_dyn_within_blocks(index_vec).bitcast();

    assert_eq!(*result_bytes, expected);
}

#[simd_test]
fn swizzle_dyn_within_blocks_i8x16<S: Simd>(simd: S) {
    let bytes: [u8; 16] = core::array::from_fn(swizzle_test_byte);
    let indices: [u8; 16] = core::array::from_fn(swizzle_test_index);
    let expected = expected_swizzle_within_blocks(bytes, indices);

    let byte_vec = u8x16::simd_from(simd, bytes);
    let index_vec = u8x16::simd_from(simd, indices);
    let value: i8x16<S> = byte_vec.bitcast();
    let result_bytes: u8x16<S> = value.swizzle_dyn_within_blocks(index_vec).bitcast();

    assert_eq!(*result_bytes, expected);
}

#[simd_test]
fn swizzle_dyn_within_blocks_u8x16<S: Simd>(simd: S) {
    let bytes: [u8; 16] = core::array::from_fn(swizzle_test_byte);
    let indices: [u8; 16] = core::array::from_fn(swizzle_test_index);
    let expected = expected_swizzle_within_blocks(bytes, indices);

    let byte_vec = u8x16::simd_from(simd, bytes);
    let index_vec = u8x16::simd_from(simd, indices);
    let value: u8x16<S> = byte_vec.bitcast();
    let result_bytes: u8x16<S> = value.swizzle_dyn_within_blocks(index_vec).bitcast();

    assert_eq!(*result_bytes, expected);
}

#[simd_test]
fn swizzle_dyn_within_blocks_i16x8<S: Simd>(simd: S) {
    let bytes: [u8; 16] = core::array::from_fn(swizzle_test_byte);
    let indices: [u8; 16] = core::array::from_fn(swizzle_test_index);
    let expected = expected_swizzle_within_blocks(bytes, indices);

    let byte_vec = u8x16::simd_from(simd, bytes);
    let index_vec = u8x16::simd_from(simd, indices);
    let value: i16x8<S> = byte_vec.bitcast();
    let result_bytes: u8x16<S> = value.swizzle_dyn_within_blocks(index_vec).bitcast();

    assert_eq!(*result_bytes, expected);
}

#[simd_test]
fn swizzle_dyn_within_blocks_u16x8<S: Simd>(simd: S) {
    let bytes: [u8; 16] = core::array::from_fn(swizzle_test_byte);
    let indices: [u8; 16] = core::array::from_fn(swizzle_test_index);
    let expected = expected_swizzle_within_blocks(bytes, indices);

    let byte_vec = u8x16::simd_from(simd, bytes);
    let index_vec = u8x16::simd_from(simd, indices);
    let value: u16x8<S> = byte_vec.bitcast();
    let result_bytes: u8x16<S> = value.swizzle_dyn_within_blocks(index_vec).bitcast();

    assert_eq!(*result_bytes, expected);
}

#[simd_test]
fn swizzle_dyn_within_blocks_i32x4<S: Simd>(simd: S) {
    let bytes: [u8; 16] = core::array::from_fn(swizzle_test_byte);
    let indices: [u8; 16] = core::array::from_fn(swizzle_test_index);
    let expected = expected_swizzle_within_blocks(bytes, indices);

    let byte_vec = u8x16::simd_from(simd, bytes);
    let index_vec = u8x16::simd_from(simd, indices);
    let value: i32x4<S> = byte_vec.bitcast();
    let result_bytes: u8x16<S> = value.swizzle_dyn_within_blocks(index_vec).bitcast();

    assert_eq!(*result_bytes, expected);
}

#[simd_test]
fn swizzle_dyn_within_blocks_u32x4<S: Simd>(simd: S) {
    let bytes: [u8; 16] = core::array::from_fn(swizzle_test_byte);
    let indices: [u8; 16] = core::array::from_fn(swizzle_test_index);
    let expected = expected_swizzle_within_blocks(bytes, indices);

    let byte_vec = u8x16::simd_from(simd, bytes);
    let index_vec = u8x16::simd_from(simd, indices);
    let value: u32x4<S> = byte_vec.bitcast();
    let result_bytes: u8x16<S> = value.swizzle_dyn_within_blocks(index_vec).bitcast();

    assert_eq!(*result_bytes, expected);
}

#[simd_test]
fn swizzle_dyn_within_blocks_f64x2<S: Simd>(simd: S) {
    let bytes: [u8; 16] = core::array::from_fn(swizzle_test_byte);
    let indices: [u8; 16] = core::array::from_fn(swizzle_test_index);
    let expected = expected_swizzle_within_blocks(bytes, indices);

    let byte_vec = u8x16::simd_from(simd, bytes);
    let index_vec = u8x16::simd_from(simd, indices);
    let value: f64x2<S> = byte_vec.bitcast();
    let result_bytes: u8x16<S> = value.swizzle_dyn_within_blocks(index_vec).bitcast();

    assert_eq!(*result_bytes, expected);
}

#[simd_test]
fn swizzle_dyn_within_blocks_f32x8<S: Simd>(simd: S) {
    let bytes: [u8; 32] = core::array::from_fn(swizzle_test_byte);
    let indices: [u8; 32] = core::array::from_fn(swizzle_test_index);
    let expected = expected_swizzle_within_blocks(bytes, indices);

    let byte_vec = u8x32::simd_from(simd, bytes);
    let index_vec = u8x32::simd_from(simd, indices);
    let value: f32x8<S> = byte_vec.bitcast();
    let result_bytes: u8x32<S> = value.swizzle_dyn_within_blocks(index_vec).bitcast();

    assert_eq!(*result_bytes, expected);
}

#[simd_test]
fn swizzle_dyn_within_blocks_i8x32<S: Simd>(simd: S) {
    let bytes: [u8; 32] = core::array::from_fn(swizzle_test_byte);
    let indices: [u8; 32] = core::array::from_fn(swizzle_test_index);
    let expected = expected_swizzle_within_blocks(bytes, indices);

    let byte_vec = u8x32::simd_from(simd, bytes);
    let index_vec = u8x32::simd_from(simd, indices);
    let value: i8x32<S> = byte_vec.bitcast();
    let result_bytes: u8x32<S> = value.swizzle_dyn_within_blocks(index_vec).bitcast();

    assert_eq!(*result_bytes, expected);
}

#[simd_test]
fn swizzle_dyn_within_blocks_u8x32<S: Simd>(simd: S) {
    let bytes: [u8; 32] = core::array::from_fn(swizzle_test_byte);
    let indices: [u8; 32] = core::array::from_fn(swizzle_test_index);
    let expected = expected_swizzle_within_blocks(bytes, indices);

    let byte_vec = u8x32::simd_from(simd, bytes);
    let index_vec = u8x32::simd_from(simd, indices);
    let value: u8x32<S> = byte_vec.bitcast();
    let result_bytes: u8x32<S> = value.swizzle_dyn_within_blocks(index_vec).bitcast();

    assert_eq!(*result_bytes, expected);
}

#[simd_test]
fn swizzle_dyn_within_blocks_i16x16<S: Simd>(simd: S) {
    let bytes: [u8; 32] = core::array::from_fn(swizzle_test_byte);
    let indices: [u8; 32] = core::array::from_fn(swizzle_test_index);
    let expected = expected_swizzle_within_blocks(bytes, indices);

    let byte_vec = u8x32::simd_from(simd, bytes);
    let index_vec = u8x32::simd_from(simd, indices);
    let value: i16x16<S> = byte_vec.bitcast();
    let result_bytes: u8x32<S> = value.swizzle_dyn_within_blocks(index_vec).bitcast();

    assert_eq!(*result_bytes, expected);
}

#[simd_test]
fn swizzle_dyn_within_blocks_u16x16<S: Simd>(simd: S) {
    let bytes: [u8; 32] = core::array::from_fn(swizzle_test_byte);
    let indices: [u8; 32] = core::array::from_fn(swizzle_test_index);
    let expected = expected_swizzle_within_blocks(bytes, indices);

    let byte_vec = u8x32::simd_from(simd, bytes);
    let index_vec = u8x32::simd_from(simd, indices);
    let value: u16x16<S> = byte_vec.bitcast();
    let result_bytes: u8x32<S> = value.swizzle_dyn_within_blocks(index_vec).bitcast();

    assert_eq!(*result_bytes, expected);
}

#[simd_test]
fn swizzle_dyn_within_blocks_i32x8<S: Simd>(simd: S) {
    let bytes: [u8; 32] = core::array::from_fn(swizzle_test_byte);
    let indices: [u8; 32] = core::array::from_fn(swizzle_test_index);
    let expected = expected_swizzle_within_blocks(bytes, indices);

    let byte_vec = u8x32::simd_from(simd, bytes);
    let index_vec = u8x32::simd_from(simd, indices);
    let value: i32x8<S> = byte_vec.bitcast();
    let result_bytes: u8x32<S> = value.swizzle_dyn_within_blocks(index_vec).bitcast();

    assert_eq!(*result_bytes, expected);
}

#[simd_test]
fn swizzle_dyn_within_blocks_u32x8<S: Simd>(simd: S) {
    let bytes: [u8; 32] = core::array::from_fn(swizzle_test_byte);
    let indices: [u8; 32] = core::array::from_fn(swizzle_test_index);
    let expected = expected_swizzle_within_blocks(bytes, indices);

    let byte_vec = u8x32::simd_from(simd, bytes);
    let index_vec = u8x32::simd_from(simd, indices);
    let value: u32x8<S> = byte_vec.bitcast();
    let result_bytes: u8x32<S> = value.swizzle_dyn_within_blocks(index_vec).bitcast();

    assert_eq!(*result_bytes, expected);
}

#[simd_test]
fn swizzle_dyn_within_blocks_f64x4<S: Simd>(simd: S) {
    let bytes: [u8; 32] = core::array::from_fn(swizzle_test_byte);
    let indices: [u8; 32] = core::array::from_fn(swizzle_test_index);
    let expected = expected_swizzle_within_blocks(bytes, indices);

    let byte_vec = u8x32::simd_from(simd, bytes);
    let index_vec = u8x32::simd_from(simd, indices);
    let value: f64x4<S> = byte_vec.bitcast();
    let result_bytes: u8x32<S> = value.swizzle_dyn_within_blocks(index_vec).bitcast();

    assert_eq!(*result_bytes, expected);
}

#[simd_test]
fn swizzle_dyn_within_blocks_f32x16<S: Simd>(simd: S) {
    let bytes: [u8; 64] = core::array::from_fn(swizzle_test_byte);
    let indices: [u8; 64] = core::array::from_fn(swizzle_test_index);
    let expected = expected_swizzle_within_blocks(bytes, indices);

    let byte_vec = u8x64::simd_from(simd, bytes);
    let index_vec = u8x64::simd_from(simd, indices);
    let value: f32x16<S> = byte_vec.bitcast();
    let result_bytes: u8x64<S> = value.swizzle_dyn_within_blocks(index_vec).bitcast();

    assert_eq!(*result_bytes, expected);
}

#[simd_test]
fn swizzle_dyn_within_blocks_i8x64<S: Simd>(simd: S) {
    let bytes: [u8; 64] = core::array::from_fn(swizzle_test_byte);
    let indices: [u8; 64] = core::array::from_fn(swizzle_test_index);
    let expected = expected_swizzle_within_blocks(bytes, indices);

    let byte_vec = u8x64::simd_from(simd, bytes);
    let index_vec = u8x64::simd_from(simd, indices);
    let value: i8x64<S> = byte_vec.bitcast();
    let result_bytes: u8x64<S> = value.swizzle_dyn_within_blocks(index_vec).bitcast();

    assert_eq!(*result_bytes, expected);
}

#[simd_test]
fn swizzle_dyn_within_blocks_u8x64<S: Simd>(simd: S) {
    let bytes: [u8; 64] = core::array::from_fn(swizzle_test_byte);
    let indices: [u8; 64] = core::array::from_fn(swizzle_test_index);
    let expected = expected_swizzle_within_blocks(bytes, indices);

    let byte_vec = u8x64::simd_from(simd, bytes);
    let index_vec = u8x64::simd_from(simd, indices);
    let value: u8x64<S> = byte_vec.bitcast();
    let result_bytes: u8x64<S> = value.swizzle_dyn_within_blocks(index_vec).bitcast();

    assert_eq!(*result_bytes, expected);
}

#[simd_test]
fn swizzle_dyn_within_blocks_i16x32<S: Simd>(simd: S) {
    let bytes: [u8; 64] = core::array::from_fn(swizzle_test_byte);
    let indices: [u8; 64] = core::array::from_fn(swizzle_test_index);
    let expected = expected_swizzle_within_blocks(bytes, indices);

    let byte_vec = u8x64::simd_from(simd, bytes);
    let index_vec = u8x64::simd_from(simd, indices);
    let value: i16x32<S> = byte_vec.bitcast();
    let result_bytes: u8x64<S> = value.swizzle_dyn_within_blocks(index_vec).bitcast();

    assert_eq!(*result_bytes, expected);
}

#[simd_test]
fn swizzle_dyn_within_blocks_u16x32<S: Simd>(simd: S) {
    let bytes: [u8; 64] = core::array::from_fn(swizzle_test_byte);
    let indices: [u8; 64] = core::array::from_fn(swizzle_test_index);
    let expected = expected_swizzle_within_blocks(bytes, indices);

    let byte_vec = u8x64::simd_from(simd, bytes);
    let index_vec = u8x64::simd_from(simd, indices);
    let value: u16x32<S> = byte_vec.bitcast();
    let result_bytes: u8x64<S> = value.swizzle_dyn_within_blocks(index_vec).bitcast();

    assert_eq!(*result_bytes, expected);
}

#[simd_test]
fn swizzle_dyn_within_blocks_i32x16<S: Simd>(simd: S) {
    let bytes: [u8; 64] = core::array::from_fn(swizzle_test_byte);
    let indices: [u8; 64] = core::array::from_fn(swizzle_test_index);
    let expected = expected_swizzle_within_blocks(bytes, indices);

    let byte_vec = u8x64::simd_from(simd, bytes);
    let index_vec = u8x64::simd_from(simd, indices);
    let value: i32x16<S> = byte_vec.bitcast();
    let result_bytes: u8x64<S> = value.swizzle_dyn_within_blocks(index_vec).bitcast();

    assert_eq!(*result_bytes, expected);
}

#[simd_test]
fn swizzle_dyn_within_blocks_u32x16<S: Simd>(simd: S) {
    let bytes: [u8; 64] = core::array::from_fn(swizzle_test_byte);
    let indices: [u8; 64] = core::array::from_fn(swizzle_test_index);
    let expected = expected_swizzle_within_blocks(bytes, indices);

    let byte_vec = u8x64::simd_from(simd, bytes);
    let index_vec = u8x64::simd_from(simd, indices);
    let value: u32x16<S> = byte_vec.bitcast();
    let result_bytes: u8x64<S> = value.swizzle_dyn_within_blocks(index_vec).bitcast();

    assert_eq!(*result_bytes, expected);
}

#[simd_test]
fn swizzle_dyn_within_blocks_f64x8<S: Simd>(simd: S) {
    let bytes: [u8; 64] = core::array::from_fn(swizzle_test_byte);
    let indices: [u8; 64] = core::array::from_fn(swizzle_test_index);
    let expected = expected_swizzle_within_blocks(bytes, indices);

    let byte_vec = u8x64::simd_from(simd, bytes);
    let index_vec = u8x64::simd_from(simd, indices);
    let value: f64x8<S> = byte_vec.bitcast();
    let result_bytes: u8x64<S> = value.swizzle_dyn_within_blocks(index_vec).bitcast();

    assert_eq!(*result_bytes, expected);
}

#[simd_test]
fn swizzle_dyn_within_blocks_generic_indices<S: Simd>(simd: S) {
    #[inline(always)]
    fn do_swizzle<S: Simd, V: SimdBase<S>>(value: V, indices: V::Bytes) -> V {
        value.swizzle_dyn_within_blocks(indices)
    }

    let value = u8x16::from_fn(simd, |i| u8::try_from(i).unwrap());
    let indices = u8x16::from_fn(simd, |i| u8::try_from(15 - i).unwrap());
    let result = do_swizzle::<S, u8x16<S>>(value, indices);

    assert_eq!(
        *result,
        [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    );
}

#[simd_test]
fn swizzle_dyn_within_blocks_oob_does_not_panic<S: Simd>(simd: S) {
    let value = u8x16::from_fn(simd, |i| u8::try_from(i).unwrap());
    let indices = u8x16::simd_from(
        simd,
        [
            16, 17, 31, 0x80, 0x8f, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5,
        ],
    );
    let _ = value.swizzle_dyn_within_blocks(indices);
}

// Generated gap-fill coverage rows.

#[simd_test]
fn swizzle_dyn_within_blocks_i64x2<S: Simd>(simd: S) {
    let values: [i64; 2] = core::array::from_fn(|i| (i % 23) as i64 + 10_i64);
    let indices_values: [u8; 16] = core::array::from_fn(|i| (i % 16) as u8);
    let a = i64x2::from_slice(simd, &values);
    let indices = u8x16::from_slice(simd, &indices_values);
    let result = simd.swizzle_dyn_within_blocks_i64x2(a, indices);
    assert_eq!(result.as_slice(), values.as_slice());
}

#[simd_test]
fn swizzle_dyn_within_blocks_u64x2<S: Simd>(simd: S) {
    let values: [u64; 2] = core::array::from_fn(|i| (i % 23) as u64 + 10_u64);
    let indices_values: [u8; 16] = core::array::from_fn(|i| (i % 16) as u8);
    let a = u64x2::from_slice(simd, &values);
    let indices = u8x16::from_slice(simd, &indices_values);
    let result = simd.swizzle_dyn_within_blocks_u64x2(a, indices);
    assert_eq!(result.as_slice(), values.as_slice());
}

#[simd_test]
fn swizzle_dyn_within_blocks_i64x4<S: Simd>(simd: S) {
    let values: [i64; 4] = core::array::from_fn(|i| (i % 23) as i64 + 10_i64);
    let indices_values: [u8; 32] = core::array::from_fn(|i| (i % 16) as u8);
    let a = i64x4::from_slice(simd, &values);
    let indices = u8x32::from_slice(simd, &indices_values);
    let result = simd.swizzle_dyn_within_blocks_i64x4(a, indices);
    assert_eq!(result.as_slice(), values.as_slice());
}

#[simd_test]
fn swizzle_dyn_within_blocks_u64x4<S: Simd>(simd: S) {
    let values: [u64; 4] = core::array::from_fn(|i| (i % 23) as u64 + 10_u64);
    let indices_values: [u8; 32] = core::array::from_fn(|i| (i % 16) as u8);
    let a = u64x4::from_slice(simd, &values);
    let indices = u8x32::from_slice(simd, &indices_values);
    let result = simd.swizzle_dyn_within_blocks_u64x4(a, indices);
    assert_eq!(result.as_slice(), values.as_slice());
}

#[simd_test]
fn swizzle_dyn_within_blocks_i64x8<S: Simd>(simd: S) {
    let values: [i64; 8] = core::array::from_fn(|i| (i % 23) as i64 + 10_i64);
    let indices_values: [u8; 64] = core::array::from_fn(|i| (i % 16) as u8);
    let a = i64x8::from_slice(simd, &values);
    let indices = u8x64::from_slice(simd, &indices_values);
    let result = simd.swizzle_dyn_within_blocks_i64x8(a, indices);
    assert_eq!(result.as_slice(), values.as_slice());
}

#[simd_test]
fn swizzle_dyn_within_blocks_u64x8<S: Simd>(simd: S) {
    let values: [u64; 8] = core::array::from_fn(|i| (i % 23) as u64 + 10_u64);
    let indices_values: [u8; 64] = core::array::from_fn(|i| (i % 16) as u8);
    let a = u64x8::from_slice(simd, &values);
    let indices = u8x64::from_slice(simd, &indices_values);
    let result = simd.swizzle_dyn_within_blocks_u64x8(a, indices);
    assert_eq!(result.as_slice(), values.as_slice());
}
