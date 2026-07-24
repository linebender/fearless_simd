// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// Additional concrete rows for this operation.

#[simd_test]
fn from_bytes_i64x2<S: Simd>(simd: S) {
    let a = i64x2::from_slice(simd, &[1_i64, -2_i64]);
    let bytes = a.to_bytes();
    let roundtrip = i64x2::from_bytes(bytes);
    assert_eq!(roundtrip.as_slice(), a.as_slice());
}

#[simd_test]
fn from_bytes_i64x4<S: Simd>(simd: S) {
    let a = i64x4::from_slice(simd, &[1_i64, -2_i64, 3_i64, -4_i64]);
    let bytes = a.to_bytes();
    let roundtrip = i64x4::from_bytes(bytes);
    assert_eq!(roundtrip.as_slice(), a.as_slice());
}

#[simd_test]
fn from_bytes_i64x8<S: Simd>(simd: S) {
    let a = i64x8::from_slice(
        simd,
        &[1_i64, -2_i64, 3_i64, -4_i64, 5_i64, -6_i64, 7_i64, -8_i64],
    );
    let bytes = a.to_bytes();
    let roundtrip = i64x8::from_bytes(bytes);
    assert_eq!(roundtrip.as_slice(), a.as_slice());
}

#[simd_test]
fn from_bytes_u64x2<S: Simd>(simd: S) {
    let a = u64x2::from_slice(simd, &[1_u64, 2_u64]);
    let bytes = a.to_bytes();
    let roundtrip = u64x2::from_bytes(bytes);
    assert_eq!(roundtrip.as_slice(), a.as_slice());
}

#[simd_test]
fn from_bytes_u64x4<S: Simd>(simd: S) {
    let a = u64x4::from_slice(simd, &[1_u64, 2_u64, 3_u64, 4_u64]);
    let bytes = a.to_bytes();
    let roundtrip = u64x4::from_bytes(bytes);
    assert_eq!(roundtrip.as_slice(), a.as_slice());
}

#[simd_test]
fn from_bytes_u64x8<S: Simd>(simd: S) {
    let a = u64x8::from_slice(
        simd,
        &[1_u64, 2_u64, 3_u64, 4_u64, 5_u64, 6_u64, 7_u64, 8_u64],
    );
    let bytes = a.to_bytes();
    let roundtrip = u64x8::from_bytes(bytes);
    assert_eq!(roundtrip.as_slice(), a.as_slice());
}

// Generated gap-fill coverage rows.

#[simd_test]
fn from_bytes_i8x16<S: Simd>(simd: S) {
    let byte_values: [u8; 16] = core::array::from_fn(|i| i as u8);
    let bytes = u8x16::from_slice(simd, &byte_values);
    let result = simd.cvt_from_bytes_i8x16(bytes);
    let expected: Vec<i8> = byte_values
        .chunks_exact(size_of::<i8>())
        .map(|bytes| i8::from_ne_bytes(bytes.try_into().unwrap()))
        .collect();
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn from_bytes_u8x16<S: Simd>(simd: S) {
    let byte_values: [u8; 16] = core::array::from_fn(|i| i as u8);
    let bytes = u8x16::from_slice(simd, &byte_values);
    let result = simd.cvt_from_bytes_u8x16(bytes);
    let expected: Vec<u8> = byte_values
        .chunks_exact(size_of::<u8>())
        .map(|bytes| u8::from_ne_bytes(bytes.try_into().unwrap()))
        .collect();
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn from_bytes_i8x32<S: Simd>(simd: S) {
    let byte_values: [u8; 32] = core::array::from_fn(|i| i as u8);
    let bytes = u8x32::from_slice(simd, &byte_values);
    let result = simd.cvt_from_bytes_i8x32(bytes);
    let expected: Vec<i8> = byte_values
        .chunks_exact(size_of::<i8>())
        .map(|bytes| i8::from_ne_bytes(bytes.try_into().unwrap()))
        .collect();
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn from_bytes_u8x32<S: Simd>(simd: S) {
    let byte_values: [u8; 32] = core::array::from_fn(|i| i as u8);
    let bytes = u8x32::from_slice(simd, &byte_values);
    let result = simd.cvt_from_bytes_u8x32(bytes);
    let expected: Vec<u8> = byte_values
        .chunks_exact(size_of::<u8>())
        .map(|bytes| u8::from_ne_bytes(bytes.try_into().unwrap()))
        .collect();
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn from_bytes_i8x64<S: Simd>(simd: S) {
    let byte_values: [u8; 64] = core::array::from_fn(|i| i as u8);
    let bytes = u8x64::from_slice(simd, &byte_values);
    let result = simd.cvt_from_bytes_i8x64(bytes);
    let expected: Vec<i8> = byte_values
        .chunks_exact(size_of::<i8>())
        .map(|bytes| i8::from_ne_bytes(bytes.try_into().unwrap()))
        .collect();
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn from_bytes_u8x64<S: Simd>(simd: S) {
    let byte_values: [u8; 64] = core::array::from_fn(|i| i as u8);
    let bytes = u8x64::from_slice(simd, &byte_values);
    let result = simd.cvt_from_bytes_u8x64(bytes);
    let expected: Vec<u8> = byte_values
        .chunks_exact(size_of::<u8>())
        .map(|bytes| u8::from_ne_bytes(bytes.try_into().unwrap()))
        .collect();
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn from_bytes_i16x8<S: Simd>(simd: S) {
    let byte_values: [u8; 16] = core::array::from_fn(|i| i as u8);
    let bytes = u8x16::from_slice(simd, &byte_values);
    let result = simd.cvt_from_bytes_i16x8(bytes);
    let expected: Vec<i16> = byte_values
        .chunks_exact(size_of::<i16>())
        .map(|bytes| i16::from_ne_bytes(bytes.try_into().unwrap()))
        .collect();
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn from_bytes_u16x8<S: Simd>(simd: S) {
    let byte_values: [u8; 16] = core::array::from_fn(|i| i as u8);
    let bytes = u8x16::from_slice(simd, &byte_values);
    let result = simd.cvt_from_bytes_u16x8(bytes);
    let expected: Vec<u16> = byte_values
        .chunks_exact(size_of::<u16>())
        .map(|bytes| u16::from_ne_bytes(bytes.try_into().unwrap()))
        .collect();
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn from_bytes_i16x16<S: Simd>(simd: S) {
    let byte_values: [u8; 32] = core::array::from_fn(|i| i as u8);
    let bytes = u8x32::from_slice(simd, &byte_values);
    let result = simd.cvt_from_bytes_i16x16(bytes);
    let expected: Vec<i16> = byte_values
        .chunks_exact(size_of::<i16>())
        .map(|bytes| i16::from_ne_bytes(bytes.try_into().unwrap()))
        .collect();
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn from_bytes_u16x16<S: Simd>(simd: S) {
    let byte_values: [u8; 32] = core::array::from_fn(|i| i as u8);
    let bytes = u8x32::from_slice(simd, &byte_values);
    let result = simd.cvt_from_bytes_u16x16(bytes);
    let expected: Vec<u16> = byte_values
        .chunks_exact(size_of::<u16>())
        .map(|bytes| u16::from_ne_bytes(bytes.try_into().unwrap()))
        .collect();
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn from_bytes_i16x32<S: Simd>(simd: S) {
    let byte_values: [u8; 64] = core::array::from_fn(|i| i as u8);
    let bytes = u8x64::from_slice(simd, &byte_values);
    let result = simd.cvt_from_bytes_i16x32(bytes);
    let expected: Vec<i16> = byte_values
        .chunks_exact(size_of::<i16>())
        .map(|bytes| i16::from_ne_bytes(bytes.try_into().unwrap()))
        .collect();
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn from_bytes_u16x32<S: Simd>(simd: S) {
    let byte_values: [u8; 64] = core::array::from_fn(|i| i as u8);
    let bytes = u8x64::from_slice(simd, &byte_values);
    let result = simd.cvt_from_bytes_u16x32(bytes);
    let expected: Vec<u16> = byte_values
        .chunks_exact(size_of::<u16>())
        .map(|bytes| u16::from_ne_bytes(bytes.try_into().unwrap()))
        .collect();
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn from_bytes_f32x4<S: Simd>(simd: S) {
    let byte_values: [u8; 16] = core::array::from_fn(|i| i as u8);
    let bytes = u8x16::from_slice(simd, &byte_values);
    let result = simd.cvt_from_bytes_f32x4(bytes);
    let expected: Vec<f32> = byte_values
        .chunks_exact(size_of::<f32>())
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
        .collect();
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn from_bytes_i32x4<S: Simd>(simd: S) {
    let byte_values: [u8; 16] = core::array::from_fn(|i| i as u8);
    let bytes = u8x16::from_slice(simd, &byte_values);
    let result = simd.cvt_from_bytes_i32x4(bytes);
    let expected: Vec<i32> = byte_values
        .chunks_exact(size_of::<i32>())
        .map(|bytes| i32::from_ne_bytes(bytes.try_into().unwrap()))
        .collect();
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn from_bytes_u32x4<S: Simd>(simd: S) {
    let byte_values: [u8; 16] = core::array::from_fn(|i| i as u8);
    let bytes = u8x16::from_slice(simd, &byte_values);
    let result = simd.cvt_from_bytes_u32x4(bytes);
    let expected: Vec<u32> = byte_values
        .chunks_exact(size_of::<u32>())
        .map(|bytes| u32::from_ne_bytes(bytes.try_into().unwrap()))
        .collect();
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn from_bytes_f32x8<S: Simd>(simd: S) {
    let byte_values: [u8; 32] = core::array::from_fn(|i| i as u8);
    let bytes = u8x32::from_slice(simd, &byte_values);
    let result = simd.cvt_from_bytes_f32x8(bytes);
    let expected: Vec<f32> = byte_values
        .chunks_exact(size_of::<f32>())
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
        .collect();
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn from_bytes_i32x8<S: Simd>(simd: S) {
    let byte_values: [u8; 32] = core::array::from_fn(|i| i as u8);
    let bytes = u8x32::from_slice(simd, &byte_values);
    let result = simd.cvt_from_bytes_i32x8(bytes);
    let expected: Vec<i32> = byte_values
        .chunks_exact(size_of::<i32>())
        .map(|bytes| i32::from_ne_bytes(bytes.try_into().unwrap()))
        .collect();
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn from_bytes_u32x8<S: Simd>(simd: S) {
    let byte_values: [u8; 32] = core::array::from_fn(|i| i as u8);
    let bytes = u8x32::from_slice(simd, &byte_values);
    let result = simd.cvt_from_bytes_u32x8(bytes);
    let expected: Vec<u32> = byte_values
        .chunks_exact(size_of::<u32>())
        .map(|bytes| u32::from_ne_bytes(bytes.try_into().unwrap()))
        .collect();
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn from_bytes_f32x16<S: Simd>(simd: S) {
    let byte_values: [u8; 64] = core::array::from_fn(|i| i as u8);
    let bytes = u8x64::from_slice(simd, &byte_values);
    let result = simd.cvt_from_bytes_f32x16(bytes);
    let expected: Vec<f32> = byte_values
        .chunks_exact(size_of::<f32>())
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
        .collect();
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn from_bytes_i32x16<S: Simd>(simd: S) {
    let byte_values: [u8; 64] = core::array::from_fn(|i| i as u8);
    let bytes = u8x64::from_slice(simd, &byte_values);
    let result = simd.cvt_from_bytes_i32x16(bytes);
    let expected: Vec<i32> = byte_values
        .chunks_exact(size_of::<i32>())
        .map(|bytes| i32::from_ne_bytes(bytes.try_into().unwrap()))
        .collect();
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn from_bytes_u32x16<S: Simd>(simd: S) {
    let byte_values: [u8; 64] = core::array::from_fn(|i| i as u8);
    let bytes = u8x64::from_slice(simd, &byte_values);
    let result = simd.cvt_from_bytes_u32x16(bytes);
    let expected: Vec<u32> = byte_values
        .chunks_exact(size_of::<u32>())
        .map(|bytes| u32::from_ne_bytes(bytes.try_into().unwrap()))
        .collect();
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn from_bytes_f64x2<S: Simd>(simd: S) {
    let byte_values: [u8; 16] = core::array::from_fn(|i| i as u8);
    let bytes = u8x16::from_slice(simd, &byte_values);
    let result = simd.cvt_from_bytes_f64x2(bytes);
    let expected: Vec<f64> = byte_values
        .chunks_exact(size_of::<f64>())
        .map(|bytes| f64::from_ne_bytes(bytes.try_into().unwrap()))
        .collect();
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn from_bytes_f64x4<S: Simd>(simd: S) {
    let byte_values: [u8; 32] = core::array::from_fn(|i| i as u8);
    let bytes = u8x32::from_slice(simd, &byte_values);
    let result = simd.cvt_from_bytes_f64x4(bytes);
    let expected: Vec<f64> = byte_values
        .chunks_exact(size_of::<f64>())
        .map(|bytes| f64::from_ne_bytes(bytes.try_into().unwrap()))
        .collect();
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn from_bytes_f64x8<S: Simd>(simd: S) {
    let byte_values: [u8; 64] = core::array::from_fn(|i| i as u8);
    let bytes = u8x64::from_slice(simd, &byte_values);
    let result = simd.cvt_from_bytes_f64x8(bytes);
    let expected: Vec<f64> = byte_values
        .chunks_exact(size_of::<f64>())
        .map(|bytes| f64::from_ne_bytes(bytes.try_into().unwrap()))
        .collect();
    assert_eq!(result.as_slice(), expected.as_slice());
}
