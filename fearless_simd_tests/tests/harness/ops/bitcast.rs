// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// One concrete test row per supported vector type.

#[simd_test]
fn bitcast_native<S: Simd>(simd: S) {
    let a_i32 = S::i32s::from_slice(simd, &vec![-1; S::i32s::N]);
    assert_eq!(
        a_i32.bitcast::<S::u32s>().as_slice(),
        &vec![u32::MAX; S::i32s::N]
    );
}

// Generated gap-fill coverage rows.

#[simd_test]
fn bitcast_i8x16<S: Simd>(simd: S) {
    let values: [i8; 16] = core::array::from_fn(|i| (i % 31) as i8 + 1_i8);
    let a = i8x16::from_slice(simd, &values);
    let casted = a.bitcast::<u8x16<S>>();
    let expected: [u8; 16] = values.map(|value| value as u8);
    assert_eq!(casted.as_slice(), expected.as_slice());
}

#[simd_test]
fn bitcast_u8x16<S: Simd>(simd: S) {
    let values: [u8; 16] = core::array::from_fn(|i| (i % 31) as u8 + 1_u8);
    let a = u8x16::from_slice(simd, &values);
    let casted = a.bitcast::<i8x16<S>>();
    let expected: [i8; 16] = values.map(|value| value as i8);
    assert_eq!(casted.as_slice(), expected.as_slice());
}

#[simd_test]
fn bitcast_i8x32<S: Simd>(simd: S) {
    let values: [i8; 32] = core::array::from_fn(|i| (i % 31) as i8 + 1_i8);
    let a = i8x32::from_slice(simd, &values);
    let casted = a.bitcast::<u8x32<S>>();
    let expected: [u8; 32] = values.map(|value| value as u8);
    assert_eq!(casted.as_slice(), expected.as_slice());
}

#[simd_test]
fn bitcast_u8x32<S: Simd>(simd: S) {
    let values: [u8; 32] = core::array::from_fn(|i| (i % 31) as u8 + 1_u8);
    let a = u8x32::from_slice(simd, &values);
    let casted = a.bitcast::<i8x32<S>>();
    let expected: [i8; 32] = values.map(|value| value as i8);
    assert_eq!(casted.as_slice(), expected.as_slice());
}

#[simd_test]
fn bitcast_i8x64<S: Simd>(simd: S) {
    let values: [i8; 64] = core::array::from_fn(|i| (i % 31) as i8 + 1_i8);
    let a = i8x64::from_slice(simd, &values);
    let casted = a.bitcast::<u8x64<S>>();
    let expected: [u8; 64] = values.map(|value| value as u8);
    assert_eq!(casted.as_slice(), expected.as_slice());
}

#[simd_test]
fn bitcast_u8x64<S: Simd>(simd: S) {
    let values: [u8; 64] = core::array::from_fn(|i| (i % 31) as u8 + 1_u8);
    let a = u8x64::from_slice(simd, &values);
    let casted = a.bitcast::<i8x64<S>>();
    let expected: [i8; 64] = values.map(|value| value as i8);
    assert_eq!(casted.as_slice(), expected.as_slice());
}

#[simd_test]
fn bitcast_i16x8<S: Simd>(simd: S) {
    let values: [i16; 8] = core::array::from_fn(|i| (i % 31) as i16 + 1_i16);
    let a = i16x8::from_slice(simd, &values);
    let casted = a.bitcast::<u8x16<S>>();
    let expected: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    assert_eq!(casted.as_slice(), expected.as_slice());
}

#[simd_test]
fn bitcast_u16x8<S: Simd>(simd: S) {
    let values: [u16; 8] = core::array::from_fn(|i| (i % 31) as u16 + 1_u16);
    let a = u16x8::from_slice(simd, &values);
    let casted = a.bitcast::<u8x16<S>>();
    let expected: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    assert_eq!(casted.as_slice(), expected.as_slice());
}

#[simd_test]
fn bitcast_i16x16<S: Simd>(simd: S) {
    let values: [i16; 16] = core::array::from_fn(|i| (i % 31) as i16 + 1_i16);
    let a = i16x16::from_slice(simd, &values);
    let casted = a.bitcast::<u8x32<S>>();
    let expected: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    assert_eq!(casted.as_slice(), expected.as_slice());
}

#[simd_test]
fn bitcast_u16x16<S: Simd>(simd: S) {
    let values: [u16; 16] = core::array::from_fn(|i| (i % 31) as u16 + 1_u16);
    let a = u16x16::from_slice(simd, &values);
    let casted = a.bitcast::<u8x32<S>>();
    let expected: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    assert_eq!(casted.as_slice(), expected.as_slice());
}

#[simd_test]
fn bitcast_i16x32<S: Simd>(simd: S) {
    let values: [i16; 32] = core::array::from_fn(|i| (i % 31) as i16 + 1_i16);
    let a = i16x32::from_slice(simd, &values);
    let casted = a.bitcast::<u8x64<S>>();
    let expected: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    assert_eq!(casted.as_slice(), expected.as_slice());
}

#[simd_test]
fn bitcast_u16x32<S: Simd>(simd: S) {
    let values: [u16; 32] = core::array::from_fn(|i| (i % 31) as u16 + 1_u16);
    let a = u16x32::from_slice(simd, &values);
    let casted = a.bitcast::<u8x64<S>>();
    let expected: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    assert_eq!(casted.as_slice(), expected.as_slice());
}

#[simd_test]
fn bitcast_f32x4<S: Simd>(simd: S) {
    let values: [f32; 4] = core::array::from_fn(|i| i as f32 + 1.0_f32);
    let a = f32x4::from_slice(simd, &values);
    let casted = a.bitcast::<u8x16<S>>();
    let expected: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    assert_eq!(casted.as_slice(), expected.as_slice());
}

#[simd_test]
fn bitcast_i32x4<S: Simd>(simd: S) {
    let values: [i32; 4] = core::array::from_fn(|i| (i % 31) as i32 + 1_i32);
    let a = i32x4::from_slice(simd, &values);
    let casted = a.bitcast::<u8x16<S>>();
    let expected: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    assert_eq!(casted.as_slice(), expected.as_slice());
}

#[simd_test]
fn bitcast_u32x4<S: Simd>(simd: S) {
    let values: [u32; 4] = core::array::from_fn(|i| (i % 31) as u32 + 1_u32);
    let a = u32x4::from_slice(simd, &values);
    let casted = a.bitcast::<u8x16<S>>();
    let expected: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    assert_eq!(casted.as_slice(), expected.as_slice());
}

#[simd_test]
fn bitcast_f32x8<S: Simd>(simd: S) {
    let values: [f32; 8] = core::array::from_fn(|i| i as f32 + 1.0_f32);
    let a = f32x8::from_slice(simd, &values);
    let casted = a.bitcast::<u8x32<S>>();
    let expected: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    assert_eq!(casted.as_slice(), expected.as_slice());
}

#[simd_test]
fn bitcast_i32x8<S: Simd>(simd: S) {
    let values: [i32; 8] = core::array::from_fn(|i| (i % 31) as i32 + 1_i32);
    let a = i32x8::from_slice(simd, &values);
    let casted = a.bitcast::<u8x32<S>>();
    let expected: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    assert_eq!(casted.as_slice(), expected.as_slice());
}

#[simd_test]
fn bitcast_u32x8<S: Simd>(simd: S) {
    let values: [u32; 8] = core::array::from_fn(|i| (i % 31) as u32 + 1_u32);
    let a = u32x8::from_slice(simd, &values);
    let casted = a.bitcast::<u8x32<S>>();
    let expected: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    assert_eq!(casted.as_slice(), expected.as_slice());
}

#[simd_test]
fn bitcast_f32x16<S: Simd>(simd: S) {
    let values: [f32; 16] = core::array::from_fn(|i| i as f32 + 1.0_f32);
    let a = f32x16::from_slice(simd, &values);
    let casted = a.bitcast::<u8x64<S>>();
    let expected: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    assert_eq!(casted.as_slice(), expected.as_slice());
}

#[simd_test]
fn bitcast_i32x16<S: Simd>(simd: S) {
    let values: [i32; 16] = core::array::from_fn(|i| (i % 31) as i32 + 1_i32);
    let a = i32x16::from_slice(simd, &values);
    let casted = a.bitcast::<u8x64<S>>();
    let expected: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    assert_eq!(casted.as_slice(), expected.as_slice());
}

#[simd_test]
fn bitcast_u32x16<S: Simd>(simd: S) {
    let values: [u32; 16] = core::array::from_fn(|i| (i % 31) as u32 + 1_u32);
    let a = u32x16::from_slice(simd, &values);
    let casted = a.bitcast::<u8x64<S>>();
    let expected: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    assert_eq!(casted.as_slice(), expected.as_slice());
}

#[simd_test]
fn bitcast_f64x2<S: Simd>(simd: S) {
    let values: [f64; 2] = core::array::from_fn(|i| i as f64 + 1.0_f64);
    let a = f64x2::from_slice(simd, &values);
    let casted = a.bitcast::<u8x16<S>>();
    let expected: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    assert_eq!(casted.as_slice(), expected.as_slice());
}

#[simd_test]
fn bitcast_i64x2<S: Simd>(simd: S) {
    let values: [i64; 2] = core::array::from_fn(|i| (i % 31) as i64 + 1_i64);
    let a = i64x2::from_slice(simd, &values);
    let casted = a.bitcast::<u8x16<S>>();
    let expected: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    assert_eq!(casted.as_slice(), expected.as_slice());
}

#[simd_test]
fn bitcast_u64x2<S: Simd>(simd: S) {
    let values: [u64; 2] = core::array::from_fn(|i| (i % 31) as u64 + 1_u64);
    let a = u64x2::from_slice(simd, &values);
    let casted = a.bitcast::<u8x16<S>>();
    let expected: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    assert_eq!(casted.as_slice(), expected.as_slice());
}

#[simd_test]
fn bitcast_f64x4<S: Simd>(simd: S) {
    let values: [f64; 4] = core::array::from_fn(|i| i as f64 + 1.0_f64);
    let a = f64x4::from_slice(simd, &values);
    let casted = a.bitcast::<u8x32<S>>();
    let expected: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    assert_eq!(casted.as_slice(), expected.as_slice());
}

#[simd_test]
fn bitcast_i64x4<S: Simd>(simd: S) {
    let values: [i64; 4] = core::array::from_fn(|i| (i % 31) as i64 + 1_i64);
    let a = i64x4::from_slice(simd, &values);
    let casted = a.bitcast::<u8x32<S>>();
    let expected: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    assert_eq!(casted.as_slice(), expected.as_slice());
}

#[simd_test]
fn bitcast_u64x4<S: Simd>(simd: S) {
    let values: [u64; 4] = core::array::from_fn(|i| (i % 31) as u64 + 1_u64);
    let a = u64x4::from_slice(simd, &values);
    let casted = a.bitcast::<u8x32<S>>();
    let expected: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    assert_eq!(casted.as_slice(), expected.as_slice());
}

#[simd_test]
fn bitcast_f64x8<S: Simd>(simd: S) {
    let values: [f64; 8] = core::array::from_fn(|i| i as f64 + 1.0_f64);
    let a = f64x8::from_slice(simd, &values);
    let casted = a.bitcast::<u8x64<S>>();
    let expected: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    assert_eq!(casted.as_slice(), expected.as_slice());
}

#[simd_test]
fn bitcast_i64x8<S: Simd>(simd: S) {
    let values: [i64; 8] = core::array::from_fn(|i| (i % 31) as i64 + 1_i64);
    let a = i64x8::from_slice(simd, &values);
    let casted = a.bitcast::<u8x64<S>>();
    let expected: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    assert_eq!(casted.as_slice(), expected.as_slice());
}

#[simd_test]
fn bitcast_u64x8<S: Simd>(simd: S) {
    let values: [u64; 8] = core::array::from_fn(|i| (i % 31) as u64 + 1_u64);
    let a = u64x8::from_slice(simd, &values);
    let casted = a.bitcast::<u8x64<S>>();
    let expected: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    assert_eq!(casted.as_slice(), expected.as_slice());
}
