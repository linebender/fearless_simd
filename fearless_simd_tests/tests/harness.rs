use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

#[simd_test]
fn add_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[1.0, 2.0, 3.0, 4.0]);
    let b = f32x4::from_slice(simd, &[4.0, 3.0, 2.0, 1.0]);

    assert_eq!((a + b).val, [5.0, 5.0, 5.0, 5.0]);
}

#[simd_test]
fn sub_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[1.0, 2.0, 3.0, 4.0]);
    let b = f32x4::from_slice(simd, &[5.0, 4.0, 3.0, 2.0]);

    assert_eq!((a - b).val, [-4.0, -2.0, 0.0, 2.0]);
}

#[simd_test]
fn mul_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[1.0, 2.0, 3.0, 4.0]);
    let b = f32x4::from_slice(simd, &[5.0, 4.0, 3.0, 2.0]);

    assert_eq!((a * b).val, [5.0, 8.0, 9.0, 8.0]);
}

#[simd_test]
fn mul_u8x16<S: Simd>(simd: S) {
    let a = u8x16::from_slice(
        simd,
        &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    );
    let b = u8x16::from_slice(simd, &[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]);

    assert_eq!(
        (a * b).val,
        [0, 0, 0, 0, 4, 5, 6, 7, 16, 18, 20, 22, 36, 39, 42, 45]
    );
}

#[simd_test]
fn mul_i8x16<S: Simd>(simd: S) {
    let a = i8x16::from_slice(
        simd,
        &[0, -0, 3, -3, 0, -0, 3, -3, 0, -0, 3, -3, 0, -0, 3, -3],
    );
    let b = i8x16::from_slice(
        simd,
        &[0, 0, 0, 0, -0, -0, -0, -0, 3, 3, 3, 3, -3, -3, -3, -3],
    );

    assert_eq!(
        (a * b).val,
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, -9, 0, 0, -9, 9]
    );
}

#[simd_test]
fn splat_f32x4<S: Simd>(simd: S) {
    assert_eq!(f32x4::splat(simd, 1.0).val, [1.0, 1.0, 1.0, 1.0]);
}

#[simd_test]
fn abs_f32x4<S: Simd>(simd: S) {
    assert_eq!(
        f32x4::from_slice(simd, &[-1.0, 0., 1.0, 2.3]).abs().val,
        [1.0, 0.0, 1.0, 2.3]
    );
}

#[simd_test]
fn neg_f32x4<S: Simd>(simd: S) {
    assert_eq!(
        f32x4::from_slice(simd, &[-1.0, 0., 1.0, 2.3]).neg().val,
        [1.0, -0.0, -1.0, -2.3]
    );
}
