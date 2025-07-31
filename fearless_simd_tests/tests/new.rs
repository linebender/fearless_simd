use fearless_simd::{Simd, SimdBase, f32x4};
use fearless_simd_dev_macros::simd_test;

#[simd_test]
fn add_f32<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[1.0, 2.0, 3.0, 4.0]);
    let b = f32x4::from_slice(simd, &[4.0, 3.0, 2.0, 1.0]);

    assert_eq!((a + b).val, [5.0, 5.0, 5.0, 5.0])
}
