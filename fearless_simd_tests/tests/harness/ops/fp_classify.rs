use fearless_simd::{f32x4, f64x4, mask32x4, mask64x4, prelude::*};
use fearless_simd_dev_macros::simd_test;

#[simd_test]
fn fp_classify_f32<S: Simd>(simd: S) {
    const VALUES: &[f32] = &[
        0.0,
        -0.0,
        1.0,
        -1.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(0xFFFF_FFFF), // -NaN
        f32::from_bits(0x7FFF_FFFF), // +NaN
        f32::MIN,
        f32::MAX,
        f32::MIN_POSITIVE,
        f32::MIN_POSITIVE.next_down(),
        f32::MIN_POSITIVE.next_up(),
        0.0_f32.next_up(),
        0.0_f32.next_down(),
        0.0_f32.next_down().next_down(),
    ];

    let classify = move |f: f32, func: fn(f32x4<S>) -> mask32x4<S>| -> bool {
        let v = f32x4::splat(simd, f);
        func(v).any_true()
    };

    for &f in VALUES {
        assert_eq!(f.is_nan(), classify(f, f32x4::is_nan), "f = {f}");
        assert_eq!(f.is_infinite(), classify(f, f32x4::is_infinite), "f = {f}");
        assert_eq!(f.is_finite(), classify(f, f32x4::is_finite), "f = {f}");
        assert_eq!(
            f.is_subnormal(),
            classify(f, f32x4::is_subnormal),
            "f = {f}"
        );
        assert_eq!(f.is_normal(), classify(f, f32x4::is_normal), "f = {f}");
        assert_eq!(
            f.is_sign_negative(),
            classify(f, f32x4::is_sign_negative),
            "f = {f}"
        );
        assert_eq!(
            f.is_sign_positive(),
            classify(f, f32x4::is_sign_positive),
            "f = {f}"
        );
    }
}
#[simd_test]
fn fp_classify_f64<S: Simd>(simd: S) {
    const VALUES: &[f64] = &[
        0.0,
        -0.0,
        1.0,
        -1.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::from_bits(0xFFFF_FFFF_FFFF_FFFF), // -NaN
        f64::from_bits(0x7FFF_FFFF_FFFF_FFFF), // +NaN
        f64::MIN,
        f64::MAX,
        f64::MIN_POSITIVE,
        f64::MIN_POSITIVE.next_down(),
        f64::MIN_POSITIVE.next_up(),
        0.0_f64.next_up(),
        0.0_f64.next_down(),
        0.0_f64.next_down().next_down(),
    ];

    let classify = move |f: f64, func: fn(f64x4<S>) -> mask64x4<S>| -> bool {
        let v = f64x4::splat(simd, f);
        func(v).any_true()
    };

    for &f in VALUES {
        assert_eq!(f.is_nan(), classify(f, f64x4::is_nan), "f = {f}");
        assert_eq!(f.is_infinite(), classify(f, f64x4::is_infinite), "f = {f}");
        assert_eq!(f.is_finite(), classify(f, f64x4::is_finite), "f = {f}");
        assert_eq!(
            f.is_subnormal(),
            classify(f, f64x4::is_subnormal),
            "f = {f}"
        );
        assert_eq!(f.is_normal(), classify(f, f64x4::is_normal), "f = {f}");
        assert_eq!(
            f.is_sign_negative(),
            classify(f, f64x4::is_sign_negative),
            "f = {f}"
        );
        assert_eq!(
            f.is_sign_positive(),
            classify(f, f64x4::is_sign_positive),
            "f = {f}"
        );
    }
}
