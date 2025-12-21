// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Exhaustive tests for the "slide" operations.

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

/// Helper macro for testing individual slide operations
macro_rules! test_vector_slide {
    ($test_vec_a:expr, $test_vec_b:expr, $fallback_vec_a:expr, $fallback_vec_b:expr, $vec_ty:ident, $shift:literal) => {{
        #[inline(never)]
        fn do_test<S: Simd>(
            test_vec_a: $vec_ty<S>,
            test_vec_b: $vec_ty<S>,
            fallback_vec_a: $vec_ty<fearless_simd::Fallback>,
            fallback_vec_b: $vec_ty<fearless_simd::Fallback>,
        ) {
            assert_eq!(
                core::hint::black_box(
                    test_vec_a
                        .witness()
                        .vectorize(|| test_vec_a.slide::<$shift>(test_vec_b))
                        .as_slice()
                ),
                core::hint::black_box(fallback_vec_a.slide::<$shift>(fallback_vec_b).as_slice()),
                "slide::<{}> mismatch",
                $shift
            );
        }

        do_test($test_vec_a, $test_vec_b, $fallback_vec_a, $fallback_vec_b)
    }};
}

macro_rules! test_block_slide {
    ($test_vec_a:expr, $test_vec_b:expr, $fallback_vec_a:expr, $fallback_vec_b:expr, $vec_ty:ident, $shift:literal) => {{
        // A bunch of weird stuff here is to prevent rustc/LLVM from inlining everything and generating enormous amounts
        // of code. Since all these tests are run one after the other in macro-generated code, we want to provide some
        // level of isolation between them to prevent the compiler from trying to optimize a huge function containing
        // all of them. So, each test is run in its own `#[inline(never)]` helper function. The `black_box` is to
        // prevent the compiler from unrolling the slice comparison into (up to 64!) individual inline comparisons, and
        // forcing a memcmp instead.
        #[inline(never)]
        fn do_test<S: Simd>(
            test_vec_a: $vec_ty<S>,
            test_vec_b: $vec_ty<S>,
            fallback_vec_a: $vec_ty<fearless_simd::Fallback>,
            fallback_vec_b: $vec_ty<fearless_simd::Fallback>,
        ) {
            assert_eq!(
                core::hint::black_box(
                    test_vec_a
                        .witness()
                        .vectorize(|| test_vec_a.slide_within_blocks::<$shift>(test_vec_b))
                        .as_slice()
                ),
                core::hint::black_box(
                    fallback_vec_a
                        .slide_within_blocks::<$shift>(fallback_vec_b)
                        .as_slice()
                ),
                "slide_within_blocks::<{}> mismatch",
                $shift
            );
        }

        do_test($test_vec_a, $test_vec_b, $fallback_vec_a, $fallback_vec_b)
    }};
}

/// Macro to iterate over shift values for slide (0 to N)
/// For slide, valid shifts are 0..=N where N is the number of elements
macro_rules! for_each_slide {
    // For 2-element vectors: shifts 0..=2
    (@n2 $callback:ident!($($args:tt)*)) => {
        $callback!($($args)* 0);
        $callback!($($args)* 1);
        $callback!($($args)* 2);
    };
    // For 4-element vectors: shifts 0..=4
    (@n4 $callback:ident!($($args:tt)*)) => {
        for_each_slide!(@n2 $callback!($($args)*));
        $callback!($($args)* 3);
        $callback!($($args)* 4);
    };
    // For 8-element vectors: shifts 0..=8
    (@n8 $callback:ident!($($args:tt)*)) => {
        for_each_slide!(@n4 $callback!($($args)*));
        $callback!($($args)* 5);
        $callback!($($args)* 6);
        $callback!($($args)* 7);
        $callback!($($args)* 8);
    };
    // For 16-element vectors: shifts 0..=16
    (@n16 $callback:ident!($($args:tt)*)) => {
        for_each_slide!(@n8 $callback!($($args)*));
        $callback!($($args)* 9);
        $callback!($($args)* 10);
        $callback!($($args)* 11);
        $callback!($($args)* 12);
        $callback!($($args)* 13);
        $callback!($($args)* 14);
        $callback!($($args)* 15);
        $callback!($($args)* 16);
    };
    // For 32-element vectors: shifts 0..=32
    (@n32 $callback:ident!($($args:tt)*)) => {
        for_each_slide!(@n16 $callback!($($args)*));
        $callback!($($args)* 17);
        $callback!($($args)* 18);
        $callback!($($args)* 19);
        $callback!($($args)* 20);
        $callback!($($args)* 21);
        $callback!($($args)* 22);
        $callback!($($args)* 23);
        $callback!($($args)* 24);
        $callback!($($args)* 25);
        $callback!($($args)* 26);
        $callback!($($args)* 27);
        $callback!($($args)* 28);
        $callback!($($args)* 29);
        $callback!($($args)* 30);
        $callback!($($args)* 31);
        $callback!($($args)* 32);
    };
    // For 64-element vectors: shifts 0..=64
    (@n64 $callback:ident!($($args:tt)*)) => {
        for_each_slide!(@n32 $callback!($($args)*));
        $callback!($($args)* 33);
        $callback!($($args)* 34);
        $callback!($($args)* 35);
        $callback!($($args)* 36);
        $callback!($($args)* 37);
        $callback!($($args)* 38);
        $callback!($($args)* 39);
        $callback!($($args)* 40);
        $callback!($($args)* 41);
        $callback!($($args)* 42);
        $callback!($($args)* 43);
        $callback!($($args)* 44);
        $callback!($($args)* 45);
        $callback!($($args)* 46);
        $callback!($($args)* 47);
        $callback!($($args)* 48);
        $callback!($($args)* 49);
        $callback!($($args)* 50);
        $callback!($($args)* 51);
        $callback!($($args)* 52);
        $callback!($($args)* 53);
        $callback!($($args)* 54);
        $callback!($($args)* 55);
        $callback!($($args)* 56);
        $callback!($($args)* 57);
        $callback!($($args)* 58);
        $callback!($($args)* 59);
        $callback!($($args)* 60);
        $callback!($($args)* 61);
        $callback!($($args)* 62);
        $callback!($($args)* 63);
        $callback!($($args)* 64);
    };
}

/// Main macro for testing slide operations
macro_rules! test_slide_impl {
    // Vector-wide operations
    (@vec2 $test_vec_a:expr, $test_vec_b:expr, $fallback_vec_a:expr, $fallback_vec_b:expr, $vec_ty:ident) => {
        for_each_slide!(@n2 test_vector_slide!($test_vec_a, $test_vec_b, $fallback_vec_a, $fallback_vec_b, $vec_ty,));
    };
    (@vec4 $test_vec_a:expr, $test_vec_b:expr, $fallback_vec_a:expr, $fallback_vec_b:expr, $vec_ty:ident) => {
        for_each_slide!(@n4 test_vector_slide!($test_vec_a, $test_vec_b, $fallback_vec_a, $fallback_vec_b, $vec_ty,));
    };
    (@vec8 $test_vec_a:expr, $test_vec_b:expr, $fallback_vec_a:expr, $fallback_vec_b:expr, $vec_ty:ident) => {
        for_each_slide!(@n8 test_vector_slide!($test_vec_a, $test_vec_b, $fallback_vec_a, $fallback_vec_b, $vec_ty,));
    };
    (@vec16 $test_vec_a:expr, $test_vec_b:expr, $fallback_vec_a:expr, $fallback_vec_b:expr, $vec_ty:ident) => {
        for_each_slide!(@n16 test_vector_slide!($test_vec_a, $test_vec_b, $fallback_vec_a, $fallback_vec_b, $vec_ty,));
    };
    (@vec32 $test_vec_a:expr, $test_vec_b:expr, $fallback_vec_a:expr, $fallback_vec_b:expr, $vec_ty:ident) => {
        for_each_slide!(@n32 test_vector_slide!($test_vec_a, $test_vec_b, $fallback_vec_a, $fallback_vec_b, $vec_ty,));
    };
    (@vec64 $test_vec_a:expr, $test_vec_b:expr, $fallback_vec_a:expr, $fallback_vec_b:expr, $vec_ty:ident) => {
        for_each_slide!(@n64 test_vector_slide!($test_vec_a, $test_vec_b, $fallback_vec_a, $fallback_vec_b, $vec_ty,));
    };

    // Within-block operations
    (@block2 $test_vec_a:expr, $test_vec_b:expr, $fallback_vec_a:expr, $fallback_vec_b:expr, $vec_ty:ident) => {
        for_each_slide!(@n2 test_block_slide!($test_vec_a, $test_vec_b, $fallback_vec_a, $fallback_vec_b, $vec_ty,));
    };
    (@block4 $test_vec_a:expr, $test_vec_b:expr, $fallback_vec_a:expr, $fallback_vec_b:expr, $vec_ty:ident) => {
        for_each_slide!(@n4 test_block_slide!($test_vec_a, $test_vec_b, $fallback_vec_a, $fallback_vec_b, $vec_ty,));
    };
    (@block8 $test_vec_a:expr, $test_vec_b:expr, $fallback_vec_a:expr, $fallback_vec_b:expr, $vec_ty:ident) => {
        for_each_slide!(@n8 test_block_slide!($test_vec_a, $test_vec_b, $fallback_vec_a, $fallback_vec_b, $vec_ty,));
    };
    (@block16 $test_vec_a:expr, $test_vec_b:expr, $fallback_vec_a:expr, $fallback_vec_b:expr, $vec_ty:ident) => {
        for_each_slide!(@n16 test_block_slide!($test_vec_a, $test_vec_b, $fallback_vec_a, $fallback_vec_b, $vec_ty,));
    };
}

/// Generate a test function for slide exhaustive testing
macro_rules! test_slide_exhaustive {
    ($test_name:ident, $vec_ty:ident, $elem_ty:ty, $n_elems:literal, $vec_n:ident, $block_n:ident) => {
        #[simd_test]
        fn $test_name<S: Simd>(simd: S) {
            let fallback = fearless_simd::Fallback::new();

            let vals_a: [$elem_ty; $n_elems] = core::hint::black_box(core::array::from_fn(|i| (i + 1) as $elem_ty));
            let vals_b: [$elem_ty; $n_elems] = core::hint::black_box(core::array::from_fn(|i| (i + 1 + $n_elems) as $elem_ty));

            let test_vec_a = $vec_ty::from_slice(simd, &vals_a);
            let test_vec_b = $vec_ty::from_slice(simd, &vals_b);
            let fallback_vec_a = <$vec_ty::<fearless_simd::Fallback>>::from_slice(fallback, &vals_a);
            let fallback_vec_b = <$vec_ty::<fearless_simd::Fallback>>::from_slice(fallback, &vals_b);

            // Test vector-wide operations
            test_slide_impl!(@$vec_n test_vec_a, test_vec_b, fallback_vec_a, fallback_vec_b, $vec_ty);
            // Test within-block operations
            test_slide_impl!(@$block_n test_vec_a, test_vec_b, fallback_vec_a, fallback_vec_b, $vec_ty);
        }
    };
}

// 128-bit vectors (block size == vector size, so within_blocks uses same range as vector-wide)
test_slide_exhaustive!(slide_exhaustive_f32x4, f32x4, f32, 4, vec4, block4);
test_slide_exhaustive!(slide_exhaustive_f64x2, f64x2, f64, 2, vec2, block2);
test_slide_exhaustive!(slide_exhaustive_i8x16, i8x16, i8, 16, vec16, block16);
test_slide_exhaustive!(slide_exhaustive_u8x16, u8x16, u8, 16, vec16, block16);
test_slide_exhaustive!(slide_exhaustive_i16x8, i16x8, i16, 8, vec8, block8);
test_slide_exhaustive!(slide_exhaustive_u16x8, u16x8, u16, 8, vec8, block8);
test_slide_exhaustive!(slide_exhaustive_i32x4, i32x4, i32, 4, vec4, block4);
test_slide_exhaustive!(slide_exhaustive_u32x4, u32x4, u32, 4, vec4, block4);

// 256-bit vectors (block size = 128 bits = half the vector size)
test_slide_exhaustive!(slide_exhaustive_f32x8, f32x8, f32, 8, vec8, block4);
test_slide_exhaustive!(slide_exhaustive_f64x4, f64x4, f64, 4, vec4, block2);
test_slide_exhaustive!(slide_exhaustive_i8x32, i8x32, i8, 32, vec32, block16);
test_slide_exhaustive!(slide_exhaustive_u8x32, u8x32, u8, 32, vec32, block16);
test_slide_exhaustive!(slide_exhaustive_i16x16, i16x16, i16, 16, vec16, block8);
test_slide_exhaustive!(slide_exhaustive_u16x16, u16x16, u16, 16, vec16, block8);
test_slide_exhaustive!(slide_exhaustive_i32x8, i32x8, i32, 8, vec8, block4);
test_slide_exhaustive!(slide_exhaustive_u32x8, u32x8, u32, 8, vec8, block4);

// 512-bit vectors (block size = 128 bits = quarter the vector size)
test_slide_exhaustive!(slide_exhaustive_f32x16, f32x16, f32, 16, vec16, block4);
test_slide_exhaustive!(slide_exhaustive_f64x8, f64x8, f64, 8, vec8, block2);
test_slide_exhaustive!(slide_exhaustive_i8x64, i8x64, i8, 64, vec64, block16);
test_slide_exhaustive!(slide_exhaustive_u8x64, u8x64, u8, 64, vec64, block16);
test_slide_exhaustive!(slide_exhaustive_i16x32, i16x32, i16, 32, vec32, block8);
test_slide_exhaustive!(slide_exhaustive_u16x32, u16x32, u16, 32, vec32, block8);
test_slide_exhaustive!(slide_exhaustive_i32x16, i32x16, i32, 16, vec16, block4);
test_slide_exhaustive!(slide_exhaustive_u32x16, u32x16, u32, 16, vec16, block4);

// Mask types (128-bit)
test_slide_exhaustive!(slide_exhaustive_mask8x16, mask8x16, i8, 16, vec16, block16);
test_slide_exhaustive!(slide_exhaustive_mask16x8, mask16x8, i16, 8, vec8, block8);
test_slide_exhaustive!(slide_exhaustive_mask32x4, mask32x4, i32, 4, vec4, block4);
test_slide_exhaustive!(slide_exhaustive_mask64x2, mask64x2, i64, 2, vec2, block2);

// Mask types (256-bit)
test_slide_exhaustive!(slide_exhaustive_mask8x32, mask8x32, i8, 32, vec32, block16);
test_slide_exhaustive!(
    slide_exhaustive_mask16x16,
    mask16x16,
    i16,
    16,
    vec16,
    block8
);
test_slide_exhaustive!(slide_exhaustive_mask32x8, mask32x8, i32, 8, vec8, block4);
test_slide_exhaustive!(slide_exhaustive_mask64x4, mask64x4, i64, 4, vec4, block2);

// Mask types (512-bit)
test_slide_exhaustive!(slide_exhaustive_mask8x64, mask8x64, i8, 64, vec64, block16);
test_slide_exhaustive!(
    slide_exhaustive_mask16x32,
    mask16x32,
    i16,
    32,
    vec32,
    block8
);
test_slide_exhaustive!(
    slide_exhaustive_mask32x16,
    mask32x16,
    i32,
    16,
    vec16,
    block4
);
test_slide_exhaustive!(slide_exhaustive_mask64x8, mask64x8, i64, 8, vec8, block2);
