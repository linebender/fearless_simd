// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use core::marker::PhantomData;

use crate::{Simd, SimdBase};

/// Rearrange a numeric SIMD vector's lanes using compile-time indices.
/// 
/// For swizzles with indices not known at compile time see
/// [swizzle_dyn](crate::SimdBase::swizzle_dyn).
///
/// The result has the same type and width as the input. The indices must be a
/// constant expression producing an array or slice of `usize`, with exactly one
/// index per output lane. Each index selects an input lane; duplicates are allowed.
/// Incorrect lengths and out-of-range indices fail to compile.
///
/// The vector expression is evaluated exactly once. No trait imports are needed
/// to invoke this macro.
///
/// ```
/// use fearless_simd::{prelude::*, simd_swizzle, u32x4};
///
/// fn permute<S: Simd>(value: u32x4<S>) -> u32x4<S> {
///     simd_swizzle!(value, [2, 0, 3, 1])
/// }
///
/// const REVERSE: [usize; 4] = [3, 2, 1, 0];
/// fn reverse<S: Simd>(value: u32x4<S>) -> u32x4<S> {
///     simd_swizzle!(value, REVERSE)
/// }
/// ```
///
/// Index expressions can reference enclosing generic parameters and call const
/// functions. For native-width vectors, use a constant table's prefix (or compute
/// a pattern for that width). `split_at` is usable in const expressions:
///
/// ```
/// use fearless_simd::{prelude::*, simd_swizzle};
///
/// const PAIRS: [usize; 16] = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14];
/// fn swap_pairs<S: Simd>(value: S::f32s) -> S::f32s {
///     simd_swizzle!(value, PAIRS.split_at(S::f32s::LEN).0)
/// }
///
/// const fn rotate<const OFFSET: usize>() -> [usize; 4] {
///     [OFFSET % 4, (OFFSET % 4 + 1) % 4, (OFFSET % 4 + 2) % 4, (OFFSET % 4 + 3) % 4]
/// }
/// fn rotate_four<S: Simd, const OFFSET: usize>(value: fearless_simd::f32x4<S>) -> fearless_simd::f32x4<S> {
///     simd_swizzle!(value, rotate::<OFFSET>())
/// }
/// ```
///
/// Each instantiated width must have valid indices. A prefix of a pattern valid
/// for a wider vector may still refer to lanes absent from a narrower vector.
#[macro_export]
macro_rules! simd_swizzle {
    ($value:expr, $indices:expr $(,)?) => {
        $crate::__simd_swizzle(
            $value,
            const {
                // Normalize arrays, array references, and slices using an operation
                // that is const on our MSRV (AsRef is not).
                $crate::__PreparedSwizzle::new(($indices).split_at(0).1)
            },
        )
    };
}

/// Implementation detail of [`crate::simd_swizzle!`].
/// Not part of the stable public API and may be changed at will.
#[derive(Debug)]
pub struct PreparedSwizzle<S: Simd, V: SimdBase<S>> {
    bytes: [u8; 64],
    marker: PhantomData<fn() -> (S, V)>,
}

impl<S: Simd, V: SimdBase<S>> PreparedSwizzle<S, V> {
    /// Validate lane indices and expand them into byte indices.
    pub const fn new(indices: &[usize]) -> Self {
        assert!(
            indices.len() == V::LEN,
            "swizzle index count must equal the vector lane count"
        );
        let lane_bytes = size_of::<V::Element>();
        // SimdBase is sealed; its implementations have nonzero lanes and elements.
        assert!(
            lane_bytes > 0 && V::LEN > 0 && V::LEN <= 64 / lane_bytes,
            "swizzle vector width must be between 1 and 64 bytes"
        );
        let mut bytes = [0; 64];
        let mut lane = 0;
        while lane < V::LEN {
            let index = indices[lane];
            assert!(
                index < V::LEN,
                "swizzle index must be less than the vector lane count"
            );
            let mut byte = 0;
            while byte < lane_bytes {
                // The width and lane-index checks above bound this value to 0..64.
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "validated byte index is below 64"
                )]
                let input_byte = (index * lane_bytes + byte) as u8;
                bytes[lane * lane_bytes + byte] = input_byte;
                byte += 1;
            }
            lane += 1;
        }
        Self {
            bytes,
            marker: PhantomData,
        }
    }
}

/// Apply a byte mask prepared by [`crate::simd_swizzle!`].
/// Not part of the stable public API and may be changed at will.
#[inline(always)]
pub fn swizzle<S: Simd, V: SimdBase<S>>(value: V, indices: PreparedSwizzle<S, V>) -> V {
    let byte_len = V::LEN * size_of::<V::Element>();
    let indices = V::ByteVector::from_slice(value.token(), &indices.bytes[..byte_len]);
    value.swizzle_dyn(indices)
}
