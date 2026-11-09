#![cfg_attr(not(feature = "std"), no_std)]
use simd_research_kernels::{mix as scramble, opaque_mix, ordinary};
use simd_research_macros::simd;

#[simd]
pub fn process(xs: &mut [u32]) {
    for x in xs {
        *x = ordinary(scramble(*x));
    }
}

#[simd]
pub fn process_opaque(xs: &mut [u32]) {
    for x in xs {
        *x = opaque_mix(*x);
    }
}

#[simd]
pub fn recursive(n: u32) -> u32 {
    if n == 0 {
        1
    } else {
        n.wrapping_mul(recursive(n - 1))
    }
}

#[simd]
pub fn local_callable(x: u32) -> u32 {
    let scramble = |v: u32| v + 1;
    scramble(x)
}

#[simd]
pub fn borrowed(x: &str) -> &str {
    x
}

#[simd]
pub fn borrowing_caller(x: &str) -> &str {
    borrowed(x)
}

// Safe concrete-token boundaries for assembly comparisons. Target-feature
// attributes are provided exclusively by the public kernel! macro.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86_codegen {
    use super::*;
    use simd_research_runtime::fearless_simd;

    fearless_simd::kernel! {
        #[unsafe(no_mangle)]
        pub fn automatic_avx2(simd: Avx2, xs: &mut [u32]) {
            process::__call(simd, process, xs)
        }
    }

    fearless_simd::kernel! {
        #[unsafe(no_mangle)]
        pub fn manual_avx2(simd: Avx2, xs: &mut [u32]) {
            for x in xs {
                *x = x.wrapping_mul(0x9e3779b9).rotate_left(7).wrapping_add(13);
            }
        }
    }

    fearless_simd::kernel! {
        #[unsafe(no_mangle)]
        pub fn automatic_opaque_avx2(simd: Avx2, xs: &mut [u32]) {
            process_opaque::__call(simd, process_opaque, xs)
        }
    }

    fearless_simd::kernel! {
        #[unsafe(no_mangle)]
        pub fn manual_opaque_avx2(simd: Avx2, xs: &mut [u32]) {
            for x in xs {
                *x = opaque_mix::__call(simd, opaque_mix, *x);
            }
        }
    }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use x86_codegen::*;

#[unsafe(no_mangle)]
pub fn public_dispatch(xs: &mut [u32]) {
    process(xs)
}

#[unsafe(no_mangle)]
pub fn unannotated_loop(xs: &mut [u32]) {
    for x in xs {
        *x = scramble(*x);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cross_crate_alias_and_ordinary_fallback() {
        let original: Vec<u32> = (0..1027).map(|x: u32| x.wrapping_mul(0xfeedbeef)).collect();
        let expected: Vec<u32> = original
            .iter()
            .map(|x| x.wrapping_mul(0x9e3779b9).rotate_left(7).wrapping_add(13))
            .collect();
        let mut dynamic = original.clone();
        process(&mut dynamic);
        assert_eq!(dynamic, expected);
        let mut baseline = original.clone();
        use simd_research_runtime::fearless_simd::{Level, dispatch};
        dispatch!(Level::baseline(), simd => process::__call(simd, process, &mut baseline));
        assert_eq!(baseline, expected);
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if let Some(simd) = Level::new().as_avx2() {
            let mut specialized = original.clone();
            automatic_avx2(simd, &mut specialized);
            assert_eq!(specialized, expected);
            let mut manual = original;
            manual_avx2(simd, &mut manual);
            assert_eq!(specialized, manual);
        }
    }

    #[test]
    fn empty_and_short_slices() {
        process(&mut []);
        let mut values = [u32::MAX];
        process(&mut values);
        assert_eq!(
            values,
            [u32::MAX
                .wrapping_mul(0x9e3779b9)
                .rotate_left(7)
                .wrapping_add(13)]
        );
    }

    #[test]
    fn recursion_locals_and_borrows() {
        assert_eq!(recursive(5), 120);
        assert_eq!(local_callable(41), 42);
        let owned = String::from("borrowed value");
        assert_eq!(borrowing_caller(&owned), owned.as_str());
    }

    #[test]
    fn non_inlined_helper() {
        let mut values = [0, 1, u32::MAX, 0xfeedbeef];
        let expected = values.map(|x: u32| x.wrapping_mul(0x9e3779b9).rotate_left(7));
        process_opaque(&mut values);
        assert_eq!(values, expected);
    }
}
