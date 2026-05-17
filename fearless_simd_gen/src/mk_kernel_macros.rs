// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::{level::Level, mk_neon::Neon, mk_wasm::WasmSimd128, mk_x86::X86};

/// This emits a String rather than a TokenStream
/// because rustfmt just gives up formatting macros
/// and we end up with a completely unreadable token soup
/// if we don't impose formatting on it manually.
pub(crate) fn mk_kernel_macros() -> String {
    [
        kernel_macro(&Neon),
        kernel_macro(&WasmSimd128),
        kernel_macro(&X86::Sse4_2),
        kernel_macro(&X86::Avx2),
    ]
    .join("\n")
}

fn kernel_macro(level: &dyn Level) -> String {
    let macro_name = format!("{}_kernel", snake_case(level.name()));
    let name = level.name();
    let cfg = level
        .availability_cfg()
        .expect("kernel macros should only be generated for cfg-gated SIMD levels");
    let body = kernel_body(level);
    let target_feature_doc = target_feature_doc(level);
    let example_doc = example_doc(level);

    KERNEL_MACRO_TEMPLATE
        .replace("@MACRO_NAME@", &macro_name)
        .replace("@LEVEL_NAME@", name)
        .replace("@CFG@", cfg)
        .replace("@BODY@", &body)
        .replace("@TARGET_FEATURE_DOC@", &target_feature_doc)
        .replace("@EXAMPLE_DOC@", &example_doc)
}

fn kernel_body(level: &dyn Level) -> String {
    if let Some(features) = level.enabled_target_features() {
        KERNEL_BODY_WITH_TARGET_FEATURES
            .replace("@FEATURES@", features)
            .replace("@LEVEL_NAME@", level.name())
    } else {
        KERNEL_BODY.to_string()
    }
}

fn target_feature_doc(level: &dyn Level) -> String {
    let body = if level.enabled_target_features().is_some() {
        r#"
#[doc = "The generated wrapper takes a SIMD token (`@LEVEL_NAME@`) as its first argument."]
#[doc = "The macro runs your body inside an inner function annotated with the appropriate"]
#[doc = "`#[target_feature]` attributes. That makes platform-specific intrinsics from `core::arch` or"]
#[doc = "`std::arch` safe to call in the body, as long as they do not have safety"]
#[doc = "requirements beyond those target features."]
"#
    } else {
        r#"
#[doc = "The generated wrapper takes a SIMD token (`@LEVEL_NAME@`) as its first argument and is"]
#[doc = "compiled only when the required target features are enabled. That makes matching"]
#[doc = "platform-specific intrinsics from `core::arch` or `std::arch` safe to call in the"]
#[doc = "body, as long as they do not have safety requirements beyond those target features."]
"#
    };

    body.replace("@LEVEL_NAME@", level.name())
}

fn example_doc(level: &dyn Level) -> String {
    let example = match level.name() {
        "Neon" => {
            r#"
## Example

```rust
#[cfg(target_arch = "aarch64")]
use fearless_simd::{f32x4, prelude::*};
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{float32x4_t, vaddq_f32};

#[cfg(target_arch = "aarch64")]
fearless_simd::neon_kernel! {
    fn add_f32x4(a: float32x4_t, b: float32x4_t) -> float32x4_t {
        vaddq_f32(a, b)
    }
}

# fn main() {
#[cfg(target_arch = "aarch64")]
if let Some(neon) = fearless_simd::Level::new().as_neon() {
    let a: f32x4<_> = [1.0, 2.0, 3.0, 4.0].simd_into(neon);
    let b: f32x4<_> = [10.0, 20.0, 30.0, 40.0].simd_into(neon);
    let sum: f32x4<_> = add_f32x4(neon, a.into(), b.into()).simd_into(neon);

    assert_eq!(<[f32; 4]>::from(sum), [11.0, 22.0, 33.0, 44.0]);
}
# }
```
"#
        }
        "WasmSimd128" => {
            r#"
## Example

```rust
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use fearless_simd::{f32x4, prelude::*};
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use std::arch::wasm32::{f32x4_add, v128};

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fearless_simd::wasm_simd128_kernel! {
    fn add_f32x4(a: v128, b: v128) -> v128 {
        f32x4_add(a, b)
    }
}

# fn main() {
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
{
    let wasm = fearless_simd::Level::new()
        .as_wasm_simd128()
        .expect("simd128 is statically enabled");
    let a: f32x4<_> = [1.0, 2.0, 3.0, 4.0].simd_into(wasm);
    let b: f32x4<_> = [10.0, 20.0, 30.0, 40.0].simd_into(wasm);
    let sum: f32x4<_> = add_f32x4(wasm, a.into(), b.into()).simd_into(wasm);

    assert_eq!(<[f32; 4]>::from(sum), [11.0, 22.0, 33.0, 44.0]);
}
# }
```
"#
        }
        "Sse4_2" => {
            r#"
## Example

```rust
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use fearless_simd::{f32x4, prelude::*};
#[cfg(target_arch = "x86")]
use std::arch::x86::{__m128, _mm_add_ps};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{__m128, _mm_add_ps};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fearless_simd::sse4_2_kernel! {
    fn add_f32x4(a: __m128, b: __m128) -> __m128 {
        _mm_add_ps(a, b)
    }
}

# fn main() {
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
if let Some(sse4_2) = fearless_simd::Level::new().as_sse4_2() {
    let a: f32x4<_> = [1.0, 2.0, 3.0, 4.0].simd_into(sse4_2);
    let b: f32x4<_> = [10.0, 20.0, 30.0, 40.0].simd_into(sse4_2);
    let sum: f32x4<_> = add_f32x4(sse4_2, a.into(), b.into()).simd_into(sse4_2);

    assert_eq!(<[f32; 4]>::from(sum), [11.0, 22.0, 33.0, 44.0]);
}
# }
```
"#
        }
        "Avx2" => {
            r#"
## Example

```rust
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use fearless_simd::{i32x8, prelude::*};
#[cfg(target_arch = "x86")]
use std::arch::x86::{__m256i, _mm256_add_epi32};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{__m256i, _mm256_add_epi32};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fearless_simd::avx2_kernel! {
    fn add_i32x8(a: __m256i, b: __m256i) -> __m256i {
        _mm256_add_epi32(a, b)
    }
}

# fn main() {
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
if let Some(avx2) = fearless_simd::Level::new().as_avx2() {
    let a: i32x8<_> = [1, 2, 3, 4, 5, 6, 7, 8].simd_into(avx2);
    let b: i32x8<_> = [10, 20, 30, 40, 50, 60, 70, 80].simd_into(avx2);
    let sum: i32x8<_> = add_i32x8(avx2, a.into(), b.into()).simd_into(avx2);

    assert_eq!(<[i32; 8]>::from(sum), [11, 22, 33, 44, 55, 66, 77, 88]);
}
# }
```
"#
        }
        _ => unreachable!("kernel macros are only generated for known SIMD levels"),
    };

    doc_block(example)
}

fn doc_block(markdown: &str) -> String {
    markdown
        .trim_matches('\n')
        .lines()
        .map(|line| {
            format!(
                r#"#[doc = "{}"]"#,
                line.replace('\\', "\\\\").replace('"', "\\\"")
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn snake_case(name: &str) -> String {
    let mut result = String::new();
    let mut prev_was_lowercase = false;
    for ch in name.chars() {
        if ch == '_' {
            result.push(ch);
            prev_was_lowercase = false;
        } else if ch.is_uppercase() {
            if prev_was_lowercase {
                result.push('_');
            }
            result.extend(ch.to_lowercase());
            prev_was_lowercase = false;
        } else {
            result.push(ch);
            prev_was_lowercase = ch.is_lowercase();
        }
    }
    result
}

const KERNEL_MACRO_TEMPLATE: &str = r#"
#[doc = "Creates a context where you can safely call intrinsics for `@LEVEL_NAME@`."]
#[doc = ""]
#[doc = "This is useful if the portable abstractions are not enough, and you need to"]
#[doc = "use platform-specific intrinsics for parts of the computation."]
#[doc = ""]
@TARGET_FEATURE_DOC@
#[doc = ""]
@EXAMPLE_DOC@
#[doc = ""]
#[doc = "See the [sRGB example] for an end-to-end use of kernel macros."]
#[doc = ""]
#[doc = "[sRGB example]: https://github.com/linebender/fearless_simd/blob/main/fearless_simd/examples/srgb.rs"]
#[doc = ""]
#[doc = "Kernel macros only accept safe functions."]
#[doc = ""]
#[doc = "```compile_fail"]
#[doc = "fearless_simd::@MACRO_NAME@! {"]
#[doc = "    unsafe fn should_not_compile() {}"]
#[doc = "}"]
#[doc = "```"]
#[macro_export]
macro_rules! @MACRO_NAME@ {
    (
        $(#[$meta:meta])*
        $vis:vis fn $name:ident(
            $($arg:ident : $arg_ty:ty),* $(,)?
        ) $(-> $ret:ty)? {
            $($kernel_body:tt)*
        }
    ) => {
        #[cfg(@CFG@)]
        $(#[$meta])*
        $vis fn $name(
            _simd: $crate::@LEVEL_NAME@,
            $($arg: $arg_ty),*
        ) $(-> $ret)? {
@BODY@
        }
    };
}
"#;

const KERNEL_BODY_WITH_TARGET_FEATURES: &str = r#"            #[inline]
            #[target_feature(enable = "@FEATURES@")]
            fn __fearless_simd_kernel(
                $($arg: $arg_ty),*
            ) $(-> $ret)? {
                $($kernel_body)*
            }

            // SAFETY: the `@LEVEL_NAME@` token proves that the required target features are available.
            unsafe { __fearless_simd_kernel($($arg),*) }"#;

const KERNEL_BODY: &str = r#"            #[inline]
            fn __fearless_simd_kernel(
                $($arg: $arg_ty),*
            ) $(-> $ret)? {
                $($kernel_body)*
            }

            let _ = _simd;
            __fearless_simd_kernel($($arg),*)"#;
