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

    KERNEL_MACRO_TEMPLATE
        .replace("@MACRO_NAME@", &macro_name)
        .replace("@LEVEL_NAME@", name)
        .replace("@CFG@", cfg)
        .replace("@BODY@", &body)
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
#[doc = "Defines a safe, non-generic kernel for `@LEVEL_NAME@`."]
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
