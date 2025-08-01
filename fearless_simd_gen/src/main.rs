// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use std::{fs::File, io::Write, path::Path};

use clap::{Parser, ValueEnum};
use proc_macro2::TokenStream;

mod arch;
mod generic;
mod mk_fallback;
mod mk_neon;
mod mk_ops;
mod mk_simd_trait;
mod mk_simd_types;
mod mk_wasm;
mod ops;
mod types;

#[derive(Clone, Copy, ValueEnum, Debug)]
enum Module {
    SimdTypes,
    SimdTrait,
    Ops,
    Neon,
    Wasm,
    Fallback,
}

#[derive(Parser)]
#[command(
    name = "fearless_simd_gen",
    about = "Generate SIMD trait implementations for `fearless_simd`",
    long_about = "Generate SIMD trait implementations for fearless_simd.\n\
                  \n\
                  Generates code for SIMD types, traits, operations, and architecture-specific \
                  implementations (NEON, WASM, fallback).\n\
                  \n\
                  Run from the root of the repository without arguments to automatically \
                  generate all module files in ./fearless_simd/src/generated/."
)]
struct Cli {
    #[arg(short, long, help = "Generate a specific module and print to stdout")]
    module: Option<Module>,
}

impl Module {
    fn generate_code(self) -> TokenStream {
        match self {
            Module::SimdTypes => mk_simd_types::mk_simd_types(),
            Module::SimdTrait => mk_simd_trait::mk_simd_trait(),
            Module::Ops => mk_ops::mk_ops(),
            Module::Neon => mk_neon::mk_neon_impl(mk_neon::Level::Neon),
            Module::Wasm => mk_wasm::mk_wasm128_impl(mk_wasm::Level::WasmSimd128),
            Module::Fallback => mk_fallback::mk_fallback_impl(),
        }
    }

    fn generate(self, out: impl Into<std::process::Stdio>) {
        let code = self.generate_code();
        let mut child = std::process::Command::new("rustfmt")
            .stdin(std::process::Stdio::piped())
            .stdout(out)
            .spawn()
            .expect("`rustfmt` should spawn");
        let mut stdin = child.stdin.take().unwrap();
        stdin
            .write_all(
                format!(
                    r#"
// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

// This file is autogenerated by fearless_simd_gen

{code}"#
                )
                .as_bytes(),
            )
            .unwrap();
        drop(stdin);
        child.wait().expect("`rustfmt` should succeed");
    }

    fn file_base(self) -> &'static str {
        match self {
            Module::SimdTypes => "simd_types",
            Module::SimdTrait => "simd_trait",
            Module::Ops => "ops",
            Module::Neon => "neon",
            Module::Fallback => "fallback",
            Module::Wasm => "wasm",
        }
    }
}

const MODULES: &[Module] = &[
    Module::SimdTypes,
    Module::SimdTrait,
    Module::Ops,
    Module::Neon,
    Module::Fallback,
    Module::Wasm,
];

const FILE_BASE: &str = "./fearless_simd/src/generated";

fn main() {
    let cli = Cli::parse();
    if let Some(module) = cli.module {
        module.generate(std::process::Stdio::inherit());
    } else {
        // generate all modules
        let base_dir = Path::new(FILE_BASE);
        if !base_dir.is_dir() {
            panic!("run in fearless_simd top directory");
        }
        for module in MODULES {
            let name = module.file_base();
            let path = base_dir.join(format!("{name}.rs"));
            let file = File::create(&path).expect("error creating {path:?}");
            module.generate(file);
        }
    }
}
