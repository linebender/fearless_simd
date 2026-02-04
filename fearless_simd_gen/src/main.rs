// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![expect(
    missing_docs,
    reason = "TODO: https://github.com/linebender/fearless_simd/issues/40"
)]

use std::{fs::File, path::Path};

use clap::{Parser, Subcommand, ValueEnum};
use proc_macro2::TokenStream;

use crate::{level::Level as _, util::write_code};

mod arch;
mod generic;
mod level;
mod mk_core_arch;
mod mk_fallback;
mod mk_neon;
mod mk_ops;
mod mk_simd_trait;
mod mk_simd_types;
mod mk_wasm;
mod mk_x86;
mod ops;
mod parse_stdarch;
mod types;
mod util;

#[derive(Clone, Copy, ValueEnum, Debug)]
enum Module {
    SimdTypes,
    SimdTrait,
    Ops,
    Neon,
    Wasm,
    Fallback,
    Sse4_2,
    Avx2,
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

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Generate `core_arch` wrappers from stdarch source for all architectures.
    CoreArch,
}

impl Module {
    fn generate_code(self) -> TokenStream {
        match self {
            Self::SimdTypes => mk_simd_types::mk_simd_types(),
            Self::SimdTrait => mk_simd_trait::mk_simd_trait(),
            Self::Ops => mk_ops::mk_ops(),
            Self::Neon => mk_neon::Neon.make_module(),
            Self::Wasm => mk_wasm::WasmSimd128.make_module(),
            Self::Fallback => mk_fallback::Fallback.make_module(),
            Self::Sse4_2 => mk_x86::X86::Sse4_2.make_module(),
            Self::Avx2 => mk_x86::X86::Avx2.make_module(),
        }
    }

    fn file_base(self) -> &'static str {
        match self {
            Self::SimdTypes => "simd_types",
            Self::SimdTrait => "simd_trait",
            Self::Ops => "ops",
            Self::Neon => "neon",
            Self::Fallback => "fallback",
            Self::Wasm => "wasm",
            Self::Sse4_2 => "sse4_2",
            Self::Avx2 => "avx2",
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
    Module::Sse4_2,
    Module::Avx2,
];

const FILE_BASE: &str = "./fearless_simd/src/generated";
const CORE_ARCH_BASE: &str = "./fearless_simd/src/core_arch";
const STDARCH_BASE: &str = "./fearless_simd_gen/stdarch";

fn main() {
    let cli = Cli::parse();

    // Handle subcommands first
    if let Some(command) = cli.command {
        match command {
            Command::CoreArch => {
                let base_dir = Path::new(CORE_ARCH_BASE);
                let stdarch_dir = Path::new(STDARCH_BASE);
                if !base_dir.is_dir() || !stdarch_dir.is_dir() {
                    panic!("run in fearless_simd top directory");
                }
                let stdarch_is_empty = std::fs::read_dir(stdarch_dir).unwrap().next().is_none();
                if stdarch_is_empty {
                    panic!(
                        "`stdarch` submodule is empty. Initialize or update your git submodules."
                    );
                }
                generate_core_arch("./fearless_simd_gen/stdarch", CORE_ARCH_BASE);
                return;
            }
        }
    }

    if let Some(module) = cli.module {
        write_code(module.generate_code(), std::process::Stdio::inherit());
    } else {
        // generate all modules
        let base_dir = Path::new(FILE_BASE);
        if !base_dir.is_dir() {
            panic!("run in fearless_simd top directory");
        }
        for module in MODULES {
            let name = module.file_base();
            let path = base_dir.join(format!("{name}.rs"));
            let file = File::create(&path).unwrap_or_else(|_| panic!("error creating {path:?}"));
            write_code(module.generate_code(), file);
        }
    }
}

fn generate_core_arch(stdarch_path: &str, output_base: &str) {
    let stdarch_root = Path::new(stdarch_path);
    let output_base = Path::new(output_base);

    if !stdarch_root.exists() {
        eprintln!(
            "Error: stdarch directory not found at {}",
            stdarch_root.display()
        );
        eprintln!("Please provide a valid path to the stdarch repository.");
        std::process::exit(1);
    }

    mk_core_arch::generate_all_modules(stdarch_root, output_base);
}
