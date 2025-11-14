// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Discontinued x86-64 target features.
//!
//! That is target features which were present on some CPUs, but later CPU families from the
//! same vendor did not include them.
//!
//! For more information, see <https://en.wikipedia.org/wiki/List_of_discontinued_x86_instructions>

mod tbm;
pub use tbm::Tbm;
