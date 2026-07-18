// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

// The following lints are part of the Linebender standard set,
// but resolving them has been deferred for now.
// Feel free to send a PR that solves one or more of these.
#![expect(
    clippy::missing_assert_message,
    reason = "TODO: https://github.com/linebender/fearless_simd/issues/40"
)]

//! Tests for `fearless_simd`.

mod lm_generated;
mod ops;

// Because the slide amount is a const generic, the exhaustive tests have to *compile* one slide per amount per vector
// type. Disable them entirely.`
#[cfg(false)]
mod slide_exhaustive;
