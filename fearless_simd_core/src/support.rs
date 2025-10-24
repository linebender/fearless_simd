// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Support for the safety checks in [`trampoline!`](crate::trampoline!).
//!
//! Methods to compute whether a each feature in a target feature string (e.g. "sse2,fma")
//! is supported by a set of target features.
//!
//! The [`trampoline`](crate::trampoline!) macro takes both a target feature string,
//! and one (or more) [`TargetFeatureToken`](crate::TargetFeatureToken).
//! It uses the functions in this module to validate that the target feature string is
//! supported by the provided tokens.
//!
//! Because evaluating whether this is safe needs to happen at compile time (for both performance
//! and predictability), the methods in this file are written as `const` functions.
//! This leads to a bit of weirdness, including treating strings as `&[u8]` internally, as that
//! actually allows slicing (i.e. reading individual bytes). As far as I know, that isn't
//! currently possibly in const contexts for strings.
//! Note that the code is still written to be UTF-8 compatible, although we believe that
//! all currently supported target features are ASCII anyway.

/// The result of `is_feature_subset`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub enum SubsetResult {
    /// The required features are a subset of the permitted features.
    Yes,
    /// The required features are not all available.
    No {
        /// The feature which was found to be missing (there may be several such features).
        failing: &'static str,
    },
}

impl SubsetResult {
    /// A utility method to panic if the target features aren't supported.
    // TODO: How much more context would we be able to give if we inlined this?
    pub const fn unwrap(self) {
        match self {
            Self::Yes => (),
            // This is const, so we can't actually format out the failing value :(
            Self::No { .. } => panic!("Tokens provided are missing a necessary target feature."),
        }
    }
}

/// Determine whether the features in the target feature string `required` are a subset of the features in `permitted`.
/// See [the module level docs][self].
///
/// We require static lifetimes as this is primarily internal to the macro.
pub const fn is_feature_subset<const N: usize>(
    required: &'static str,
    permitted: [&[&'static str]; N],
) -> SubsetResult {
    let mut required_bytes = required.as_bytes();
    let mut finished = false;
    'input_feature: while !finished {
        let mut comma_idx = 0;
        // Find the first comma in required_bytes, or the end of the string.
        while comma_idx < required_bytes.len() && required_bytes[comma_idx] != b',' {
            comma_idx += 1;
        }
        // `comma_idx` is now the index of the comma, e.g. if the string was "sse,", idx would be 3
        // This is the feature we need to validate exists in permitted.
        let (to_find, remaining_required) = required_bytes.split_at(comma_idx);
        if let [comma, rest @ ..] = remaining_required {
            if *comma != b',' {
                panic!("Internal failure of expected behaviour.");
            } else {
                required_bytes = rest;
            }
        } else {
            // Exit out of the loop after this iteration.
            // Note that for input of `""`` and "sse,", we still need to search
            // for the input target feature `` (i.e. the empty string), to match Rust's behaviour here.
            finished = true;
        }

        let mut local_permitted = permitted.as_slice();
        while let [to_test, rest @ ..] = local_permitted {
            local_permitted = rest;
            if str_array_contains(to_test, to_find) {
                continue 'input_feature;
            }
        }
        // We tried all of the items, and `to_find` wasn't one of them.
        // Therefore, at least one of the features in the requested features wasn't supported
        return SubsetResult::No {
            failing: match core::str::from_utf8(to_find) {
                Ok(x) => x,
                Err(_) => panic!(
                    "We either found a comma or the end of the string, so before then should have been valid UTF-8."
                ),
            },
        };
    }
    // We found all of the required features.
    SubsetResult::Yes
}

const fn str_array_contains(mut haystack: &[&str], needle: &[u8]) -> bool {
    while let [to_test, rest @ ..] = haystack {
        haystack = rest;
        if byte_arrays_eq(to_test.as_bytes(), needle) {
            return true;
        }
    }
    false
}

const fn byte_arrays_eq(lhs: &[u8], rhs: &[u8]) -> bool {
    if lhs.len() != rhs.len() {
        return false;
    }
    let mut idx = 0;
    while idx < lhs.len() {
        if lhs[idx] != rhs[idx] {
            return false;
        }
        idx += 1;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::{SubsetResult, is_feature_subset};

    /// Test if each feature in the feature string `required` is an element in `permitted`.
    ///
    /// Should be equivalent to [`is_feature_subset`], but not written to be const compatible.
    fn is_feature_subset_simple<const N: usize>(
        required: &'static str,
        permitted: [&[&'static str]; N],
    ) -> SubsetResult {
        'feature: for feature in required.split(',') {
            for permitted_group in &permitted {
                for permitted_feature in *permitted_group {
                    if feature == *permitted_feature {
                        continue 'feature;
                    }
                }
            }
            // We tried all permitted feature, and this item wasn't present.
            return SubsetResult::No { failing: feature };
        }
        SubsetResult::Yes
    }

    /// Expect `is_feature_subset` to succeed.
    #[track_caller]
    fn expect_success<const N: usize>(required: &'static str, permitted: [&[&'static str]; N]) {
        let res1 = is_feature_subset(required, permitted);
        assert_eq!(res1, SubsetResult::Yes, "Const version failed.");
        // Sanity check against the "trivially correct" version.
        let res2 = is_feature_subset_simple(required, permitted);
        assert_eq!(res2, SubsetResult::Yes, "Simpler version failed.");
    }

    /// Expect `is_feature_subset` to fail (with only a single possible failure).
    #[track_caller]
    fn expect_failure<const N: usize>(
        required: &'static str,
        permitted: [&[&'static str]; N],
        failing: &'static str,
    ) {
        let res1 = is_feature_subset(required, permitted);
        assert_eq!(res1, SubsetResult::No { failing }, "Const version failed.");
        // Sanity check against the "trivially correct" version.
        let res2 = is_feature_subset_simple(required, permitted);
        assert_eq!(
            res2,
            SubsetResult::No { failing },
            "Simpler version failed."
        );
    }

    /// Expect `is_feature_subset` to fail, possibly with multiple potential missing features.
    #[track_caller]
    fn expect_any_failure<const N: usize>(required: &'static str, permitted: [&[&'static str]; N]) {
        let res1 = is_feature_subset(required, permitted);
        assert!(
            matches!(res1, SubsetResult::No { .. }),
            "Const version failed."
        );
        // Sanity check against the "trivially correct" version.
        let res2 = is_feature_subset_simple(required, permitted);
        assert!(
            matches!(res2, SubsetResult::No { .. }),
            "Simpler version failed."
        );
    }

    #[test]
    fn simple_cases() {
        expect_success("a,b,c", [&["a", "b", "c"]]);
        expect_failure("a,b,c", [&["a", "b"]], "c");
        expect_success("c,a,b", [&["a", "b", "c"]]);
        expect_failure("c,a,b", [&["a", "b"]], "c");
        expect_success("a,b", [&["a", "b", "c"]]);
        expect_failure("a,b", [&["a", "c"]], "b");
        expect_success("a,b,a,a", [&["a", "b", "c"]]);
        expect_success("a,b,c", [&["c"], &["b"], &["a"]]);

        // Check it correctly catches more than single item failures
        expect_success("a1,a2,a3", [&["a1", "a2", "a3"]]);
        expect_failure("a1,a2,a3", [&["a1", "a2"]], "a3");
        expect_success("a3,a1,a2", [&["a1", "a2", "a3"]]);
        expect_failure("a3,a1,a2", [&["a1", "a2"]], "a3");
        expect_success("a1,a2", [&["a1", "a2", "a3"]]);
        expect_failure("a1,a2", [&["a1", "a3"]], "a2");

        // Check it doesn't have false positives with prefixes
        expect_failure("a1,a2,a3", [&["a1", "a2", "a"]], "a3");
        expect_any_failure("a3,a1,a2", [&["a"]]);
        expect_success("a1,a2", [&["a1", "a2", "a3"]]);
        expect_failure("a1,a2", [&["a1", "a3"]], "a2");

        expect_failure("a1b,a2b", [&["a1b", "a3b"]], "a2b");
        expect_failure("a1b,a2b", [&["a1b", "a3b"]], "a2b");
        expect_failure("a1b,a2b", [&["a1b", "a3b"]], "a2b");
        expect_failure("a1b,a2b", [&["a1b", "a3b"]], "a2b");
    }

    #[test]
    fn incorrect_token() {
        // The permitted list here only allows features which are the literal `a1,a2`
        // This is completely impossible to pass, but it's worth checking
        expect_any_failure("a1,a2", [&["a1,a2"]]);
    }

    #[test]
    fn empty_feature() {
        expect_failure("a,b,", [&["a", "b"]], "");
        expect_failure("", [&["a", "b"]], "");

        // We succeed if the empty target feature is allowed; any case where this is relevant will always
        // be validated away by rustc anyway, as there is no target with the target feature `""`.
        // As such, there's no harm in being flexible here.
        expect_success("", [&[""]]);
        expect_success(",,,,,,", [&[""]]);
    }

    #[test]
    fn non_ascii_features() {
        expect_success("café", [&["café"]]);
        expect_failure("café", [&["cafe"]], "café");
    }
}
