// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![expect(missing_docs, reason = "this integration test is not public API")]

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
#[test]
#[ignore = "run explicitly in the native Linux CI job"]
fn simd_ui() {
    use ui_test::custom_flags::{edition::Edition, rustfix::RustfixMode};
    use ui_test::dependencies::DependencyBuilder;

    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut config = ui_test::Config::rustc(manifest_dir.join("tests/ui/simd"));
    config.out_dir = manifest_dir.join("../target/tests/simd-ui");
    // Match annotated errors and their locations, not rustc's rendered notes,
    // suggestions, or formatting. Unexpected errors still fail the test.
    config.output_conflict_handling = ui_test::ignore_output_conflict;
    config
        .comment_defaults
        .base()
        .set_custom("edition", Edition("2024".into()));
    config
        .comment_defaults
        .base()
        .set_custom("rustfix", RustfixMode::Disabled);

    let mut dependencies = DependencyBuilder {
        crate_manifest_path: manifest_dir.join("Cargo.toml"),
        ..DependencyBuilder::default()
    };
    dependencies.program.program = env!("CARGO").into();
    // DependencyBuilder builds ordinary dependencies, so explicitly request
    // the same features our fixtures otherwise obtain from dev-dependencies.
    dependencies.program.args.extend(
        [
            "--lib",
            "--locked",
            "--offline",
            "--features=fearless_simd/std,fearless_simd/force_support_fallback",
        ]
        .map(Into::into),
    );
    config
        .comment_defaults
        .base()
        .set_custom("dependencies", dependencies);

    let pass_dir = config.root_dir.join("pass");
    let fail_dir = config.root_dir.join("fail");
    // libtest already consumed --ignored. run_tests() would parse it again
    // and skip our ordinary UI cases; use the entry point without CLI parsing.
    ui_test::run_tests_generic(
        vec![config],
        move |path, config| {
            if path.starts_with(&pass_dir) || path.starts_with(&fail_dir) {
                ui_test::default_file_filter(path, config)
            } else {
                None
            }
        },
        ui_test::default_per_file_config,
        ui_test::status_emitter::Text::verbose(),
    )
    .expect("SIMD UI fixtures should pass");

    let status = std::process::Command::new(env!("CARGO"))
        .args(["check", "--offline", "--manifest-path"])
        .arg(manifest_dir.join("tests/ui/simd/renamed/Cargo.toml"))
        .arg("--target-dir")
        .arg(manifest_dir.join("../target/tests/simd-renamed"))
        .status()
        .expect("renamed-dependency fixture should start cargo");
    assert!(
        status.success(),
        "renamed-dependency fixture should compile"
    );
}
