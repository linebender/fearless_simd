use std::cmp::Reverse;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

struct Artifacts {
    dependencies: PathBuf,
    macro_library: PathBuf,
    runtime_library: PathBuf,
}

fn candidates(directory: &Path, prefix: &str, extension: &str) -> Vec<PathBuf> {
    let mut paths: Vec<_> = std::fs::read_dir(directory)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with(prefix)
                && path.extension().is_some_and(|found| found == extension)
        })
        .collect();
    paths.sort_by_key(|path| Reverse(std::fs::metadata(path).unwrap().modified().unwrap()));
    paths
}

// Different compiler versions and build flags can leave multiple libraries in
// the same directory. Select explicit paths using a separate positive load
// probe, newest first. Never use a failing safety case to choose dependencies.
fn artifacts() -> &'static Artifacts {
    static ARTIFACTS: OnceLock<Artifacts> = OnceLock::new();
    ARTIFACTS.get_or_init(|| {
        let dependencies = std::env::current_exe()
            .unwrap()
            .parent()
            .unwrap()
            .to_owned();
        let macros = candidates(
            &dependencies,
            "libsimd_research_macros-",
            std::env::consts::DLL_EXTENSION,
        );
        let runtimes = candidates(&dependencies, "libsimd_research_runtime-", "rlib");
        let mut diagnostics = String::new();
        for macro_library in macros {
            for runtime_library in &runtimes {
                let artifacts = Artifacts {
                    dependencies: dependencies.clone(),
                    macro_library: macro_library.clone(),
                    runtime_library: runtime_library.clone(),
                };
                let result = compile_with(
                    r#"
use simd_research_macros::simd;
#[simd]
pub fn probe(value: u32) -> u32 { value.wrapping_add(1) }
pub fn force_runtime_load(_: simd_research_runtime::fearless_simd::Level) {}
"#,
                    "artifact-load",
                    &artifacts,
                );
                if result.status.success() {
                    return artifacts;
                }
                diagnostics.push_str(&String::from_utf8_lossy(&result.stderr));
            }
        }
        panic!("no compatible compiled macro/runtime pair found: {diagnostics}");
    })
}

fn compile_with(source: &str, name: &str, artifacts: &Artifacts) -> std::process::Output {
    let directory: PathBuf =
        std::env::temp_dir().join(format!("simd-research-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let input = directory.join("probe.rs");
    std::fs::write(&input, source).unwrap();
    let result = Command::new("rustc")
        .args(["--edition=2024", "--crate-type=lib", "--emit=metadata"])
        .arg("--extern")
        .arg(format!(
            "simd_research_macros={}",
            artifacts.macro_library.display()
        ))
        .arg("--extern")
        .arg(format!(
            "simd_research_runtime={}",
            artifacts.runtime_library.display()
        ))
        .arg("-L")
        .arg(format!("dependency={}", artifacts.dependencies.display()))
        .arg("--out-dir")
        .arg(&directory)
        .arg(&input)
        .output()
        .unwrap();
    std::fs::remove_dir_all(directory).unwrap();
    result
}

// A nonzero exit alone is insufficient: the checks below require Rust's unsafe
// operation diagnostic, so missing dependencies cannot masquerade as success.
fn compile(source: &str, name: &str) -> std::process::Output {
    compile_with(source, name, artifacts())
}

#[test]
fn rewriting_does_not_allow_unsafe_argument_expressions() {
    let result = compile(
        r#"
use simd_research_macros::simd;
fn consume(value: u32) -> u32 { value }
#[simd]
fn invalid(pointer: *const u32) -> u32 {
    consume(*pointer)
}
"#,
        "unsafe-argument",
    );
    let diagnostics = String::from_utf8_lossy(&result.stderr);
    assert!(!result.status.success(), "unsafe argument compiled");
    assert!(diagnostics.contains("E0133"), "{diagnostics}");
    assert!(
        diagnostics.contains("dereference of raw pointer"),
        "{diagnostics}"
    );
}

#[test]
fn original_function_bodies_remain_subject_to_safe_rust_rules() {
    let result = compile(
        r#"
use simd_research_macros::simd;
#[simd]
fn invalid(pointer: *const u32) -> u32 {
    *pointer
}
"#,
        "unsafe-body",
    );
    let diagnostics = String::from_utf8_lossy(&result.stderr);
    assert!(!result.status.success(), "unsafe body compiled");
    assert!(diagnostics.contains("E0133"), "{diagnostics}");
    assert!(
        diagnostics.contains("dereference of raw pointer"),
        "{diagnostics}"
    );
}

#[test]
fn lint_overrides_cannot_grant_unsafe_permission() {
    let result = compile(
        r#"
use simd_research_macros::simd;
#[simd]
fn invalid(pointer: *const u32) -> u32 {
    #[allow(unsafe_op_in_unsafe_fn)]
    { *pointer }
}
"#,
        "unsafe-lint-override",
    );
    let diagnostics = String::from_utf8_lossy(&result.stderr);
    assert!(
        !result.status.success(),
        "lint override granted unsafe permission"
    );
    assert!(diagnostics.contains("E0133"), "{diagnostics}");
    assert!(
        diagnostics.contains("dereference of raw pointer"),
        "{diagnostics}"
    );
}

#[test]
fn explicit_user_unsafe_blocks_remain_available() {
    let result = compile(
        r#"
use simd_research_macros::simd;
fn consume(value: u32) -> u32 { value }
#[simd]
fn valid() -> u32 {
    let value = 7u32;
    let pointer = &value as *const u32;
    consume(unsafe { *pointer })
}
"#,
        "explicit-unsafe",
    );
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
}

#[test]
fn accidental_companion_modules_cannot_grant_unsafe_permission() {
    let result = compile(
        r#"
use simd_research_macros::simd;
fn accidental(value: u32) -> u32 { value }
mod accidental {
    pub unsafe fn __call<W, F>(_: W, _: F, value: u32) -> u32 { value }
}
#[simd]
fn invalid() -> u32 {
    accidental(3)
}
"#,
        "unsafe-companion",
    );
    let diagnostics = String::from_utf8_lossy(&result.stderr);
    assert!(!result.status.success(), "unsafe companion compiled");
    assert!(diagnostics.contains("E0133"), "{diagnostics}");
    assert!(
        diagnostics.contains("call to unsafe function"),
        "{diagnostics}"
    );
    assert!(diagnostics.contains("__call"), "{diagnostics}");
}
