#!/usr/bin/env python3
"""Build/run the two x86-64 experiments and compare optimized assembly."""

from pathlib import Path
import re
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT.parent))
from check_codegen import Assembly


with tempfile.TemporaryDirectory(prefix="fearless-alternatives-") as temp:
    for name, generated, manual in [
        ("inline_region", "region_avx2", "manual_avx2"),
        ("callable_marker", "marker_process", "manual_process"),
    ]:
        binary = Path(temp) / name
        subprocess.run(
            [
                "rustc", "--edition=2024", "-C", "opt-level=3",
                "--emit=asm,link", "--out-dir", temp, str(ROOT / f"{name}.rs"),
            ],
            check=True,
        )
        subprocess.run([str(binary)], check=True)
        assembly = Assembly(binary.with_suffix(".s"))
        assembly.assert_equal(generated, manual)
        generated_body = assembly.body(generated)
        assert not re.search(r"\bcall\w*\s", generated_body), name
        print(f"{name}: identical assembly to manual AVX2; no call instructions")

    # Generated safe adapters must not grant unsafe permissions to an argument
    # expression or to an ordinary callee in the original source.
    bad_source = Path(temp) / "unsafe_calls.rs"
    bad_source.write_text((ROOT / "callable_marker.rs").read_text() + """
unsafe fn requires_unsafe(x: u32) -> u32 { x }
fn unsafe_argument_is_rejected(witness: Avx2, pointer: *const u32) {
    let _ = call!(witness, ordinary, *pointer);
}
fn unsafe_callee_is_rejected(witness: Avx2) {
    let _ = call!(witness, requires_unsafe, 42);
}
""")
    result = subprocess.run(
        ["rustc", "--edition=2024", "--out-dir", temp, str(bad_source)],
        capture_output=True, text=True,
    )
    assert result.returncode != 0, "Unsafe call examples unexpectedly compiled"
    assert result.stderr.count("error[E0133]") == 2, result.stderr
    print("callable_marker: unsafe source arguments and callees remain rejected")
