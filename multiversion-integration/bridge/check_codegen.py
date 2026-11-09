#!/usr/bin/env python3
"""Compare the hidden-generic bridge with a handwritten AVX2 entry."""

from pathlib import Path
import re
import subprocess

root = Path(__file__).resolve().parent
subprocess.run(
    ["cargo", "rustc", "--offline", "--release", "--", "--emit=asm"],
    cwd=root,
    check=True,
)
assembly = max(
    (root / "target/release/deps").glob("fearless_generic_bridge-*.s"),
    key=lambda path: path.stat().st_mtime,
).read_text()
bodies = re.findall(
    r"^(_R\w*vectorize_avx2\w*):\n(.*?)^\.Lfunc_end\d+:",
    assembly,
    re.M | re.S,
)
selected = {}
for symbol, body in bodies:
    if "process6___simd" in symbol:
        name = "automatic"
    elif "11manual_avx2" in symbol:
        name = "manual"
    else:
        continue
    body = re.sub(r"\.LBB\d+_(\d+)", r".LBB_\1", body)
    body = re.sub(r"\.LCPI\d+_(\d+)", r".LCPI_\1", body)
    selected[name] = body

assert set(selected) == {"automatic", "manual"}, selected.keys()
assert selected["automatic"] == selected["manual"], "assembly differs"
assert "ymm" in selected["automatic"], "no AVX2 vectorization found"
assert not re.search(r"\bcall\w*\b", selected["automatic"]), "loop contains call"
print("Automatic and manual AVX2 bodies are identical after label normalization.")
print("Automatic body uses YMM vectors and contains no calls.")
