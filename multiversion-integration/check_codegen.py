#!/usr/bin/env python3
"""Check this prototype's optimized x86-64 Linux assembly, without LTO.

Run: python3 multiversion-integration/check_codegen.py
Optional comparison: python3 multiversion-integration/check_codegen.py --static

These are observations about the installed Rust/LLVM toolchain and this example,
not guarantees about future optimization or arbitrary annotated Rust programs.
Only local assembly labels are normalized; constants, instructions, registers,
and branch destinations are compared. No third-party Python packages are needed.
"""

import argparse
import difflib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def run(argv, *, env=None):
    result = subprocess.run(argv, cwd=ROOT, env=env, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return result.stdout


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


class Assembly:
    def __init__(self, path):
        self.path = path
        self.text = path.read_text()
        self.functions = dict(re.findall(
            r"^([A-Za-z_][\w.$]*):\n(.*?)^\.Lfunc_end\d+:",
            self.text, re.MULTILINE | re.DOTALL,
        ))
        self.aliases = dict(re.findall(
            r"^([\w.$]+) = ([\w.$]+)$", self.text, re.MULTILINE,
        ))
        self.aliases.update(re.findall(
            r"^\s*\.set\s+([\w.$]+),\s*([\w.$]+)$", self.text, re.MULTILINE,
        ))

    def body(self, symbol):
        visited = set()
        while symbol in self.aliases:
            require(symbol not in visited, f"Cyclic assembly alias: {symbol}")
            visited.add(symbol)
            symbol = self.aliases[symbol]
        require(symbol in self.functions, f"Missing function {symbol} in {self.path}")
        return self.functions[symbol]

    def constant(self, label):
        match = re.search(r"^" + re.escape(label) + r":\n(.*?)(?=^\S|^\s*\.section|\Z)",
                          self.text, re.MULTILINE | re.DOTALL)
        require(match is not None, f"Missing constant {label}")
        return " ".join(line.strip() for line in match[1].splitlines()
                        if line.strip() and not line.lstrip().startswith(".p2align"))

    def normalized(self, symbol):
        body = self.body(symbol)
        # Substitute constant contents, so different values cannot compare equal
        # merely because they have corresponding labels in the constant pools.
        body = re.sub(r"\.LCPI\d+_\d+", lambda m: "CONST[" + self.constant(m[0]) + "]", body)
        body = re.sub(r"\.LBB\d+_(\d+)", r".LBB_\1", body)
        return "\n".join(line.strip() for line in body.splitlines() if line.strip())

    def assert_equal(self, first, second):
        a, b = self.normalized(first), self.normalized(second)
        require(a == b, "Assembly differs:\n" + "\n".join(difflib.unified_diff(
            a.splitlines(), b.splitlines(), fromfile=first, tofile=second,
        )))

    def follow_forwarders(self, symbol):
        """Follow one-instruction tail-jump wrappers, retaining their cost.

        kernel! emits a safe baseline wrapper around a feature-enabled body.
        Find that body from the instructions, without depending on names of
        private implementation details in the production macro.
        """
        chain = []
        while True:
            require(symbol not in chain, f"Cyclic forwarding wrappers: {symbol}")
            chain.append(symbol)
            instructions = [line.strip() for line in self.body(symbol).splitlines()
                            if line.strip() and not line.lstrip().startswith(".")]
            match = re.fullmatch(r"jmpq?\s+\*?([\w.$]+)(?:@GOTPCREL\(%rip\))?",
                                 instructions[0]) if len(instructions) == 1 else None
            if match is None:
                return chain
            symbol = match[1]

    def references(self, symbol):
        return set(re.findall(r"\b_R\w+", self.body(symbol)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static", action="store_true", help="also compare static relocation code")
    args = parser.parse_args()
    compiler = run(["rustc", "-vV"]).strip()
    require("host: x86_64-unknown-linux-gnu" in compiler,
            "This assembly checker currently supports x86_64-unknown-linux-gnu hosts.")
    evidence = [compiler, "", "Release optimization; LTO disabled; one codegen unit; target-cpu=x86-64.",
                "Offline builds; no global AVX2 assumption. These results are toolchain-specific."]
    for relocation in (["pic", "static"] if args.static else ["pic"]):
        env = os.environ.copy()
        # Do not accidentally measure a caller's target-cpu=native or LTO flags.
        env.pop("RUSTFLAGS", None)
        env["CARGO_ENCODED_RUSTFLAGS"] = "\x1f".join([
            "-Ctarget-cpu=x86-64", f"-Crelocation-model={relocation}",
        ])
        target_dir = ROOT / "target" / ("codegen-" + relocation)
        env["CARGO_TARGET_DIR"] = str(target_dir)
        assemblies = {}
        for package in ["simd-research-kernels", "simd-research-demo"]:
            print(f"Building {package} ({relocation}, no LTO)...", flush=True)
            output = run(["cargo", "rustc", "--offline", "--release", "--lib",
                 "--message-format=json-render-diagnostics",
                 "--target", "x86_64-unknown-linux-gnu", "-p", package,
                 "--config", "profile.release.lto=false",
                 "--config", "profile.release.codegen-units=1", "--", "--emit=asm"], env=env)
            # Use Cargo's exact artifact hash. Choosing the newest matching .s
            # could inspect stale code when an older build is reused from cache.
            paths = []
            for line in output.splitlines():
                event = json.loads(line)
                if event.get("reason") != "compiler-artifact" or event["target"]["name"] != package.replace("-", "_"):
                    continue
                for filename in event["filenames"]:
                    artifact = Path(filename)
                    if artifact.suffix == ".rmeta":
                        paths.append(artifact.with_name(artifact.name.removeprefix("lib")).with_suffix(".s"))
            require(len(paths) == 1 and paths[0].is_file(), f"Expected one emitted assembly for {package}: {paths}")
            assemblies[package] = Assembly(paths[0])

        demo = assemblies["simd-research-demo"]
        kernels = assemblies["simd-research-kernels"]
        automatic_chain = demo.follow_forwarders("automatic_avx2")
        manual_chain = demo.follow_forwarders("manual_avx2")
        opaque_chain = demo.follow_forwarders("automatic_opaque_avx2")
        manual_opaque_chain = demo.follow_forwarders("manual_opaque_avx2")
        demo.assert_equal(automatic_chain[-1], manual_chain[-1])
        demo.assert_equal(opaque_chain[-1], manual_opaque_chain[-1])
        require(len(automatic_chain) == len(manual_chain),
                "Automatic and manual vector loops have different wrapper costs")
        require(len(opaque_chain) == len(manual_opaque_chain),
                "Automatic and manual opaque loops have different wrapper costs")
        automatic = demo.body(automatic_chain[-1])
        opaque = demo.body(opaque_chain[-1])
        boundary = demo.body(demo.follow_forwarders("public_dispatch")[-1])
        loop = demo.body("unannotated_loop")
        detection = r"std_detect|detect_and_initialize|try_detect|selected_level"
        require("vpmulld" in automatic and "%ymm" in automatic, "AVX2 vector loop missing")
        require(not re.search(r"\bcallq?\b|" + detection, automatic),
                "Automatic AVX2 loop contains a call or CPU feature detection")
        require(not re.search(detection, opaque),
                "Opaque loop contains CPU feature detection")
        callees = demo.references(opaque_chain[-1])
        require(len(callees) == 1, f"Expected exactly one opaque AVX2 callee: {callees}")
        callee = callees.pop()
        require("Avx2" in callee and "opaque_mix" in callee,
                "Opaque loop does not reference its helper instantiated with Avx2")
        owner = next((assembly for assembly in [demo, kernels]
                      if callee in assembly.functions or callee in assembly.aliases), None)
        require(owner is not None, "Referenced opaque AVX2 definition missing")
        callee_chain = owner.follow_forwarders(callee)
        require(not re.search(r"\bcallq?\b|" + detection, owner.body(callee_chain[-1])),
                "Opaque AVX2 callee body contains a call or CPU feature detection")
        require(not re.search(r"\bjmpq?\s+\*", owner.body(callee_chain[-1])),
                "Opaque AVX2 callee body contains an indirect branch")
        require("try_detect" in boundary,
                "Expected public entry dispatch is missing")
        public_mix = next((name for name in kernels.functions
                           if "__simd_research" not in name and name.endswith("3mix")), None)
        # cargo rustc --emit=asm can give a crate a different disambiguator from
        # the same crate built as a dependency. Compare the public item name.
        loop_mix = [name for name in demo.references("unannotated_loop") if name.endswith("3mix")]
        require(public_mix is not None and len(loop_mix) == 1 and re.search(r"\bcallq?\b", loop),
                "Expected unannotated loop call to public mix is missing")
        require("try_detect" in kernels.body(public_mix), "Public mix wrapper dispatch missing")
        evidence += ["", f"{relocation.upper()} relocation:",
                     "PASS automatic_avx2 feature body == manual_avx2 feature body after label normalization.",
                     f"PASS both public kernel! entries have {len(automatic_chain) - 1} forwarding tail jump(s).",
                     "PASS AVX2 vector loop present; no calls or CPU feature detection in that body.",
                     "PASS automatic_opaque_avx2 feature body == manual_opaque_avx2 feature body after label normalization.",
                     f"PASS both opaque kernel! entries have {len(opaque_chain) - 1} forwarding tail jump(s).",
                     "PASS opaque loop calls the Avx2 helper instantiation; no backend selection in caller or callee.",
                     f"NOTE opaque helper has {len(callee_chain) - 1} forwarding tail jump(s) into its feature body.",
                     "The manual opaque comparison deliberately calls the same generic helper bridge.",
                     "A direct target_feature clone can avoid that extra helper forwarding jump.",
                     "PASS public_dispatch reaches a wrapper that calls Level::try_detect and selects a backend.",
                     "PASS unannotated_loop calls public mix, whose wrapper still calls Level::try_detect."]
        if re.search(r"\bcallq?\s+\*", opaque):
            evidence.append("Opaque calls use ordinary linkage indirection (also in manual code); no backend selection.")
        else:
            evidence.append("Opaque calls use direct machine calls to the selected AVX2 symbol.")
        evidence.append("Opaque loop call instruction(s):")
        evidence.extend(line.strip() for line in opaque.splitlines() if re.search(r"\bcallq?\b", line))
        evidence.append("Assembly: " + str(demo.path.relative_to(ROOT)))
    evidence += ["", "Exact assembly equivalence is evidence for these examples, not a language guarantee.",
                 "Public entry points retain Level detection/dispatch; unannotated callers pay it per call.",
                 "The inlined call graph eliminates internal dispatch; deliberately non-inlined helpers retain",
                 "a fixed forwarding jump in this generic Simd::vectorize integration."]
    report = "\n".join(evidence) + "\n"
    (ROOT / "codegen-results.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(f"Codegen check failed: {error}", file=sys.stderr)
        sys.exit(1)
