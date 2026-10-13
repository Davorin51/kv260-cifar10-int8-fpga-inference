#!/usr/bin/env python3
"""Offline numerical and compiled C++ checks; never launches AMD tools or hardware."""

import importlib.util
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from generate_report import ROOT, extract_hls, model_complexity


def load_quantizer():
    spec = importlib.util.spec_from_file_location("quantizer", ROOT / "quantize_cifar10_hls.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def host_check():
    compiler = shutil.which("g++")
    if not compiler:
        raise RuntimeError("g++ is required for host verification. No FPGA tools are needed.")
    sources = ROOT / "HLS/cifar10_cnn"
    with tempfile.TemporaryDirectory(prefix="kv260-check-") as name:
        work = Path(name)
        for filename in ["cnn.cpp", "cnn.h", "model_params.h", "test_vectors.h", "tb_cnn.cpp"]:
            shutil.copyfile(sources / filename, work / filename)
        binary = work / "cnn_check"

        def compile_run():
            subprocess.run([compiler, "-std=c++14", "-O2", "-Wno-unknown-pragmas",
                            str(work / "cnn.cpp"), str(work / "tb_cnn.cpp"), "-o", str(binary)],
                           check=True, capture_output=True, text=True)
            return subprocess.run([str(binary)], capture_output=True, text=True, timeout=60)

        result = compile_run()
        print(result.stdout, end="")
        if result.returncode:
            raise RuntimeError("Committed model and golden vectors disagree:\n" + result.stderr)
        vector_text = (work / "test_vectors.h").read_text()
        sample_match = re.search(r"NUM_TEST_SAMPLES\s*=\s*(\d+)", vector_text)
        expected = int(sample_match.group(1))
        if result.stdout.count("MATCH sample=") != expected + 1 or "(replay)" not in result.stdout:
            raise AssertionError("The testbench must exercise every committed sample and replay")
        # Independent negative check: corrupt sample 0, which the previous bench never ran.
        corrupted, changes = re.subn(
            r"(EXPECTED_LOGITS\s*\[.*?\]\s*=\s*\{\s*\{\s*)(-?\d+)",
            lambda match: match[1] + str(int(match[2]) + 1), vector_text, count=1, flags=re.S)
        if changes != 1:
            raise AssertionError("Could not locate a golden logit for the negative test")
        (work / "test_vectors.h").write_text(corrupted)
        rejected = compile_run()
        if rejected.returncode == 0 or "FAIL sample=0 class=0" not in rejected.stdout:
            raise AssertionError("The host test accepted a deliberately incorrect sample-0 logit")
        print("PASS: deliberately incorrect sample-0 logit rejected.")


def main():
    load_quantizer().self_test()
    host_check()
    for filename, calculated in [("hls_metrics", extract_hls()),
                                  ("model_complexity", model_complexity())]:
        recorded = json.loads((ROOT / f"results/{filename}.json").read_text())
        if recorded != calculated:
            raise AssertionError(f"{filename}.json is stale; regenerate the report")
    for name in ["accuracy", "resources", "workload", "latency"]:
        for suffix in ["png", "svg"]:
            path = ROOT / f"docs/assets/{name}.{suffix}"
            if not path.is_file() or path.stat().st_size == 0:
                raise AssertionError(f"Missing generated figure: {path}")
    print("PASS: HLS snapshot, model counts and portfolio figure files verified.")
    print("Scope: CPU numerical and C++ checks only; no RTL or board execution.")


if __name__ == "__main__":
    main()
