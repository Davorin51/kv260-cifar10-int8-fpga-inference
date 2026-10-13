# SmallCifarCNN_v1 — HLS component

Synthesizable C++ for the power-of-two INT8 export in this repository. Top function: `cifar10_cnn`. The archived component targets K26 part `xck26-sfvc784-2LV-c`, a 10 ns clock target and Vitis Unified IDE 2025.2.

| File | Purpose |
|---|---|
| `cnn.cpp`, `cnn.h` | Convolution, integer ReLU/requantization, pooling, GAP, classifier and AXI interfaces |
| `model_params.h` | Committed learned constants and arithmetic scales |
| `test_vectors.h` | Two committed reference images and expected INT32 logits; testbench only |
| `tb_cnn.cpp` | Checks every logit for every exported sample, then replays sample 0 |
| `prepare_hls.py` | Validates a new INT8 export and prepares headers/configuration |
| `hls_config.cfg` | Historical component configuration with the owner's absolute paths |
| `cifar10_cnn/` | Historical generated HLS/RTL artifacts and flow summaries |

## Host check of committed constants

From this directory, with a C++14 compiler:

```bash
g++ -std=c++14 -O2 -Wno-unknown-pragmas cnn.cpp tb_cnn.cpp -o cnn_check
./cnn_check
```

Expected: sample 0 predicts class 6, sample 1 predicts class 5, then sample 0 is replayed; all ten logits match their saved references. This is a logit-parity test, not a dataset accuracy measurement. Host compilation ignores HLS pragmas and does not execute RTL.

The previous testbench only checked sample 1. The current bench restores all-sample/replay coverage; historical co-simulation records retain their original scope.

## Prepare a new trained export

See the repository-root [reproduction guide](../../docs/usage.md) for the isolated `work/hls` workflow. `prepare_hls.py` checks arrays, header contents, fractional bits, input LUT and accumulator bounds before writing `model_params.h`, `test_vectors.h` and `hls_config.ready.cfg`.

Copy the generated configuration into your local HLS component configuration, preserving your component identity. Run C simulation → synthesis → C/RTL co-simulation → package. Use matching golden vectors from the same trained export, and rebuild the hardware whenever constants change.

The image interface accepts raw RGB HWC bytes and produces ten INT32 logits. Input normalization is inside the IP. [Architecture and arithmetic](../../docs/architecture.md) · [Hardware/cache contract](../../docs/hardware.md) · [Results and evidence](../../docs/results.md).
