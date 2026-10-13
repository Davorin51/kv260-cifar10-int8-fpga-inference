# Usage and reproduction

## 1. Offline verification and figures

Python 3.11+, NumPy, matplotlib and g++ are sufficient. The commands below need no CIFAR-10 download, GPU or AMD installation:

```bash
python -m pip install -r requirements-reporting.txt
python quantize_cifar10_hls.py --self-test
python tools/check_project.py
python tools/generate_report.py
```

Run from the repository root. The checker uses temporary compiled binaries and leaves the committed parameter and vector headers unchanged. CI follows this same offline path.

## 2. Train a new FP32 model

Install matching PyTorch and torchvision builds for your platform. For a simple pip environment:

```bash
python -m pip install torch torchvision numpy
python train_cifar10_hls.py --epochs 150 --out runs/cifar10_01
```

Use `--device cuda` only with a working CUDA-enabled PyTorch installation. The default `auto` chooses available CUDA or CPU. The script downloads CIFAR-10, uses a seeded stratified 45,000/5,000 training/validation split, selects best validation weights and evaluates the official test set afterward.

The output directory must be new or empty. Important artifacts:

| File | Purpose |
|---|---|
| `best_weights.pt` | CPU FP32 state dict of the selected model |
| `best_weights_fp32.npz` | NumPy weights/biases for the integer reference |
| `config.json`, `split_indices.npz` | Architecture, preprocessing, configuration and exact split |
| `history.csv` | Measured epoch losses, accuracies, learning rate and wall time |
| `results.json` | Best epoch and final test metrics |
| `golden_samples.npz` | FP32 reference images and logits |
| `last_checkpoint.pt` | Last model/optimizer/scheduler state; the script has no automatic resume command |

## 3. Calibrate and export INT8

```bash
python quantize_cifar10_hls.py --run-dir runs/cifar10_01 --calib-size 1000 --eval-size 5000
```

Calibration uses original training images without augmentation. Evaluation uses the saved validation split. `--eval-size 5000` selects the full validation set; default is a 500-image subset. This NumPy implementation is a correctness reference, not an optimized CPU benchmark.

The default export directory is `runs/cifar10_01/int8`. Outputs include `model_int8.npz`, `quantization.json`, `model_params.h`, `comparison.json`, prediction CSV and `golden_int8.npz`. Keep these files together: they bind numeric scales, weights and expected logits.

## 4. Prepare an isolated HLS component

Keep a new trained export separate from the historical headers committed in this repository:

```bash
mkdir -p work/hls
cp HLS/cifar10_cnn/cnn.cpp HLS/cifar10_cnn/cnn.h HLS/cifar10_cnn/tb_cnn.cpp work/hls/
python HLS/cifar10_cnn/prepare_hls.py \
  --quant-dir runs/cifar10_01/int8 \
  --component-dir work/hls \
  --samples 2
```

The preparation script validates the quantized arrays, scales, accumulator bounds and header agreement, then writes model constants, golden vectors and `hls_config.ready.cfg` with paths for your machine. Do not use the archived `/home/davorin/...` paths as portable paths.

Create/import the HLS component in Vitis Unified IDE using your local sources and generated configuration. Run C simulation, synthesis, C/RTL co-simulation and packaging. The archived project targets tools 2025.2, K26 part `xck26-sfvc784-2LV-c`, 10 ns clock target and top `cifar10_cnn`.

## 5. Integrate and run on the board

Continue with [hardware integration](hardware.md). Vivado/Vitis, a compatible board setup, a built bitstream/XSA, standalone A53 BSP and DDR linker placement are required. The repository includes source and historical HLS artifacts; it does not provide a portable prebuilt board release or a fully scripted Vivado project recreation.

## 6. Publish results from a new run

```bash
python tools/generate_report.py --history runs/cifar10_01/history.csv
```

The optional training plot uses your actual CSV. The four default figures still use the documented historical snapshot: a new training run does not automatically replace `reported_metrics.json`. Update the reported-results record only after storing and documenting the corresponding evaluation and hardware evidence.

Useful official references: [PyTorch installation](https://pytorch.org/get-started/locally/), [PyTorch model saving/loading](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html), [Vitis HLS configuration](https://docs.amd.com/r/en-US/ug1399-vitis-hls/HLS-Config-File-Commands).
