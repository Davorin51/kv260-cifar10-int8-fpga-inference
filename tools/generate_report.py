#!/usr/bin/env python3
"""Generate portfolio figures from reported metrics and committed HLS artifacts.

No training, hardware execution or invented history is performed. Optional
--history reads the real history.csv produced by train_cifar10_hls.py.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
HLS_REPORT = Path("HLS/cifar10_cnn/cifar10_cnn/hls/hls_data.json")
INK = "#20313c"
ORANGE = "#cf6537"
TEAL = "#367c84"
MUTED = "#566773"


def extract_hls(root=ROOT):
    path = root / HLS_REPORT
    raw = path.read_bytes()
    report = json.loads(raw)
    top = report["ModuleInfo"]["Metrics"][report["Top"]]
    period = float(report["ClockInfo"]["ClockPeriod"])
    latency = int(top["Latency"]["LatencyAvg"])
    resources = {}
    for name, available in [("LUT", "AVAIL_LUT"), ("FF", "AVAIL_FF"),
                            ("BRAM_18K", "AVAIL_BRAM"), ("DSP", "AVAIL_DSP")]:
        used, capacity = int(top["Area"][name]), int(top["Area"][available])
        resources[name] = {"used": used, "capacity": capacity,
                           "utilization_pct": 100 * used / capacity}
    return {"schema_version": 1, "source": HLS_REPORT.as_posix(),
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "scope": "HLS top-function synthesis estimates; not post-route whole-design utilization",
            "top": report["Top"], "target_part": "xck26-sfvc784-2LV-c",
            "target_period_ns": period,
            "estimated_period_ns": float(top["Timing"]["Estimate"]),
            "latency_cycles": latency,
            "initiation_interval_cycles": int(top["Latency"]["PipelineII"]),
            "latency_at_target_clock_ms": latency * period / 1_000_000,
            "resources": resources}


def model_complexity():
    """Dense mathematical MACs, including padded positions; no timing estimate."""
    layers = []
    for name, ci, co, spatial in [("Conv 1", 3, 16, 32),
                                  ("Conv 2", 16, 32, 16),
                                  ("Conv 3", 32, 64, 8)]:
        layers.append({"layer": name, "weight_count": ci * co * 9,
                       "bias_count": co, "dense_macs": spatial**2 * co * ci * 9})
    layers.append({"layer": "Classifier", "weight_count": 64 * 10,
                   "bias_count": 10, "dense_macs": 64 * 10})
    weights = sum(row["weight_count"] for row in layers)
    biases = sum(row["bias_count"] for row in layers)
    return {"schema_version": 1, "architecture": "SmallCifarCNN_v1", "layers": layers,
            "parameter_count": weights + biases,
            "total_dense_macs": sum(row["dense_macs"] for row in layers),
            "fp32_parameter_bytes": (weights + biases) * 4,
            "int8_weight_int32_bias_bytes": weights + biases * 4,
            "counting_rule": "Dense convolution/linear products only; padded zero products included. Pooling, LUT, rounding, additions and AXI transfer excluded. Parameter bytes exclude input LUT and buffers; not measured FPGA storage or a file size."}


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.labelcolor": MUTED, "text.color": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.edgecolor": "#dce3e6", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.spines.left": False,
                         "axes.titleweight": "bold", "axes.titlepad": 22,
                         "figure.facecolor": "#ffffff", "axes.facecolor": "#ffffff",
                         "svg.hashsalt": "kv260-cifar10-report-v1", "svg.fonttype": "none"})


def save(fig, output, name, note):
    fig.text(.08, .025, note, fontsize=9, color=MUTED, va="bottom")
    fig.subplots_adjust(left=.12, right=.94, bottom=.23, top=.80)
    fig.savefig(output / f"{name}.png", dpi=180, metadata={"Software": "KV260 report generator"})
    fig.savefig(output / f"{name}.svg", metadata={"Date": None})
    plt.close(fig)


def history_plot(path, output):
    fields = ["epoch", "train_loss", "val_loss", "train_accuracy_pct", "val_accuracy_pct"]
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not set(fields) <= set(reader.fieldnames or []):
            raise ValueError("history.csv is missing training/validation columns")
        rows = [{key: float(row[key]) for key in fields} for row in reader]
    if not rows or any(not math.isfinite(v) for row in rows for v in row.values()):
        raise ValueError("history.csv must contain finite, nonempty measurements")
    epochs = [row["epoch"] for row in rows]
    if epochs != sorted(set(epochs)) or any(e < 1 or e != int(e) for e in epochs):
        raise ValueError("Epochs must be positive, unique integers in ascending order")
    if any(not 0 <= row[k] <= 100 for row in rows for k in fields[3:]):
        raise ValueError("Accuracy must be between 0 and 100 percent")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, suffix, title in [(axes[0], "accuracy_pct", "Accuracy"), (axes[1], "loss", "Cross-entropy loss")]:
        for split, color in [("train", TEAL), ("val", ORANGE)]:
            ax.plot(epochs, [row[f"{split}_{suffix}"] for row in rows], color=color, label=split.title())
        ax.set(title=title, xlabel="Epoch")
        ax.grid(axis="y", alpha=.2)
        ax.legend(frameon=False)
    axes[0].set_ylim(0, 100)
    axes[0].yaxis.set_major_formatter(PercentFormatter())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    save(fig, output, "training_history", f"Actual supplied history.csv · SHA-256 prefix {digest} · No interpolated or synthetic epochs")


def generate(root=ROOT, output=None, history=None):
    output = output or root / "docs/assets"
    output.mkdir(parents=True, exist_ok=True)
    reported = json.loads((root / "results/reported_metrics.json").read_text())
    hls, complexity = extract_hls(root), model_complexity()
    for name, content in [("hls_metrics", hls), ("model_complexity", complexity)]:
        (root / f"results/{name}.json").write_text(json.dumps(content, indent=2) + "\n")
    style()
    accuracy = reported["accuracy"]
    fig, ax = plt.subplots(figsize=(9, 5))
    values = [accuracy["fp32_pct"], accuracy["int8_pct"]]
    bars = ax.bar(["FP32 reference", "Power-of-two INT8"], values, color=[TEAL, ORANGE], width=.48)
    ax.set(ylim=(0, 100), ylabel="Reported accuracy", title="CIFAR-10 · reported accuracy comparison")
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.grid(axis="y", color="#e8edef")
    ax.set_axisbelow(True)
    ax.bar_label(bars, labels=[f"{v:.2f}%" for v in values], padding=8, fontweight="bold")
    drop = values[0] - values[1]
    ax.text(.5, .92, f"{drop:.2f} percentage-point difference", transform=ax.transAxes, ha="center", color=ORANGE)
    save(fig, output, "accuracy", "Source: owner-reported historical results · Original evaluation artifacts not committed\nHLS guide identifies INT8 as validation accuracy; this is not a verified official-test-set comparison.")
    fig, ax = plt.subplots(figsize=(10, 5))
    labels, percents = [], []
    for name in ["BRAM_18K", "LUT", "FF", "DSP"]:
        r = hls["resources"][name]
        labels.append(f'{name.replace("_", " ")}\n{r["used"]:,} / {r["capacity"]:,}')
        percents.append(r["utilization_pct"])
    bars = ax.bar(labels, percents, color=[ORANGE, TEAL, TEAL, TEAL], width=.55)
    ax.set(ylim=(0, 12), ylabel="Used / report-specified capacity (%)", title="HLS synthesis · estimated resource utilization")
    ax.grid(axis="y", color="#e8edef")
    ax.set_axisbelow(True)
    ax.bar_label(bars, labels=[f"{v:.2f}%" for v in percents], padding=6, fontweight="bold")
    save(fig, output, "resources", "Source: committed hls_data.json · cifar10_cnn top-function estimate\nNot a post-route result; excludes the complete Vivado system.")
    fig, ax = plt.subplots(figsize=(10, 5))
    layers = complexity["layers"]
    bars = ax.bar([r["layer"] for r in layers], [r["dense_macs"] / 1e6 for r in layers], color=[TEAL, ORANGE, ORANGE, TEAL], width=.55)
    ax.set(ylim=(0, 1.45), ylabel="Dense MACs per image (millions)", title="Model workload · 2.802 million dense MACs/image")
    ax.grid(axis="y", color="#e8edef")
    ax.set_axisbelow(True)
    ax.bar_label(bars, labels=[f'{r["dense_macs"]:,}' for r in layers], padding=7, fontweight="bold")
    save(fig, output, "workload", "Calculated from SmallCifarCNN_v1 layer dimensions · 24,234 trainable parameters\nDense MACs include padded positions; exclude pooling, LUT, transfers and control overhead.")
    fig, ax = plt.subplots(figsize=(10, 5))
    latencies = [hls["latency_at_target_clock_ms"], reported["hardware"]["reported_observed_latency_ms"]]
    bars = ax.bar(["HLS cycles × target period", "Reported KV260 observed"], latencies, color=[TEAL, ORANGE], width=.48)
    ax.set(ylim=(0, 36), ylabel="Latency (ms/image)", title="Latency · estimate and reported board observation")
    ax.grid(axis="y", color="#e8edef")
    ax.set_axisbelow(True)
    ax.bar_label(bars, labels=[f"{latencies[0]:.5f} ms", f"≈{latencies[1]:.2f} ms"], padding=8, fontweight="bold")
    save(fig, output, "latency", "HLS: 2,918,914 cycles at a 10 ns target period · Board: owner-reported ≈29.19 ms\nObserved start-to-done includes MMIO/polling; no raw distribution or end-to-end camera rate is available.")
    if history:
        history_plot(history, output)
    print(f"Generated four PNG/SVG figures in {output}")
    print("Reported measurements were not rerun. HLS and model-derived JSON snapshots refreshed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/assets")
    parser.add_argument("--history", type=Path, help="Actual training history.csv; omitted when unavailable")
    args = parser.parse_args()
    generate(output=args.output, history=args.history)


if __name__ == "__main__":
    main()
