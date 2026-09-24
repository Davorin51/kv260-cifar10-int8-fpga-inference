# KV260 CIFAR-10 INT8 FPGA Inference

End-to-end deployment of a quantized convolutional neural network for CIFAR-10 on the AMD/Xilinx Kria KV260 using PyTorch, Vitis HLS, Vivado and Vitis.

## Overview

This project demonstrates a complete workflow for deploying a CNN from software training to FPGA-based inference.

The workflow includes:

- CNN training in PyTorch
- INT8 quantization
- Export of model parameters and test vectors
- C/C++ inference implementation
- Vitis HLS synthesis and C/RTL co-simulation
- Vivado block design integration
- Vitis bare-metal application
- Hardware execution and benchmarking on the KV260

## Hardware

- AMD/Xilinx Kria KV260 Vision AI Starter Kit
- Zynq UltraScale+ MPSoC

## Software

- Python
- PyTorch
- C / C++
- Vitis HLS
- Vivado
- Vitis

## Workflow

```text
PyTorch CNN
    |
    v
INT8 Quantization
    |
    v
Model Parameters / Test Vectors
    |
    v
C/C++ CNN Implementation
    |
    v
Vitis HLS
    |
    v
C/RTL Co-Simulation
    |
    v
Vivado IP Integration
    |
    v
Vitis Application
    |
    v
KV260 Hardware Inference
