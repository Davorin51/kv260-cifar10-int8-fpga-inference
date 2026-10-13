# KV260 hardware integration

## Interface contract

```cpp
void cifar10_cnn(const uint8_t image[3072], int32_t logits[10]);
```

The image is raw 32×32 RGB in HWC/interleaved byte order. The accelerator applies `INPUT_LUT` internally. Do not normalize it a second time on the ARM side. All output logits share the same scale, so CPU argmax operates directly on INT32 values.

| Interface | Purpose |
|---|---|
| `m_axi_gmem_in` | Read 3,072 image bytes from DDR |
| `m_axi_gmem_out` | Write ten INT32 logits to DDR |
| `s_axi_control` | Start/status and input/output pointer registers |

Buffers must not overlap. Weights are synthesized constants; rebuild the accelerator and bitstream when changing the trained export.

## Existing platform sources

`Vivado/kv260_cifar10/kv260_cifar10.srcs/sources_1/bd/design_1/` contains the block-design source and IP configuration. Import/recreate it in your local Vivado project, provide the packaged CNN IP repository, generate the HDL wrapper and build/export the design for your tool version and board setup. Generated XSA/bitstream files are excluded from Git.

Create a Vitis platform from the matching XSA, with a standalone A53-64 domain and the required BSP. Use `Vitis/cifar10_app/src/main.c` and its matching `test_vectors.h` in the application. The source requires Xilinx cache/MMIO/printing headers and is not a normal host C program.

## Register map used by the application

| Offset | Register |
|---:|---|
| `0x00` | Control (`ap_start`, observed `ap_done`, `ap_idle`) |
| `0x04` | Global interrupt enable |
| `0x08` | Interrupt enable |
| `0x10`, `0x14` | Image pointer low/high |
| `0x1C`, `0x20` | Logits pointer low/high |

The current application base is `0xA0000000`. Always confirm the generated driver map and Vivado address assignment for your build; these values are specific to this design. Control uses polling with auto-restart and interrupts disabled.

## DDR and cache correctness

The current code checks that buffers are located at or above `0x00100000` and below `0x80000000`. Preserve a matching DDR LOW linker configuration in your application; an appropriate generated BSP/linker script is not supplied as a portable artifact here.

Input/output buffers are 64-byte aligned. The application flushes input and initialized output ranges before starting the accelerator, uses barriers, then invalidates output after `done`. This ensures the ARM does not read stale cache lines.

## Correctness and timing

The application warms up every exported image and replays sample 0. It then measures 100 calls while checking all ten logits after each timed service interval. A mismatch or timeout returns an error.

The timer reads A53 `cntpct_el0` and gets its frequency from `cntfrq_el0`. This is the physical generic counter, not the CPU core frequency. Conversion uses `ticks × 1,000,000 / timer_hz`.

Keep the original UART output with the bitstream/export identities when publishing new measurements. Distinguish IP-observed timing from service timing and from full input-acquisition/application throughput. A software timeout cannot recover a nonresponding MMIO transaction; reset/reload the platform after a failed run.

## Current verification scope

Historical flow records and owner-reported KV260 results are available, but the portfolio/CI update runs only CPU numerical and compiled C++ checks. It does not rerun synthesis, post-route timing, RTL simulation or board programming.
