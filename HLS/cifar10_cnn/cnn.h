#ifndef CIFAR10_CNN_H
#define CIFAR10_CNN_H
#include <stdint.h>

// One image, 32x32 RGB, interleaved HWC bytes: R,G,B,R,G,B,...
// Input and output buffers must not overlap.
// Output: 10 signed INT32 logits in the quantizer's class order.
// The CPU can apply argmax directly: all 10 logits share the same scale.
void cifar10_cnn(const uint8_t image[3072], int32_t logits[10]);
#endif
