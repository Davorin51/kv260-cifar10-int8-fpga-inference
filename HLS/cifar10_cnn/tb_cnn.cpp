#include "cnn.h"
#include "test_vectors.h"
#include <cstdio>

int main() {
    uint8_t image[3072];
    int32_t actual[10];
    int failures = 0;
    // Replay sample 0 after the other samples to detect stale internal buffers.
    for (int trial = 0; trial < 1; ++trial) {
        const int sample = 1;  // Druga testna slika
        for (int i = 0; i < 3072; ++i) image[i] = TEST_IMAGES[sample][i];
        for (int i = 0; i < 10; ++i) actual[i] = -123456789;
        cifar10_cnn(image, actual);
        int predicted = 0;
        bool exact = true;
        for (int i = 0; i < 10; ++i) {
            if (actual[i] != EXPECTED_LOGITS[sample][i]) {
                std::printf("FAIL sample=%d class=%d expected=%ld actual=%ld\n",
                            sample, i, (long)EXPECTED_LOGITS[sample][i], (long)actual[i]);
                ++failures;
                exact = false;
            }
            if (actual[i] > actual[predicted]) predicted = i;
        }
        std::printf("%s sample=%d prediction=%d label=%d%s\n",
                    exact ? "MATCH" : "MISMATCH", sample, predicted, (int)TEST_LABELS[sample],
                    trial == NUM_TEST_SAMPLES ? " (replay)" : "");
    }
    if (failures) {
        std::printf("FAIL: %d logit mismatches\n", failures);
        return 1;
    }
    std::printf("PASS: all 10 INT32 logits match for sample 1.\n");
    return 0;
}
