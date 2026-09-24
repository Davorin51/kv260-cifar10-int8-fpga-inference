/* KV260 standalone A53-64 benchmark. Replace application main.c with this file.
 * Requires the original test_vectors.h. Keep the working DDR LOW linker script.
 * Uses the A53 physical generic counter, not CPU cycle frequency.
 * Hardware and BSP build must be verified locally. */
#include <stdint.h>
#include <string.h>
#include "xil_io.h"
#include "xil_cache.h"
#include "xil_printf.h"
#include "test_vectors.h"

#define CNN_BASE ((UINTPTR)0xA0000000ULL)
#define RUNS 100U
#define SAMPLES (sizeof(TEST_IMAGES) / sizeof(TEST_IMAGES[0]))
#define CTRL 0x00U
#define GIE 0x04U
#define IER 0x08U
#define IMAGE_REG 0x10U
#define LOGITS_REG 0x1CU

static uint8_t image[3072] __attribute__((aligned(64)));
static int32_t output[16] __attribute__((aligned(64)));
static uint64_t timer_hz;
_Static_assert(SAMPLES >= 2, "Need at least two samples");
_Static_assert(sizeof(TEST_IMAGES[0]) == 3072, "Expected raw RGB HWC");
_Static_assert(sizeof(EXPECTED_LOGITS[0]) == 40, "Expected ten INT32 logits");
_Static_assert(sizeof(EXPECTED_LOGITS)/sizeof(EXPECTED_LOGITS[0]) == SAMPLES,
               "Sample counts must match");

static void barrier(void) { __asm__ volatile ("dsb sy" ::: "memory"); }
static uint64_t ticks(void)
{
    uint64_t value;
    __asm__ volatile ("isb\n\tmrs %0, cntpct_el0\n\tisb" : "=r"(value) :: "memory");
    return value;
}
static uint64_t frequency(void)
{
    uint64_t value;
    __asm__ volatile ("mrs %0, cntfrq_el0" : "=r"(value));
    return value;
}
static void pointer_reg(unsigned reg, const void *p)
{
    uint64_t a = (uint64_t)(UINTPTR)p;
    Xil_Out32(CNN_BASE + reg, (uint32_t)a);
    Xil_Out32(CNN_BASE + reg + 4U, (uint32_t)(a >> 32));
}
static int ddr_low(const void *p, unsigned size)
{
    uint64_t a = (uint64_t)(UINTPTR)p;
    return a >= 0x100000ULL && a + size <= 0x80000000ULL;
}
static unsigned usec(uint64_t t)
{
    return (unsigned)((t * 1000000ULL) / timer_hz);
}

/* No UART prints or result comparison inside either timed interval.
 * service_ticks: copy/initialization, cache maintenance, register setup,
 * start/poll, output invalidation. Excludes image acquisition/preprocessing.
 * ip_ticks: start-register write to observed done (includes AXI/poll overhead).
 * The MMIO reads themselves must respond for the timeout to be effective. */
static int run_sample(unsigned sample, uint64_t *ip_ticks, uint64_t *service_ticks)
{
    uint64_t service_begin = ticks();
    memcpy(image, TEST_IMAGES[sample], sizeof(image));
    for (unsigned c = 0; c < 16; ++c) output[c] = -123456789;
    Xil_DCacheFlushRange((INTPTR)image, sizeof(image));
    Xil_DCacheFlushRange((INTPTR)output, sizeof(output));
    barrier();
    pointer_reg(IMAGE_REG, image);
    pointer_reg(LOGITS_REG, output);
    (void)Xil_In32(CNN_BASE + CTRL); /* clear stale done */
    barrier();
    uint64_t begin = ticks();
    Xil_Out32(CNN_BASE + CTRL, 1U); /* start, auto restart disabled */
    barrier();
    for (;;) {
        unsigned status = Xil_In32(CNN_BASE + CTRL);
        uint64_t now = ticks();
        if (status & 2U) {
            *ip_ticks = now - begin;
            break;
        }
        if (now - begin >= 5ULL * timer_hz) {
            xil_printf("TIMEOUT sample=%u ctrl=0x%08x; reset/reload hardware.\r\n",
                       sample, status);
            return 2;
        }
    }
    barrier();
    Xil_DCacheInvalidateRange((INTPTR)output, sizeof(output));
    barrier();
    *service_ticks = ticks() - service_begin;
    for (unsigned c = 0; c < 10; ++c) {
        if (output[c] != EXPECTED_LOGITS[sample][c]) {
            xil_printf("FAIL sample=%u class=%u expected=%d actual=%d\r\n",
                       sample, c, (int)EXPECTED_LOGITS[sample][c], (int)output[c]);
            return 1;
        }
    }
    return 0;
}

typedef struct { uint64_t sum, min, max; } Stats;
static void record(Stats *s, uint64_t value)
{
    s->sum += value;
    if (value < s->min) s->min = value;
    if (value > s->max) s->max = value;
}
static void report(const char *name, const Stats *s)
{
    xil_printf("%s us: min=%u avg=%u max=%u\r\n", name,
               usec(s->min), usec(s->sum / RUNS), usec(s->max));
}
int main(void)
{
    uint64_t ip, service;
    Stats ip_stats = {0, UINT64_MAX, 0};
    Stats service_stats = {0, UINT64_MAX, 0};
    xil_printf("\r\nCIFAR10 KV260 BENCHMARK\r\n");
    timer_hz = frequency();
    if (!timer_hz || timer_hz > UINT32_MAX) {
        xil_printf("ERROR: invalid generic timer frequency.\r\n");
        return 1;
    }
    xil_printf("Generic timer: %u Hz\r\n", (unsigned)timer_hz);
    if (!ddr_low(image, sizeof(image)) || !ddr_low(output, sizeof(output))) {
        xil_printf("ERROR: buffers must be in DDR LOW, at or above 0x00100000.\r\n");
        return 1;
    }
    Xil_Out32(CNN_BASE + GIE, 0U);
    Xil_Out32(CNN_BASE + IER, 0U);
    Xil_Out32(CNN_BASE + CTRL, 0U);
    barrier();
    if (!(Xil_In32(CNN_BASE + CTRL) & 4U)) {
        xil_printf("ERROR: CNN not idle; reset/reload hardware.\r\n");
        return 1;
    }
    /* Correctness and warm-up: all exported images, then sample 0 again. */
    for (unsigned trial = 0; trial < (unsigned)SAMPLES + 1U; ++trial) {
        unsigned sample = trial == (unsigned)SAMPLES ? 0U : trial;
        int rc = run_sample(sample, &ip, &service);
        if (rc) return rc;
        xil_printf("MATCH warmup sample=%u%s\r\n", sample,
                   trial == (unsigned)SAMPLES ? " (replay)" : "");
    }
    xil_printf("Measuring %u calls; checking every logit.\r\n", RUNS);
    for (unsigned n = 0; n < RUNS; ++n) {
        int rc = run_sample(n % (unsigned)SAMPLES, &ip, &service);
        if (rc) return rc;
        record(&ip_stats, ip);
        record(&service_stats, service);
    }
    report("IP observed", &ip_stats);
    report("Service", &service_stats);
    unsigned avg_us = usec(service_stats.sum / RUNS);
    if (avg_us) {
        unsigned fps_x1000 = (unsigned)(1000000000ULL / avg_us);
        xil_printf("Service rate estimate: %u.%03u images/s\r\n",
                   fps_x1000 / 1000U, fps_x1000 % 1000U);
    }
    xil_printf("PASS: warmup/replay and %u measured calls, all logits match.\r\n", RUNS);
    return 0;
}
