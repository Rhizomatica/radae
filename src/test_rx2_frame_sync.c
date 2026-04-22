/*
 * Streaming bit-accuracy test for rx2_frame_sync_apply().
 *
 * Stdin format (binary):
 *   int32      ncases
 *   repeated ncases times:
 *     int32       sym_index
 *     float32[]   z_hat (FRAME_SYNC_INPUT_DIM)
 *     float32     expected_metric
 *     float32     expected_even
 *     float32     expected_odd
 *     int32       expected_valid
 *     float32[]   expected_az_hat (FRAME_SYNC_INPUT_DIM)
 *
 * Copyright (c) 2026 Rhizomatica, BSD-3-Clause
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "rx2_frame_sync.h"

#define FLOAT_TOL 5e-4f

int main(void) {
    int32_t ncases;
    if (fread(&ncases, sizeof(ncases), 1, stdin) != 1) {
        fprintf(stderr, "truncated header\n");
        return 1;
    }

    struct rx2_frame_sync fs;
    if (rx2_frame_sync_init(&fs) != 0) {
        fprintf(stderr, "rx2_frame_sync_init failed\n");
        return 1;
    }

    float max_err = 0.0f;
    int fails = 0;
    int compared = 0;

    for (int c = 0; c < ncases; c++) {
        int32_t sym_index, expected_valid;
        float z_hat[FRAME_SYNC_INPUT_DIM];
        float expected_metric, expected_even, expected_odd;
        float expected_az_hat[FRAME_SYNC_INPUT_DIM];
        int valid;

        if (fread(&sym_index, sizeof(sym_index), 1, stdin) != 1 ||
            fread(z_hat, sizeof(*z_hat), FRAME_SYNC_INPUT_DIM, stdin) != FRAME_SYNC_INPUT_DIM ||
            fread(&expected_metric, sizeof(expected_metric), 1, stdin) != 1 ||
            fread(&expected_even, sizeof(expected_even), 1, stdin) != 1 ||
            fread(&expected_odd, sizeof(expected_odd), 1, stdin) != 1 ||
            fread(&expected_valid, sizeof(expected_valid), 1, stdin) != 1 ||
            fread(expected_az_hat, sizeof(*expected_az_hat), FRAME_SYNC_INPUT_DIM, stdin) != FRAME_SYNC_INPUT_DIM) {
            fprintf(stderr, "truncated payload at c=%d\n", c);
            fails = 1;
            break;
        }

        valid = rx2_frame_sync_apply(&fs, z_hat, sym_index);
        if (valid < 0) {
            fprintf(stderr, "rx2_frame_sync_apply failed at c=%d\n", c);
            fails = 1;
            break;
        }

        {
            float err = fabsf(fs.metric - expected_metric);
            if (err > max_err) max_err = err;
            if (err >= FLOAT_TOL) {
                if (fails < 5) {
                    fprintf(stderr, "METRIC FAIL c=%d got=%.9e expected=%.9e\n",
                            c, fs.metric, expected_metric);
                }
                fails++;
            }
            compared++;
        }
        {
            float err = fabsf(fs.frame_sync_even - expected_even);
            if (err > max_err) max_err = err;
            if (err >= FLOAT_TOL) {
                if (fails < 5) {
                    fprintf(stderr, "EVEN FAIL c=%d got=%.9e expected=%.9e\n",
                            c, fs.frame_sync_even, expected_even);
                }
                fails++;
            }
            compared++;
        }
        {
            float err = fabsf(fs.frame_sync_odd - expected_odd);
            if (err > max_err) max_err = err;
            if (err >= FLOAT_TOL) {
                if (fails < 5) {
                    fprintf(stderr, "ODD FAIL c=%d got=%.9e expected=%.9e\n",
                            c, fs.frame_sync_odd, expected_odd);
                }
                fails++;
            }
            compared++;
        }
        if (valid != expected_valid) {
            if (fails < 5) {
                fprintf(stderr, "VALID FAIL c=%d got=%d expected=%d\n",
                        c, valid, (int)expected_valid);
            }
            fails++;
        }
        for (int i = 0; i < FRAME_SYNC_INPUT_DIM; i++) {
            float err = fabsf(fs.az_hat[i] - expected_az_hat[i]);
            if (err > max_err) max_err = err;
            if (err >= FLOAT_TOL) {
                if (fails < 5) {
                    fprintf(stderr,
                            "AZ FAIL c=%d i=%d got=%.9e expected=%.9e\n",
                            c, i, fs.az_hat[i], expected_az_hat[i]);
                }
                fails++;
            }
            compared++;
        }
    }

    fprintf(stderr, "cases=%d compared=%d max_err=%.3e tol=%.0e fails=%d\n",
            ncases, compared, max_err, FLOAT_TOL, fails);

    rx2_frame_sync_destroy(&fs);
    return fails == 0 ? 0 : 1;
}
