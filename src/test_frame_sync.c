/*
 * Bit-accuracy test for frame_sync_forward().
 *
 * Stdin format (binary):
 *   [input_dim floats] [1 float: expected output]
 *   ... repeated ...
 *
 * Exits 0 if every pair passes |C_out - Python_out| < 1e-5, else 1.
 * Prints max absolute error on stderr.
 *
 * Copyright (c) 2026 Rhizomatica, BSD-3-Clause
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "frame_sync.h"

#define TOL 1e-5f

int main(int argc, char *argv[]) {
    float in[FRAME_SYNC_INPUT_DIM];
    float expected;
    float max_abs_err = 0.0f;
    int n = 0;
    int fails = 0;

    while (fread(in, sizeof(float), FRAME_SYNC_INPUT_DIM, stdin)
           == (size_t)FRAME_SYNC_INPUT_DIM) {
        if (fread(&expected, sizeof(float), 1, stdin) != 1) {
            fprintf(stderr, "truncated input at n=%d\n", n);
            return 1;
        }
        float got = frame_sync_forward(in);
        float err = fabsf(got - expected);
        if (err > max_abs_err) max_abs_err = err;
        if (err >= TOL) {
            if (fails < 5) {
                fprintf(stderr, "FAIL n=%d got=%.9e expected=%.9e err=%.3e\n",
                        n, got, expected, err);
            }
            fails++;
        }
        n++;
    }

    fprintf(stderr, "n=%d max_abs_err=%.3e tol=%.0e fails=%d\n",
            n, max_abs_err, TOL, fails);
    return fails == 0 ? 0 : 1;
}
