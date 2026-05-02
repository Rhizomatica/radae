/*
 * Bit-accuracy test for complex_bpf_process().
 *
 * Stdin format (binary):
 *   int32    ntap
 *   float32  Fs_Hz
 *   float32  bandwidth_Hz
 *   float32  centre_freq_Hz
 *   int32    max_len
 *   int32    ncases
 *   repeated ncases times:
 *     int32        nin
 *     complex64[]  nin input samples
 *     complex64[]  nin expected output samples
 *
 * Exits 0 if every sample passes |C - Python| < TOL, else 1.
 *
 * Copyright (c) 2026 Rhizomatica, BSD-3-Clause
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "complex_bpf.h"

#define TOL 5e-5f

static float comp_abs_err(COMP a, COMP b) {
    float dr = a.real - b.real;
    float di = a.imag - b.imag;
    return hypotf(dr, di);
}

int main(void) {
    _Static_assert(sizeof(COMP) == 2 * sizeof(float),
                   "COMP must be two packed float values");

    int32_t ntap;
    float Fs_Hz, bandwidth_Hz, centre_freq_Hz;
    int32_t max_len, ncases;
    float max_abs_err = 0.0f;
    int total = 0;
    int fails = 0;

    if (fread(&ntap, sizeof(ntap), 1, stdin) != 1 ||
        fread(&Fs_Hz, sizeof(Fs_Hz), 1, stdin) != 1 ||
        fread(&bandwidth_Hz, sizeof(bandwidth_Hz), 1, stdin) != 1 ||
        fread(&centre_freq_Hz, sizeof(centre_freq_Hz), 1, stdin) != 1 ||
        fread(&max_len, sizeof(max_len), 1, stdin) != 1 ||
        fread(&ncases, sizeof(ncases), 1, stdin) != 1) {
        fprintf(stderr, "truncated header\n");
        return 1;
    }

    struct complex_bpf bpf;
    if (complex_bpf_init(&bpf, ntap, Fs_Hz, bandwidth_Hz, centre_freq_Hz, max_len) != 0) {
        fprintf(stderr, "complex_bpf_init failed\n");
        return 1;
    }

    COMP *in = (COMP *)malloc((size_t)max_len * sizeof(*in));
    COMP *expected = (COMP *)malloc((size_t)max_len * sizeof(*expected));
    COMP *got = (COMP *)malloc((size_t)max_len * sizeof(*got));
    if (!in || !expected || !got) {
        fprintf(stderr, "allocation failure\n");
        complex_bpf_destroy(&bpf);
        free(in);
        free(expected);
        free(got);
        return 1;
    }

    for (int c = 0; c < ncases; c++) {
        int32_t nin;
        if (fread(&nin, sizeof(nin), 1, stdin) != 1) {
            fprintf(stderr, "truncated case header at c=%d\n", c);
            fails = 1;
            break;
        }
        if (nin < 0 || nin > max_len) {
            fprintf(stderr, "invalid nin=%d at c=%d\n", (int)nin, c);
            fails = 1;
            break;
        }
        if (fread(in, sizeof(*in), (size_t)nin, stdin) != (size_t)nin ||
            fread(expected, sizeof(*expected), (size_t)nin, stdin) != (size_t)nin) {
            fprintf(stderr, "truncated payload at c=%d nin=%d\n", c, (int)nin);
            fails = 1;
            break;
        }

        if (complex_bpf_process(&bpf, in, nin, got) != 0) {
            fprintf(stderr, "complex_bpf_process failed at c=%d\n", c);
            fails = 1;
            break;
        }

        for (int i = 0; i < nin; i++) {
            float err = comp_abs_err(got[i], expected[i]);
            if (err > max_abs_err) max_abs_err = err;
            if (err >= TOL) {
                if (fails < 5) {
                    fprintf(stderr,
                            "FAIL c=%d i=%d got=(%.9e, %.9e) expected=(%.9e, %.9e) err=%.3e\n",
                            c, i,
                            got[i].real, got[i].imag,
                            expected[i].real, expected[i].imag,
                            err);
                }
                fails++;
            }
            total++;
        }
    }

    fprintf(stderr, "samples=%d max_abs_err=%.3e tol=%.0e fails=%d\n",
            total, max_abs_err, TOL, fails);

    complex_bpf_destroy(&bpf);
    free(in);
    free(expected);
    free(got);
    return fails == 0 ? 0 : 1;
}
