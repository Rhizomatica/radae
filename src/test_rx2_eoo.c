/*
 * Streaming bit-accuracy test for rx2_eoo_apply().
 *
 * Stdin format (binary):
 *   int32      M
 *   int32      Ncp
 *   int32      ncases
 *   complex64[] M pend time-domain symbol
 *   repeated ncases times:
 *     complex64[] M rx_sym_td
 *     float32     expected_eoo_corr
 *     float32     expected_eoo_smooth
 *     int32       expected_detect
 *
 * Copyright (c) 2026 Rhizomatica, BSD-3-Clause
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "rx2_eoo.h"

#define FLOAT_TOL 5e-5f

int main(void) {
    int32_t M, Ncp, ncases;
    if (fread(&M, sizeof(M), 1, stdin) != 1 ||
        fread(&Ncp, sizeof(Ncp), 1, stdin) != 1 ||
        fread(&ncases, sizeof(ncases), 1, stdin) != 1) {
        fprintf(stderr, "truncated header\n");
        return 1;
    }

    COMP *pend = (COMP *)malloc((size_t)M * sizeof(*pend));
    COMP *rx_sym_td = (COMP *)malloc((size_t)M * sizeof(*rx_sym_td));
    if (!pend || !rx_sym_td) {
        fprintf(stderr, "allocation failure\n");
        free(pend);
        free(rx_sym_td);
        return 1;
    }
    if (fread(pend, sizeof(*pend), (size_t)M, stdin) != (size_t)M) {
        fprintf(stderr, "truncated pend\n");
        free(pend);
        free(rx_sym_td);
        return 1;
    }

    struct rx2_eoo eoo;
    if (rx2_eoo_init(&eoo, M, Ncp, pend) != 0) {
        fprintf(stderr, "rx2_eoo_init failed\n");
        free(pend);
        free(rx_sym_td);
        return 1;
    }

    float max_err = 0.0f;
    int fails = 0;

    for (int c = 0; c < ncases; c++) {
        float expected_corr, expected_smooth;
        int32_t expected_detect;
        int detect;
        if (fread(rx_sym_td, sizeof(*rx_sym_td), (size_t)M, stdin) != (size_t)M ||
            fread(&expected_corr, sizeof(expected_corr), 1, stdin) != 1 ||
            fread(&expected_smooth, sizeof(expected_smooth), 1, stdin) != 1 ||
            fread(&expected_detect, sizeof(expected_detect), 1, stdin) != 1) {
            fprintf(stderr, "truncated payload at c=%d\n", c);
            fails = 1;
            break;
        }

        detect = rx2_eoo_apply(&eoo, rx_sym_td);
        if (detect < 0) {
            fprintf(stderr, "rx2_eoo_apply failed at c=%d\n", c);
            fails = 1;
            break;
        }

        {
            float corr_err = fabsf(eoo.eoo_corr - expected_corr);
            float smooth_err = fabsf(eoo.eoo_smooth - expected_smooth);
            if (corr_err > max_err) max_err = corr_err;
            if (smooth_err > max_err) max_err = smooth_err;
            if (corr_err >= FLOAT_TOL || smooth_err >= FLOAT_TOL || detect != expected_detect) {
                if (fails < 5) {
                    fprintf(stderr,
                            "EOO FAIL c=%d corr=%.9e/%.9e smooth=%.9e/%.9e detect=%d/%d\n",
                            c,
                            eoo.eoo_corr, expected_corr,
                            eoo.eoo_smooth, expected_smooth,
                            detect, (int)expected_detect);
                }
                fails++;
            }
        }
    }

    fprintf(stderr,
            "cases=%d max_err=%.3e tol=%.0e fails=%d\n",
            ncases, max_err, FLOAT_TOL, fails);

    rx2_eoo_destroy(&eoo);
    free(pend);
    free(rx_sym_td);
    return fails == 0 ? 0 : 1;
}
