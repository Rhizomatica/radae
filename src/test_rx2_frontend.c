/*
 * Streaming bit-accuracy test for rx2_frontend_apply().
 *
 * Stdin format (binary):
 *   int32    sym_len
 *   int32    max_nin
 *   int32    ncases
 *   repeated ncases times:
 *     int32        agc_en
 *     int32        nin
 *     complex64[]  nin input samples
 *     float32      expected gain
 *     complex64[]  3*sym_len expected rx_buf state
 *
 * Exits 0 if all gains and rx_buf samples pass tolerance.
 *
 * Copyright (c) 2026 Rhizomatica, BSD-3-Clause
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "rx2_frontend.h"

#define GAIN_TOL 1e-5f
#define BUF_TOL  1e-5f

static float comp_abs_err(COMP a, COMP b) {
    float dr = a.real - b.real;
    float di = a.imag - b.imag;
    return hypotf(dr, di);
}

int main(void) {
    _Static_assert(sizeof(COMP) == 2 * sizeof(float),
                   "COMP must be two packed float values");

    int32_t sym_len, max_nin, ncases;
    if (fread(&sym_len, sizeof(sym_len), 1, stdin) != 1 ||
        fread(&max_nin, sizeof(max_nin), 1, stdin) != 1 ||
        fread(&ncases, sizeof(ncases), 1, stdin) != 1) {
        fprintf(stderr, "truncated header\n");
        return 1;
    }

    struct rx2_frontend fe;
    if (rx2_frontend_init(&fe, sym_len, max_nin, 1) != 0) {
        fprintf(stderr, "rx2_frontend_init failed\n");
        return 1;
    }

    COMP *in = (COMP *)malloc((size_t)max_nin * sizeof(*in));
    COMP *expected_buf = (COMP *)malloc((size_t)(3 * sym_len) * sizeof(*expected_buf));
    if (!in || !expected_buf) {
        fprintf(stderr, "allocation failure\n");
        rx2_frontend_destroy(&fe);
        free(in);
        free(expected_buf);
        return 1;
    }

    float max_gain_err = 0.0f;
    float max_buf_err = 0.0f;
    int fails = 0;
    int total_buf = 0;

    for (int c = 0; c < ncases; c++) {
        int32_t agc_en, nin;
        float expected_gain, gain;
        if (fread(&agc_en, sizeof(agc_en), 1, stdin) != 1 ||
            fread(&nin, sizeof(nin), 1, stdin) != 1) {
            fprintf(stderr, "truncated case header at c=%d\n", c);
            fails = 1;
            break;
        }
        if (nin < 0 || nin > max_nin) {
            fprintf(stderr, "invalid nin=%d at c=%d\n", (int)nin, c);
            fails = 1;
            break;
        }
        if (fread(in, sizeof(*in), (size_t)nin, stdin) != (size_t)nin ||
            fread(&expected_gain, sizeof(expected_gain), 1, stdin) != 1 ||
            fread(expected_buf, sizeof(*expected_buf), (size_t)(3 * sym_len), stdin)
                != (size_t)(3 * sym_len)) {
            fprintf(stderr, "truncated payload at c=%d\n", c);
            fails = 1;
            break;
        }

        rx2_frontend_set_agc(&fe, agc_en);
        if (rx2_frontend_apply(&fe, in, nin, &gain) != 0) {
            fprintf(stderr, "rx2_frontend_apply failed at c=%d\n", c);
            fails = 1;
            break;
        }

        float gain_err = fabsf(gain - expected_gain);
        if (gain_err > max_gain_err) max_gain_err = gain_err;
        if (gain_err >= GAIN_TOL) {
            if (fails < 5) {
                fprintf(stderr, "GAIN FAIL c=%d got=%.9e expected=%.9e err=%.3e\n",
                        c, gain, expected_gain, gain_err);
            }
            fails++;
        }

        for (int i = 0; i < 3 * sym_len; i++) {
            float err = comp_abs_err(fe.rx_buf[i], expected_buf[i]);
            if (err > max_buf_err) max_buf_err = err;
            if (err >= BUF_TOL) {
                if (fails < 5) {
                    fprintf(stderr,
                            "BUF FAIL c=%d i=%d got=(%.9e, %.9e) expected=(%.9e, %.9e) err=%.3e\n",
                            c, i,
                            fe.rx_buf[i].real, fe.rx_buf[i].imag,
                            expected_buf[i].real, expected_buf[i].imag,
                            err);
                }
                fails++;
            }
            total_buf++;
        }
    }

    fprintf(stderr,
            "cases=%d max_gain_err=%.3e gain_tol=%.0e max_buf_err=%.3e buf_tol=%.0e fails=%d\n",
            ncases, max_gain_err, GAIN_TOL, max_buf_err, BUF_TOL, fails);

    rx2_frontend_destroy(&fe);
    free(in);
    free(expected_buf);
    return fails == 0 ? 0 : 1;
}
