/*
 * Streaming bit-accuracy test for rx2_demod_apply().
 *
 * Stdin format (binary):
 *   int32      M
 *   int32      Ncp
 *   int32      Ns
 *   int32      Nc
 *   int32      sym_len
 *   int32      latent_dim
 *   int32      ncases
 *   float32    Fs
 *   int32      time_offset
 *   int32      correct_time_offset
 *   float32[]  Nc carrier frequencies w
 *   repeated ncases times:
 *     complex64[]  3*sym_len rx_buf snapshot
 *     float32      delta_hat
 *     float32      freq_offset
 *     complex64    expected rx_phase
 *     complex64[]  M expected rx_sym_td
 *     complex64[]  Ns*sym_len expected rx_i
 *     float32[]    latent_dim expected z_hat
 *
 * Copyright (c) 2026 Rhizomatica, BSD-3-Clause
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "rx2_demod.h"

#define COMP_TOL 2e-5f
#define FLOAT_TOL 5e-5f

static float comp_abs_err(COMP a, COMP b) {
    float dr = a.real - b.real;
    float di = a.imag - b.imag;
    return hypotf(dr, di);
}

int main(void) {
    _Static_assert(sizeof(COMP) == 2 * sizeof(float),
                   "COMP must be two packed float values");

    int32_t M, Ncp, Ns, Nc, sym_len, latent_dim, ncases;
    int32_t time_offset, correct_time_offset;
    float Fs;
    if (fread(&M, sizeof(M), 1, stdin) != 1 ||
        fread(&Ncp, sizeof(Ncp), 1, stdin) != 1 ||
        fread(&Ns, sizeof(Ns), 1, stdin) != 1 ||
        fread(&Nc, sizeof(Nc), 1, stdin) != 1 ||
        fread(&sym_len, sizeof(sym_len), 1, stdin) != 1 ||
        fread(&latent_dim, sizeof(latent_dim), 1, stdin) != 1 ||
        fread(&ncases, sizeof(ncases), 1, stdin) != 1 ||
        fread(&Fs, sizeof(Fs), 1, stdin) != 1 ||
        fread(&time_offset, sizeof(time_offset), 1, stdin) != 1 ||
        fread(&correct_time_offset, sizeof(correct_time_offset), 1, stdin) != 1) {
        fprintf(stderr, "truncated header\n");
        return 1;
    }

    float *w = (float *)malloc((size_t)Nc * sizeof(*w));
    COMP *rx_buf = (COMP *)malloc((size_t)(3 * sym_len) * sizeof(*rx_buf));
    COMP *expected_rx_sym_td = (COMP *)malloc((size_t)M * sizeof(*expected_rx_sym_td));
    COMP *expected_rx_i = (COMP *)malloc((size_t)(Ns * sym_len) * sizeof(*expected_rx_i));
    float *expected_z_hat = (float *)malloc((size_t)latent_dim * sizeof(*expected_z_hat));
    float *actual_z_hat = (float *)malloc((size_t)latent_dim * sizeof(*actual_z_hat));
    if (!w || !rx_buf || !expected_rx_sym_td || !expected_rx_i || !expected_z_hat || !actual_z_hat) {
        fprintf(stderr, "allocation failure\n");
        free(w);
        free(rx_buf);
        free(expected_rx_sym_td);
        free(expected_rx_i);
        free(expected_z_hat);
        free(actual_z_hat);
        return 1;
    }
    if (fread(w, sizeof(*w), (size_t)Nc, stdin) != (size_t)Nc) {
        fprintf(stderr, "truncated w vector\n");
        free(w);
        free(rx_buf);
        free(expected_rx_sym_td);
        free(expected_rx_i);
        free(expected_z_hat);
        free(actual_z_hat);
        return 1;
    }

    struct rx2_demod dm;
    if (rx2_demod_init(&dm, M, Ncp, Ns, Nc, Fs, time_offset, correct_time_offset, w) != 0) {
        fprintf(stderr, "rx2_demod_init failed\n");
        free(w);
        free(rx_buf);
        free(expected_rx_sym_td);
        free(expected_rx_i);
        free(expected_z_hat);
        free(actual_z_hat);
        return 1;
    }
    if (sym_len != dm.sym_len || latent_dim != dm.latent_dim) {
        fprintf(stderr, "header mismatch: sym_len=%d/%d latent_dim=%d/%d\n",
                (int)sym_len, dm.sym_len, (int)latent_dim, dm.latent_dim);
        rx2_demod_destroy(&dm);
        free(w);
        free(rx_buf);
        free(expected_rx_sym_td);
        free(expected_rx_i);
        free(expected_z_hat);
        free(actual_z_hat);
        return 1;
    }

    float max_comp_err = 0.0f;
    float max_float_err = 0.0f;
    int fails = 0;
    int total_comp = 0;
    int total_float = 0;

    for (int c = 0; c < ncases; c++) {
        float delta_hat, freq_offset;
        COMP expected_rx_phase;
        if (fread(rx_buf, sizeof(*rx_buf), (size_t)(3 * sym_len), stdin) != (size_t)(3 * sym_len) ||
            fread(&delta_hat, sizeof(delta_hat), 1, stdin) != 1 ||
            fread(&freq_offset, sizeof(freq_offset), 1, stdin) != 1 ||
            fread(&expected_rx_phase, sizeof(expected_rx_phase), 1, stdin) != 1 ||
            fread(expected_rx_sym_td, sizeof(*expected_rx_sym_td), (size_t)M, stdin) != (size_t)M ||
            fread(expected_rx_i, sizeof(*expected_rx_i), (size_t)(Ns * sym_len), stdin) != (size_t)(Ns * sym_len) ||
            fread(expected_z_hat, sizeof(*expected_z_hat), (size_t)latent_dim, stdin) != (size_t)latent_dim) {
            fprintf(stderr, "truncated payload at c=%d\n", c);
            fails = 1;
            break;
        }

        if (rx2_demod_apply(&dm, rx_buf, delta_hat, freq_offset, actual_z_hat) != 0) {
            fprintf(stderr, "rx2_demod_apply failed at c=%d\n", c);
            fails = 1;
            break;
        }

        {
            float err = comp_abs_err(dm.rx_phase, expected_rx_phase);
            if (err > max_comp_err) max_comp_err = err;
            if (err >= COMP_TOL) {
                if (fails < 5) {
                    fprintf(stderr,
                            "PHASE FAIL c=%d got=(%.9e, %.9e) expected=(%.9e, %.9e) err=%.3e\n",
                            c,
                            dm.rx_phase.real, dm.rx_phase.imag,
                            expected_rx_phase.real, expected_rx_phase.imag,
                            err);
                }
                fails++;
            }
            total_comp++;
        }

        for (int i = 0; i < M; i++) {
            float err = comp_abs_err(dm.rx_sym_td[i], expected_rx_sym_td[i]);
            if (err > max_comp_err) max_comp_err = err;
            if (err >= COMP_TOL) {
                if (fails < 5) {
                    fprintf(stderr,
                            "SYM FAIL c=%d i=%d got=(%.9e, %.9e) expected=(%.9e, %.9e) err=%.3e\n",
                            c, i,
                            dm.rx_sym_td[i].real, dm.rx_sym_td[i].imag,
                            expected_rx_sym_td[i].real, expected_rx_sym_td[i].imag,
                            err);
                }
                fails++;
            }
            total_comp++;
        }

        for (int i = 0; i < Ns * sym_len; i++) {
            float err = comp_abs_err(dm.rx_i[i], expected_rx_i[i]);
            if (err > max_comp_err) max_comp_err = err;
            if (err >= COMP_TOL) {
                if (fails < 5) {
                    fprintf(stderr,
                            "RXI FAIL c=%d i=%d got=(%.9e, %.9e) expected=(%.9e, %.9e) err=%.3e\n",
                            c, i,
                            dm.rx_i[i].real, dm.rx_i[i].imag,
                            expected_rx_i[i].real, expected_rx_i[i].imag,
                            err);
                }
                fails++;
            }
            total_comp++;
        }

        for (int i = 0; i < latent_dim; i++) {
            float err = fabsf(actual_z_hat[i] - expected_z_hat[i]);
            if (err > max_float_err) max_float_err = err;
            if (err >= FLOAT_TOL) {
                if (fails < 5) {
                    fprintf(stderr,
                            "ZHAT FAIL c=%d i=%d got=%.9e expected=%.9e err=%.3e\n",
                            c, i, actual_z_hat[i], expected_z_hat[i], err);
                }
                fails++;
            }
            total_float++;
        }
    }

    fprintf(stderr,
            "cases=%d comp=%d floats=%d max_comp_err=%.3e comp_tol=%.0e max_float_err=%.3e float_tol=%.0e fails=%d\n",
            ncases, total_comp, total_float, max_comp_err, COMP_TOL, max_float_err, FLOAT_TOL, fails);

    rx2_demod_destroy(&dm);
    free(w);
    free(rx_buf);
    free(expected_rx_sym_td);
    free(expected_rx_i);
    free(expected_z_hat);
    free(actual_z_hat);
    return fails == 0 ? 0 : 1;
}
