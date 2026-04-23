/*
 * Streaming bit-accuracy test for rx2_coarse_sync_apply().
 *
 * Stdin format (binary):
 *   int32      M
 *   int32      Ncp
 *   int32      sym_len
 *   int32      ncases
 *   float32    Fs
 *   float32    B_bpf
 *   repeated ncases times:
 *     complex64[]  3*sym_len rx_buf snapshot
 *     complex64[]  sym_len expected Ry_smooth
 *     int32        expected delta_hat_g
 *     float32      expected Ry_max
 *     float32      expected Ry_min
 *     int32        expected sig_det
 *     int32        expected sine_det
 *     float32      expected snr_est_dB
 *
 * Copyright (c) 2026 Rhizomatica, BSD-3-Clause
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "rx2_coarse_sync.h"

#define COMP_TOL 1e-5f
#define FLOAT_TOL 1e-5f

static float comp_abs_err(COMP a, COMP b) {
    float dr = a.real - b.real;
    float di = a.imag - b.imag;
    return hypotf(dr, di);
}

int main(void) {
    _Static_assert(sizeof(COMP) == 2 * sizeof(float),
                   "COMP must be two packed float values");

    int32_t M, Ncp, sym_len, ncases;
    float Fs, B_bpf;
    if (fread(&M, sizeof(M), 1, stdin) != 1 ||
        fread(&Ncp, sizeof(Ncp), 1, stdin) != 1 ||
        fread(&sym_len, sizeof(sym_len), 1, stdin) != 1 ||
        fread(&ncases, sizeof(ncases), 1, stdin) != 1 ||
        fread(&Fs, sizeof(Fs), 1, stdin) != 1 ||
        fread(&B_bpf, sizeof(B_bpf), 1, stdin) != 1) {
        fprintf(stderr, "truncated header\n");
        return 1;
    }

    struct rx2_coarse_sync cs;
    if (rx2_coarse_sync_init(&cs, M, Ncp, Fs, B_bpf) != 0) {
        fprintf(stderr, "rx2_coarse_sync_init failed\n");
        return 1;
    }
    if (sym_len != cs.sym_len) {
        fprintf(stderr, "sym_len mismatch header=%d init=%d\n", (int)sym_len, cs.sym_len);
        rx2_coarse_sync_destroy(&cs);
        return 1;
    }

    COMP *rx_buf = (COMP *)malloc((size_t)(3 * sym_len) * sizeof(*rx_buf));
    COMP *expected_Ry_smooth = (COMP *)malloc((size_t)sym_len * sizeof(*expected_Ry_smooth));
    if (!rx_buf || !expected_Ry_smooth) {
        fprintf(stderr, "allocation failure\n");
        rx2_coarse_sync_destroy(&cs);
        free(rx_buf);
        free(expected_Ry_smooth);
        return 1;
    }

    float max_comp_err = 0.0f;
    float max_float_err = 0.0f;
    int fails = 0;
    int total_comp = 0;

    for (int c = 0; c < ncases; c++) {
        int32_t expected_delta_hat_g, expected_sig_det, expected_sine_det;
        float expected_Ry_max, expected_Ry_min, expected_snr_est_dB;
        int sig_det, sine_det;

        if (fread(rx_buf, sizeof(*rx_buf), (size_t)(3 * sym_len), stdin)
                != (size_t)(3 * sym_len) ||
            fread(expected_Ry_smooth, sizeof(*expected_Ry_smooth), (size_t)sym_len, stdin)
                != (size_t)sym_len ||
            fread(&expected_delta_hat_g, sizeof(expected_delta_hat_g), 1, stdin) != 1 ||
            fread(&expected_Ry_max, sizeof(expected_Ry_max), 1, stdin) != 1 ||
            fread(&expected_Ry_min, sizeof(expected_Ry_min), 1, stdin) != 1 ||
            fread(&expected_sig_det, sizeof(expected_sig_det), 1, stdin) != 1 ||
            fread(&expected_sine_det, sizeof(expected_sine_det), 1, stdin) != 1 ||
            fread(&expected_snr_est_dB, sizeof(expected_snr_est_dB), 1, stdin) != 1) {
            fprintf(stderr, "truncated payload at c=%d\n", c);
            fails = 1;
            break;
        }

        if (rx2_coarse_sync_apply(&cs, rx_buf, &sig_det, &sine_det) != 0) {
            fprintf(stderr, "rx2_coarse_sync_apply failed at c=%d\n", c);
            fails = 1;
            break;
        }

        for (int i = 0; i < sym_len; i++) {
            float err = comp_abs_err(cs.Ry_smooth[i], expected_Ry_smooth[i]);
            if (err > max_comp_err) max_comp_err = err;
            if (err >= COMP_TOL) {
                if (fails < 5) {
                    fprintf(stderr,
                            "RY FAIL c=%d i=%d got=(%.9e, %.9e) expected=(%.9e, %.9e) err=%.3e\n",
                            c, i,
                            cs.Ry_smooth[i].real, cs.Ry_smooth[i].imag,
                            expected_Ry_smooth[i].real, expected_Ry_smooth[i].imag,
                            err);
                }
                fails++;
            }
            total_comp++;
        }

        float delta_err = fabsf((float)cs.delta_hat_g - (float)expected_delta_hat_g);
        float Ry_max_err = fabsf(cs.Ry_max - expected_Ry_max);
        float Ry_min_err = fabsf(cs.Ry_min - expected_Ry_min);
        float snr_err = fabsf(cs.snr_est_dB - expected_snr_est_dB);
        if (delta_err > max_float_err) max_float_err = delta_err;
        if (Ry_max_err > max_float_err) max_float_err = Ry_max_err;
        if (Ry_min_err > max_float_err) max_float_err = Ry_min_err;
        if (snr_err > max_float_err) max_float_err = snr_err;

        if (cs.delta_hat_g != expected_delta_hat_g ||
            Ry_max_err >= FLOAT_TOL ||
            Ry_min_err >= FLOAT_TOL ||
            snr_err >= FLOAT_TOL ||
            sig_det != expected_sig_det ||
            sine_det != expected_sine_det) {
            if (fails < 5) {
                fprintf(stderr,
                        "DET FAIL c=%d delta=%d/%d Ry_max=%.9e/%.9e Ry_min=%.9e/%.9e "
                        "sig=%d/%d sine=%d/%d snr=%.9e/%.9e\n",
                        c,
                        cs.delta_hat_g, (int)expected_delta_hat_g,
                        cs.Ry_max, expected_Ry_max,
                        cs.Ry_min, expected_Ry_min,
                        sig_det, (int)expected_sig_det,
                        sine_det, (int)expected_sine_det,
                        cs.snr_est_dB, expected_snr_est_dB);
            }
            fails++;
        }
    }

    fprintf(stderr,
            "cases=%d comp=%d max_comp_err=%.3e comp_tol=%.0e max_float_err=%.3e float_tol=%.0e fails=%d\n",
            ncases, total_comp, max_comp_err, COMP_TOL, max_float_err, FLOAT_TOL, fails);

    rx2_coarse_sync_destroy(&cs);
    free(rx_buf);
    free(expected_Ry_smooth);
    return fails == 0 ? 0 : 1;
}
