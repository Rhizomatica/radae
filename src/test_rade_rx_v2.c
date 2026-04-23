/*
 * Streaming API-level parity test for rade_rx_v2_pure_c().
 *
 * Binary layout:
 *   int32      output_dim
 *   int32      ncases
 *   repeated:
 *     int32      nin
 *     complex64[] rx_in (nin)
 *     int32      expected_valid
 *     int32      expected_next_nin
 *     int32      expected_sync
 *     int32      expected_snr
 *     float32    expected_freq_offset
 *     float32[]  expected_features (output_dim)
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rade_api.h"

#define FREQ_TOL 5e-4f
#define FEATURE_TOL 5e-2f

int main(void) {
    int32_t output_dim, ncases;
    struct rade *r;
    float *features_out = NULL;
    RADE_COMP *rx_in = NULL;
    float *expected_features = NULL;
    int fails = 0;
    int compared = 0;
    float max_err = 0.0f;

    if (fread(&output_dim, sizeof(output_dim), 1, stdin) != 1 ||
        fread(&ncases, sizeof(ncases), 1, stdin) != 1) {
        fprintf(stderr, "truncated header\n");
        return 1;
    }

    rade_initialize();
    if (rade_rx_v2_pure_c_open("unexpected-model",
                               RADE_RX_V2_COMPILED_FRAME_SYNC_MODEL_NAME,
                               RADE_VERBOSE_0) != NULL) {
        fprintf(stderr, "unexpected success for invalid model identifier\n");
        rade_finalize();
        return 1;
    }
    if (rade_rx_v2_pure_c_open(RADE_RX_V2_COMPILED_MODEL_NAME,
                               "unexpected-frame-sync",
                               RADE_VERBOSE_0) != NULL) {
        fprintf(stderr, "unexpected success for invalid frame sync identifier\n");
        rade_finalize();
        return 1;
    }
    r = rade_rx_v2_pure_c_open(
        RADE_RX_V2_COMPILED_MODEL_NAME,
        RADE_RX_V2_COMPILED_FRAME_SYNC_MODEL_NAME,
        RADE_VERBOSE_0);
    if (!r) {
        fprintf(stderr, "rade_rx_v2_pure_c_open failed\n");
        return 1;
    }
    if (output_dim != rade_n_features_in_out(r)) {
        fprintf(stderr, "output_dim mismatch %d/%d\n", (int)output_dim, rade_n_features_in_out(r));
        rade_close(r);
        rade_finalize();
        return 1;
    }

    features_out = (float *)malloc(sizeof(*features_out) * (size_t)output_dim);
    expected_features = (float *)malloc(sizeof(*expected_features) * (size_t)output_dim);
    rx_in = (RADE_COMP *)malloc(sizeof(*rx_in) * (size_t)rade_nin_max(r));
    if (!features_out || !expected_features || !rx_in) {
        fprintf(stderr, "allocation failure\n");
        rade_close(r);
        rade_finalize();
        free(features_out);
        free(expected_features);
        free(rx_in);
        return 1;
    }

    for (int c = 0; c < ncases; c++) {
        int32_t nin, expected_valid, expected_next_nin, expected_sync, expected_snr;
        float expected_freq_offset;
        int has_eoo_out = -1;
        float eoo_dummy = 0.0f;
        int n_out;

        if (fread(&nin, sizeof(nin), 1, stdin) != 1 ||
            nin <= 0 || nin > rade_nin_max(r) ||
            fread(rx_in, sizeof(*rx_in), (size_t)nin, stdin) != (size_t)nin ||
            fread(&expected_valid, sizeof(expected_valid), 1, stdin) != 1 ||
            fread(&expected_next_nin, sizeof(expected_next_nin), 1, stdin) != 1 ||
            fread(&expected_sync, sizeof(expected_sync), 1, stdin) != 1 ||
            fread(&expected_snr, sizeof(expected_snr), 1, stdin) != 1 ||
            fread(&expected_freq_offset, sizeof(expected_freq_offset), 1, stdin) != 1 ||
            fread(expected_features, sizeof(*expected_features), (size_t)output_dim, stdin) != (size_t)output_dim) {
            fprintf(stderr, "truncated payload at c=%d\n", c);
            fails = 1;
            break;
        }

        memset(features_out, 0, sizeof(*features_out) * (size_t)output_dim);
        n_out = rade_rx_v2_pure_c(r, features_out, &has_eoo_out, &eoo_dummy, rx_in);

        if (!!n_out != !!expected_valid) {
            if (fails < 5) fprintf(stderr, "VALID FAIL c=%d got=%d expected=%d\n", c, !!n_out, !!expected_valid);
            fails++;
        }
        compared++;
        if (rade_nin(r) != expected_next_nin) {
            if (fails < 5) fprintf(stderr, "NIN FAIL c=%d got=%d expected=%d\n", c, rade_nin(r), (int)expected_next_nin);
            fails++;
        }
        compared++;
        if (rade_sync(r) != expected_sync) {
            if (fails < 5) fprintf(stderr, "SYNC FAIL c=%d got=%d expected=%d\n", c, rade_sync(r), (int)expected_sync);
            fails++;
        }
        compared++;
        if (rade_snrdB_3k_est(r) != expected_snr) {
            if (fails < 5) fprintf(stderr, "SNR FAIL c=%d got=%d expected=%d\n", c, rade_snrdB_3k_est(r), (int)expected_snr);
            fails++;
        }
        compared++;
        {
            float err = fabsf(rade_freq_offset(r) - expected_freq_offset);
            if (err > max_err) max_err = err;
            if (err >= FREQ_TOL) {
                if (fails < 5) fprintf(stderr, "FREQ FAIL c=%d got=%.9e expected=%.9e\n", c, rade_freq_offset(r), expected_freq_offset);
                fails++;
            }
            compared++;
        }
        if (has_eoo_out != 0) {
            if (fails < 5) fprintf(stderr, "EOO FAIL c=%d got=%d expected=0\n", c, has_eoo_out);
            fails++;
        }
        compared++;

        for (int i = 0; i < output_dim; i++) {
            float err = fabsf(features_out[i] - expected_features[i]);
            if (err > max_err) max_err = err;
            if (err >= FEATURE_TOL) {
                if (fails < 5) {
                    fprintf(stderr, "FEAT FAIL c=%d i=%d got=%.9e expected=%.9e err=%.3e\n",
                            c, i, features_out[i], expected_features[i], err);
                }
                fails++;
            }
            compared++;
        }
    }

    fprintf(stderr, "cases=%d compared=%d max_err=%.3e freq_tol=%.0e feat_tol=%.0e fails=%d\n",
            ncases, compared, max_err, FREQ_TOL, FEATURE_TOL, fails);

    rade_close(r);
    rade_finalize();
    free(features_out);
    free(expected_features);
    free(rx_in);
    return fails == 0 ? 0 : 1;
}
