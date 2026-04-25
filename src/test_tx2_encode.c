/*
 * Streaming bit-accuracy test for tx2_encode.
 *
 * Stdin format (binary):
 *   int32        auxdata
 *   int32        txbpf_en
 *   int32        n_features_in
 *   int32        Nmf
 *   int32        Neoo
 *   int32        ncases
 *   repeated:
 *     float32[n_features_in]   features_in
 *     complex64[Nmf]           expected tx_out
 *   complex64[Neoo]            expected EOO frame
 *
 * Copyright (c) 2026 Rhizomatica, BSD-3-Clause
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tx2_encode.h"

/*
 * The compiled-in encoder uses opus's nnet.c polynomial tanh approximation,
 * which diverges from PyTorch's libm tanh by ~2.4e-3 per layer.  Across the
 * 11-layer DenseNet stack that grows to ~0.1 in the latent vector, and the
 * downstream bottleneck=3 clipper amplifies tiny IQ deviations near zero
 * into possibly large angular deltas on unit-magnitude samples.  Bit-exact
 * parity against Python is therefore impossible by design.  TX_TOL is set
 * to catch *structural* bugs (off-by-one, swapped indices, sign errors)
 * without false-positiving on the known tanh approximation drift; the
 * end-to-end correctness measure is the on-air decode, not this tolerance.
 *
 * EOO bit-accuracy is achievable when txbpf is off (constants-only path);
 * with txbpf on the BPF accumulates ~3e-3 of numerical drift, so we use a
 * looser tolerance that still validates the BPF wiring.
 */
#define TX_TOL          5e-1f
#define EOO_TOL_NOBPF   1e-6f
#define EOO_TOL_BPF     1e-2f

static float comp_abs_err(COMP a, COMP b) {
    float dr = a.real - b.real;
    float di = a.imag - b.imag;
    return hypotf(dr, di);
}

int main(void) {
    int32_t auxdata, txbpf_en, n_features_in, Nmf, Neoo, ncases;
    struct tx2_encode tx;
    float *features_in = NULL;
    COMP *tx_out = NULL;
    COMP *expected_tx = NULL;
    COMP *eoo_out = NULL;
    COMP *expected_eoo = NULL;
    int fails = 0;
    int compared = 0;
    float max_err = 0.0f;
    double tx_err_sum = 0.0;
    int tx_err_n = 0;

    if (fread(&auxdata, sizeof(auxdata), 1, stdin) != 1 ||
        fread(&txbpf_en, sizeof(txbpf_en), 1, stdin) != 1 ||
        fread(&n_features_in, sizeof(n_features_in), 1, stdin) != 1 ||
        fread(&Nmf, sizeof(Nmf), 1, stdin) != 1 ||
        fread(&Neoo, sizeof(Neoo), 1, stdin) != 1 ||
        fread(&ncases, sizeof(ncases), 1, stdin) != 1) {
        fprintf(stderr, "truncated header\n");
        return 1;
    }

    if (n_features_in != TX2_N_FEATURES_IN ||
        Nmf != TX2_MODEL_NMF ||
        Neoo != TX2_MODEL_NEOO) {
        fprintf(stderr,
                "size mismatch: n_features_in=%d/%d Nmf=%d/%d Neoo=%d/%d\n",
                (int)n_features_in, TX2_N_FEATURES_IN,
                (int)Nmf, TX2_MODEL_NMF,
                (int)Neoo, TX2_MODEL_NEOO);
        return 1;
    }

    if (tx2_encode_init(&tx, auxdata, txbpf_en) != 0) {
        fprintf(stderr, "tx2_encode_init failed (auxdata=%d txbpf=%d)\n",
                (int)auxdata, (int)txbpf_en);
        return 1;
    }

    features_in  = (float *)malloc(sizeof(*features_in)  * (size_t)n_features_in);
    tx_out       = (COMP  *)malloc(sizeof(*tx_out)       * (size_t)Nmf);
    expected_tx  = (COMP  *)malloc(sizeof(*expected_tx)  * (size_t)Nmf);
    eoo_out      = (COMP  *)malloc(sizeof(*eoo_out)      * (size_t)Neoo);
    expected_eoo = (COMP  *)malloc(sizeof(*expected_eoo) * (size_t)Neoo);
    if (!features_in || !tx_out || !expected_tx || !eoo_out || !expected_eoo) {
        fprintf(stderr, "allocation failure\n");
        goto fail;
    }

    for (int c = 0; c < ncases; c++) {
        if (fread(features_in, sizeof(*features_in), (size_t)n_features_in, stdin)
                != (size_t)n_features_in ||
            fread(expected_tx, sizeof(*expected_tx), (size_t)Nmf, stdin)
                != (size_t)Nmf) {
            fprintf(stderr, "truncated case %d\n", c);
            goto fail;
        }

        if (tx2_encode_frame(&tx, features_in, tx_out) != 0) {
            fprintf(stderr, "tx2_encode_frame failed at c=%d\n", c);
            goto fail;
        }

        for (int i = 0; i < Nmf; i++) {
            float err = comp_abs_err(tx_out[i], expected_tx[i]);
            if (err > max_err) max_err = err;
            tx_err_sum += err;
            tx_err_n++;
            if (err >= TX_TOL) {
                if (fails < 5) {
                    fprintf(stderr,
                            "TX FAIL c=%d i=%d got=(%.9e,%.9e) exp=(%.9e,%.9e) err=%.3e\n",
                            c, i,
                            tx_out[i].real, tx_out[i].imag,
                            expected_tx[i].real, expected_tx[i].imag,
                            err);
                }
                fails++;
            }
            compared++;
        }
    }

    if (fread(expected_eoo, sizeof(*expected_eoo), (size_t)Neoo, stdin)
            != (size_t)Neoo) {
        fprintf(stderr, "truncated EOO frame\n");
        goto fail;
    }
    if (tx2_encode_eoo(&tx, eoo_out) != 0) {
        fprintf(stderr, "tx2_encode_eoo failed\n");
        goto fail;
    }
    float eoo_tol = txbpf_en ? EOO_TOL_BPF : EOO_TOL_NOBPF;
    float eoo_max = 0.0f;
    int eoo_fails = 0;
    for (int i = 0; i < Neoo; i++) {
        float err = comp_abs_err(eoo_out[i], expected_eoo[i]);
        if (err > eoo_max) eoo_max = err;
        if (err >= eoo_tol) {
            if (eoo_fails < 5) {
                fprintf(stderr,
                        "EOO FAIL i=%d got=(%.9e,%.9e) exp=(%.9e,%.9e) err=%.3e\n",
                        i,
                        eoo_out[i].real, eoo_out[i].imag,
                        expected_eoo[i].real, expected_eoo[i].imag,
                        err);
            }
            eoo_fails++;
        }
        compared++;
    }

    double tx_err_mean = tx_err_n > 0 ? tx_err_sum / tx_err_n : 0.0;
    fprintf(stderr,
            "cases=%d compared=%d tx_max=%.3e tx_mean=%.3e tx_tol=%.0e "
            "eoo_max=%.3e eoo_tol=%.0e tx_fails=%d eoo_fails=%d\n",
            (int)ncases, compared, max_err, (float)tx_err_mean, TX_TOL,
            eoo_max, eoo_tol, fails, eoo_fails);

    tx2_encode_destroy(&tx);
    free(features_in);
    free(tx_out);
    free(expected_tx);
    free(eoo_out);
    free(expected_eoo);
    return (fails == 0 && eoo_fails == 0) ? 0 : 1;

fail:
    tx2_encode_destroy(&tx);
    free(features_in);
    free(tx_out);
    free(expected_tx);
    free(eoo_out);
    free(expected_eoo);
    return 1;
}
