/*
 * Streaming parity test for the thin rx2_receiver coordinator.
 *
 * Binary layout:
 *   int32      auxdata
 *   int32      agc
 *   int32      limit_pitch
 *   int32      mute
 *   int32      hangover
 *   int32      timing_adj_at
 *   int32      reset_output_on_resync
 *   int32      M
 *   int32      Ncp
 *   int32      Ns
 *   int32      Nc
 *   float32    Fs
 *   float32    B_bpf
 *   int32      time_offset
 *   int32      correct_time_offset
 *   float32[]  w (Nc)
 *   complex64[] pend (M)
 *   int32      output_dim
 *   int32      ncases
 *   repeated ncases:
 *     int32      nin
 *     complex64[] rx_in (nin)
 *     int32      expected_state
 *     int32      expected_count
 *     int32      expected_count1
 *     int32      expected_n_acq
 *     int32      expected_s
 *     int32      expected_i
 *     int32      expected_timing_adj
 *     int32      expected_sig_det
 *     int32      expected_sine_det
 *     int32      expected_decoded_valid
 *     int32      expected_next_nin
 *     int32      expected_new_sig_delta_hat
 *     int32      expected_new_sig_f_hat
 *     float32    expected_gain
 *     float32    expected_delta_hat
 *     int32      expected_delta_hat_g
 *     float32    expected_freq_offset
 *     float32    expected_freq_offset_g
 *     float32    expected_Ry_max
 *     float32    expected_Ry_min
 *     float32    expected_snr_est_dB
 *     float32    expected_frame_sync_even
 *     float32    expected_frame_sync_odd
 *     float32    expected_eoo_smooth
 *     float32    expected_eoo_corr
 *     float32[]  expected_features (output_dim)
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rx2_receiver.h"

#define STATE_TOL 5e-4f
#define FEATURE_TOL 5e-2f

static int read_comp_vec(COMP *dst, int n) {
    return fread(dst, sizeof(*dst), (size_t)n, stdin) == (size_t)n ? 0 : -1;
}

int main(void) {
    int32_t auxdata, agc, limit_pitch, mute, hangover, timing_adj_at, reset_output_on_resync;
    int32_t M, Ncp, Ns, Nc, time_offset, correct_time_offset, output_dim, ncases;
    float Fs, B_bpf;
    float *w = NULL;
    COMP *pend = NULL;
    COMP *rx_in = NULL;
    float *features = NULL;
    float *expected_features = NULL;
    struct rx2_receiver rx;
    struct rx2_receiver_config cfg;
    int fails = 0;
    int compared = 0;
    float max_err = 0.0f;

    if (fread(&auxdata, sizeof(auxdata), 1, stdin) != 1 ||
        fread(&agc, sizeof(agc), 1, stdin) != 1 ||
        fread(&limit_pitch, sizeof(limit_pitch), 1, stdin) != 1 ||
        fread(&mute, sizeof(mute), 1, stdin) != 1 ||
        fread(&hangover, sizeof(hangover), 1, stdin) != 1 ||
        fread(&timing_adj_at, sizeof(timing_adj_at), 1, stdin) != 1 ||
        fread(&reset_output_on_resync, sizeof(reset_output_on_resync), 1, stdin) != 1 ||
        fread(&M, sizeof(M), 1, stdin) != 1 ||
        fread(&Ncp, sizeof(Ncp), 1, stdin) != 1 ||
        fread(&Ns, sizeof(Ns), 1, stdin) != 1 ||
        fread(&Nc, sizeof(Nc), 1, stdin) != 1 ||
        fread(&Fs, sizeof(Fs), 1, stdin) != 1 ||
        fread(&B_bpf, sizeof(B_bpf), 1, stdin) != 1 ||
        fread(&time_offset, sizeof(time_offset), 1, stdin) != 1 ||
        fread(&correct_time_offset, sizeof(correct_time_offset), 1, stdin) != 1) {
        fprintf(stderr, "truncated header\n");
        return 1;
    }

    w = (float *)malloc((size_t)Nc * sizeof(*w));
    pend = (COMP *)malloc((size_t)M * sizeof(*pend));
    if (!w || !pend ||
        fread(w, sizeof(*w), (size_t)Nc, stdin) != (size_t)Nc ||
        read_comp_vec(pend, M) != 0 ||
        fread(&output_dim, sizeof(output_dim), 1, stdin) != 1 ||
        fread(&ncases, sizeof(ncases), 1, stdin) != 1) {
        fprintf(stderr, "truncated model payload\n");
        free(w);
        free(pend);
        return 1;
    }

    rx_in = (COMP *)malloc((size_t)(3 * (M + Ncp)) * sizeof(*rx_in));
    features = (float *)malloc((size_t)output_dim * sizeof(*features));
    expected_features = (float *)malloc((size_t)output_dim * sizeof(*expected_features));
    if (!rx_in || !features || !expected_features) {
        fprintf(stderr, "allocation failure\n");
        free(w);
        free(pend);
        free(rx_in);
        free(features);
        free(expected_features);
        return 1;
    }

    memset(&cfg, 0, sizeof(cfg));
    cfg.M = M;
    cfg.Ncp = Ncp;
    cfg.Ns = Ns;
    cfg.Nc = Nc;
    cfg.Fs = Fs;
    cfg.B_bpf = B_bpf;
    cfg.time_offset = time_offset;
    cfg.correct_time_offset = correct_time_offset;
    cfg.auxdata = auxdata;
    cfg.limit_pitch = limit_pitch;
    cfg.mute = mute;
    cfg.agc = agc;
    cfg.hangover = hangover;
    cfg.timing_adj_at = timing_adj_at;
    cfg.reset_output_on_resync = reset_output_on_resync;
    cfg.w = w;
    cfg.pend = pend;
    if (rx2_receiver_init(&rx, &cfg) != 0) {
        fprintf(stderr, "rx2_receiver_init failed\n");
        free(w);
        free(pend);
        free(rx_in);
        free(features);
        free(expected_features);
        return 1;
    }
    if (output_dim != rx.frame_sync.output_dim) {
        fprintf(stderr, "output_dim mismatch %d/%d\n", (int)output_dim, rx.frame_sync.output_dim);
        rx2_receiver_destroy(&rx);
        free(w);
        free(pend);
        free(rx_in);
        free(features);
        free(expected_features);
        return 1;
    }

    for (int c = 0; c < ncases; c++) {
        int32_t nin;
        int32_t expected_state, expected_count, expected_count1, expected_n_acq;
        int32_t expected_s, expected_i, expected_timing_adj;
        int32_t expected_sig_det, expected_sine_det, expected_decoded_valid, expected_next_nin;
        int32_t expected_new_sig_delta_hat, expected_new_sig_f_hat, expected_delta_hat_g;
        float expected_gain, expected_delta_hat, expected_freq_offset, expected_freq_offset_g;
        float expected_Ry_max, expected_Ry_min, expected_snr_est_dB;
        float expected_frame_sync_even, expected_frame_sync_odd, expected_eoo_smooth, expected_eoo_corr;
        struct rx2_receiver_step step;

        if (fread(&nin, sizeof(nin), 1, stdin) != 1 ||
            nin <= 0 || nin > rx.max_nin ||
            read_comp_vec(rx_in, nin) != 0 ||
            fread(&expected_state, sizeof(expected_state), 1, stdin) != 1 ||
            fread(&expected_count, sizeof(expected_count), 1, stdin) != 1 ||
            fread(&expected_count1, sizeof(expected_count1), 1, stdin) != 1 ||
            fread(&expected_n_acq, sizeof(expected_n_acq), 1, stdin) != 1 ||
            fread(&expected_s, sizeof(expected_s), 1, stdin) != 1 ||
            fread(&expected_i, sizeof(expected_i), 1, stdin) != 1 ||
            fread(&expected_timing_adj, sizeof(expected_timing_adj), 1, stdin) != 1 ||
            fread(&expected_sig_det, sizeof(expected_sig_det), 1, stdin) != 1 ||
            fread(&expected_sine_det, sizeof(expected_sine_det), 1, stdin) != 1 ||
            fread(&expected_decoded_valid, sizeof(expected_decoded_valid), 1, stdin) != 1 ||
            fread(&expected_next_nin, sizeof(expected_next_nin), 1, stdin) != 1 ||
            fread(&expected_new_sig_delta_hat, sizeof(expected_new_sig_delta_hat), 1, stdin) != 1 ||
            fread(&expected_new_sig_f_hat, sizeof(expected_new_sig_f_hat), 1, stdin) != 1 ||
            fread(&expected_gain, sizeof(expected_gain), 1, stdin) != 1 ||
            fread(&expected_delta_hat, sizeof(expected_delta_hat), 1, stdin) != 1 ||
            fread(&expected_delta_hat_g, sizeof(expected_delta_hat_g), 1, stdin) != 1 ||
            fread(&expected_freq_offset, sizeof(expected_freq_offset), 1, stdin) != 1 ||
            fread(&expected_freq_offset_g, sizeof(expected_freq_offset_g), 1, stdin) != 1 ||
            fread(&expected_Ry_max, sizeof(expected_Ry_max), 1, stdin) != 1 ||
            fread(&expected_Ry_min, sizeof(expected_Ry_min), 1, stdin) != 1 ||
            fread(&expected_snr_est_dB, sizeof(expected_snr_est_dB), 1, stdin) != 1 ||
            fread(&expected_frame_sync_even, sizeof(expected_frame_sync_even), 1, stdin) != 1 ||
            fread(&expected_frame_sync_odd, sizeof(expected_frame_sync_odd), 1, stdin) != 1 ||
            fread(&expected_eoo_smooth, sizeof(expected_eoo_smooth), 1, stdin) != 1 ||
            fread(&expected_eoo_corr, sizeof(expected_eoo_corr), 1, stdin) != 1 ||
            fread(expected_features, sizeof(*expected_features), (size_t)output_dim, stdin) != (size_t)output_dim) {
            fprintf(stderr, "truncated payload at c=%d\n", c);
            fails = 1;
            break;
        }

        if (rx2_receiver_process(&rx, rx_in, nin, features, &step) != 0) {
            fprintf(stderr, "rx2_receiver_process failed at c=%d\n", c);
            fails = 1;
            break;
        }

#define CHECK_INT(label, got, exp) \
        do { \
            if ((got) != (exp)) { \
                if (fails < 5) fprintf(stderr, label " FAIL c=%d got=%d expected=%d\n", c, (int)(got), (int)(exp)); \
                fails++; \
            } \
            compared++; \
        } while (0)

#define CHECK_FLOAT(label, got, exp, tol) \
        do { \
            float err__ = fabsf((got) - (exp)); \
            if (err__ > max_err) max_err = err__; \
            if (err__ >= (tol)) { \
                if (fails < 5) fprintf(stderr, label " FAIL c=%d got=%.9e expected=%.9e err=%.3e\n", c, (double)(got), (double)(exp), (double)err__); \
                fails++; \
            } \
            compared++; \
        } while (0)

        CHECK_INT("STATE", rx.state, expected_state);
        CHECK_INT("COUNT", rx.count, expected_count);
        CHECK_INT("COUNT1", rx.count1, expected_count1);
        CHECK_INT("N_ACQ", rx.n_acq, expected_n_acq);
        CHECK_INT("S", rx.s, expected_s);
        CHECK_INT("I", rx.i, expected_i);
        CHECK_INT("TIMING_ADJ", rx.timing_adj, expected_timing_adj);
        CHECK_INT("SIG", step.sig_det, expected_sig_det);
        CHECK_INT("SINE", step.sine_det, expected_sine_det);
        CHECK_INT("VALID", step.decoded_valid, expected_decoded_valid);
        CHECK_INT("NEXT_NIN", step.next_nin, expected_next_nin);
        CHECK_INT("NEW_DH", rx.new_sig_delta_hat, expected_new_sig_delta_hat);
        CHECK_INT("NEW_F", rx.new_sig_f_hat, expected_new_sig_f_hat);
        CHECK_INT("DELTA_HAT_G", rx.coarse_sync.delta_hat_g, expected_delta_hat_g);
        CHECK_FLOAT("GAIN", step.gain, expected_gain, STATE_TOL);
        CHECK_FLOAT("DELTA_HAT", rx.delta_hat, expected_delta_hat, STATE_TOL);
        CHECK_FLOAT("FREQ", rx.freq_offset, expected_freq_offset, STATE_TOL);
        CHECK_FLOAT("FREQ_G", rx.freq_offset_g, expected_freq_offset_g, STATE_TOL);
        CHECK_FLOAT("RY_MAX", rx.coarse_sync.Ry_max, expected_Ry_max, STATE_TOL);
        CHECK_FLOAT("RY_MIN", rx.coarse_sync.Ry_min, expected_Ry_min, STATE_TOL);
        CHECK_FLOAT("SNR", rx.coarse_sync.snr_est_dB, expected_snr_est_dB, STATE_TOL);
        CHECK_FLOAT("FS_EVEN", rx.frame_sync.frame_sync_even, expected_frame_sync_even, STATE_TOL);
        CHECK_FLOAT("FS_ODD", rx.frame_sync.frame_sync_odd, expected_frame_sync_odd, STATE_TOL);
        CHECK_FLOAT("EOO", rx.eoo.eoo_smooth, expected_eoo_smooth, STATE_TOL);
        CHECK_FLOAT("EOO_CORR", rx.eoo.eoo_corr, expected_eoo_corr, STATE_TOL);

        for (int i = 0; i < output_dim; i++) {
            CHECK_FLOAT("FEAT", features[i], expected_features[i], FEATURE_TOL);
        }

#undef CHECK_FLOAT
#undef CHECK_INT
    }

    fprintf(stderr, "cases=%d compared=%d max_err=%.3e state_tol=%.0e feat_tol=%.0e fails=%d\n",
            ncases, compared, max_err, STATE_TOL, FEATURE_TOL, fails);

    rx2_receiver_destroy(&rx);
    free(w);
    free(pend);
    free(rx_in);
    free(features);
    free(expected_features);
    return fails == 0 ? 0 : 1;
}
