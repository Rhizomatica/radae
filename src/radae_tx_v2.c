/*
 * RADE V2 streaming transmitter, native C drop-in for radae_txe2.py.
 *
 *   stdin  : float32 FARGAN features (Nzmf*enc_stride*36 floats per frame)
 *   stdout : float32 complex IQ (Nmf samples per frame)
 *
 * Same CLI surface as radae_txe2.py:
 *   --model_name PATH     accepted for compat; must match the compiled-in
 *                         identifier (RADE_TX_V2_COMPILED_MODEL_NAME)
 *   --pid_file PATH       write own PID at startup, unlink on exit
 *   --no_eoo              skip the shutdown EOO frame
 *   --txbpf               (currently disabled; complex_bpf needs the c42466d
 *                         streaming-state fix mirrored to C first)
 *   -v / --verbose        chatty logging
 *
 * SIGUSR1 -> emit one EOO frame at the next safe boundary.  Hermes
 * (sbitx_radae.c:radae_tx_emit_eoo) signals the PID stored in
 * /tmp/radae_tx.pid, which this binary writes on startup when invoked
 * with --pid_file.
 *
 * Copyright (c) 2026 Rhizomatica, BSD-3-Clause
 */

#include <assert.h>
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#endif

#include "rade_api.h"

static volatile sig_atomic_t eoo_request = 0;

static void usage(const char *prog) {
    fprintf(stderr,
            "Usage: %s [--model_name PATH] [--pid_file PATH] "
            "[--no_eoo] [--txbpf] [-v|--verbose]\n",
            prog);
}

static void handle_sigusr1(int signum) {
    (void)signum;
    eoo_request = 1;
}

static void write_pid_file(const char *path) {
    FILE *f = fopen(path, "w");
    if (f == NULL) {
        fprintf(stderr, "radae_tx_v2: failed to open pid_file %s: %s\n",
                path, strerror(errno));
        return;
    }
    fprintf(f, "%d\n", (int)getpid());
    fclose(f);
}

int main(int argc, char *argv[]) {
    const char *model_name = RADE_TX_V2_COMPILED_MODEL_NAME;
    const char *pid_file = NULL;
    int flags = RADE_VERBOSE_0;
    int send_eoo = 1;
    int txbpf = 0;

    for (int i = 1; i < argc; i++) {
        if ((!strcmp(argv[i], "--model_name") || !strcmp(argv[i], "--model")) &&
            i + 1 < argc) {
            model_name = argv[++i];
        } else if (!strcmp(argv[i], "--pid_file") && i + 1 < argc) {
            pid_file = argv[++i];
        } else if (!strcmp(argv[i], "--no_eoo")) {
            send_eoo = 0;
        } else if (!strcmp(argv[i], "--txbpf")) {
            /* The C complex_bpf streaming-state convention diverged from
             * the Python fix in commit c42466d.  Refuse rather than ship
             * a silent BPF mismatch on the air. */
            fprintf(stderr,
                    "radae_tx_v2: --txbpf is not yet supported in the C path; "
                    "see C_TX_MIGRATION.md.\n");
            (void)txbpf;
            return 1;
        } else if (!strcmp(argv[i], "-v") || !strcmp(argv[i], "--verbose")) {
            flags &= ~RADE_VERBOSE_0;
        } else if (!strcmp(argv[i], "--quiet")) {
            flags |= RADE_VERBOSE_0;
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    rade_initialize();

    struct rade *r = rade_tx_v2_pure_c_open(model_name, flags);
    if (r == NULL) {
        rade_finalize();
        return 1;
    }

    int n_features_in = rade_n_features_in_out(r);
    int Nmf  = rade_n_tx_out(r);
    int Neoo = rade_n_tx_eoo_out(r);

    float *features_in = (float *)malloc(sizeof(float) * (size_t)n_features_in);
    RADE_COMP *tx_out = (RADE_COMP *)malloc(sizeof(RADE_COMP) * (size_t)Nmf);
    RADE_COMP *eoo_out = (RADE_COMP *)malloc(sizeof(RADE_COMP) * (size_t)Neoo);
    if (!features_in || !tx_out || !eoo_out) {
        free(features_in);
        free(tx_out);
        free(eoo_out);
        rade_close(r);
        rade_finalize();
        return 1;
    }

#ifdef _WIN32
    _setmode(_fileno(stdin),  O_BINARY);
    _setmode(_fileno(stdout), O_BINARY);
#endif

    /* Install SIGUSR1 handler with restartable=false so a pending stdin
     * read returns EINTR and we get a chance to service the EOO request
     * promptly between frames. */
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = handle_sigusr1;
    sa.sa_flags = 0;   /* default: SA_RESTART is OFF on Linux */
    sigemptyset(&sa.sa_mask);
    sigaction(SIGUSR1, &sa, NULL);

    if (pid_file) {
        write_pid_file(pid_file);
    }

    int rc = 0;
    while (1) {
        size_t want = (size_t)n_features_in;
        size_t got = 0;
        while (got < want) {
            size_t n = fread(features_in + got, sizeof(float), want - got, stdin);
            if (n == 0) {
                if (feof(stdin)) goto eof;
                if (ferror(stdin) && errno == EINTR) {
                    clearerr(stdin);
                    if (eoo_request) {
                        /* SIGUSR1 mid-frame: drop the partial frame, emit EOO,
                         * then resume reading.  Python radae_txe2.py does the
                         * same via raise/finally semantics. */
                        eoo_request = 0;
                        sigset_t blk, old;
                        sigemptyset(&blk);
                        sigaddset(&blk, SIGUSR1);
                        sigprocmask(SIG_BLOCK, &blk, &old);
                        if (rade_tx_eoo(r, eoo_out) != Neoo ||
                            fwrite(eoo_out, sizeof(*eoo_out), (size_t)Neoo, stdout) != (size_t)Neoo) {
                            sigprocmask(SIG_SETMASK, &old, NULL);
                            rc = 1; goto eof;
                        }
                        fflush(stdout);
                        sigprocmask(SIG_SETMASK, &old, NULL);
                        got = 0;   /* discard the partial frame */
                        continue;
                    }
                    continue;
                }
                rc = 1; goto eof;
            }
            got += n;
        }

        if (rade_tx(r, tx_out, features_in) != Nmf ||
            fwrite(tx_out, sizeof(*tx_out), (size_t)Nmf, stdout) != (size_t)Nmf) {
            rc = 1; goto eof;
        }
        fflush(stdout);

        if (eoo_request) {
            eoo_request = 0;
            sigset_t blk, old;
            sigemptyset(&blk);
            sigaddset(&blk, SIGUSR1);
            sigprocmask(SIG_BLOCK, &blk, &old);
            if (rade_tx_eoo(r, eoo_out) != Neoo ||
                fwrite(eoo_out, sizeof(*eoo_out), (size_t)Neoo, stdout) != (size_t)Neoo) {
                sigprocmask(SIG_SETMASK, &old, NULL);
                rc = 1; goto eof;
            }
            fflush(stdout);
            sigprocmask(SIG_SETMASK, &old, NULL);
        }
    }

eof:
    if (send_eoo && rc == 0) {
        if (rade_tx_eoo(r, eoo_out) == Neoo) {
            fwrite(eoo_out, sizeof(*eoo_out), (size_t)Neoo, stdout);
            fflush(stdout);
        }
    }
    if (pid_file) {
        unlink(pid_file);
    }

    free(features_in);
    free(tx_out);
    free(eoo_out);
    rade_close(r);
    rade_finalize();
    return rc;
}
