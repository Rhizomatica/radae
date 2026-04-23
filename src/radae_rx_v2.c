#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#endif // _WIN32

#include "rade_api.h"

static void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s [--model_name path] [--model path] "
            "[--frame_sync_model_name path] [--fsync path] [-v|--verbose]\n",
            prog);
}

int main(int argc, char *argv[])
{
    const char *model_name = RADE_RX_V2_COMPILED_MODEL_NAME;
    const char *frame_sync_model_name = RADE_RX_V2_COMPILED_FRAME_SYNC_MODEL_NAME;
    int flags = RADE_VERBOSE_0;

    for (int i = 1; i < argc; i++) {
        if ((!strcmp(argv[i], "--model_name") || !strcmp(argv[i], "--model")) && i + 1 < argc) {
            model_name = argv[++i];
        } else if ((!strcmp(argv[i], "--frame_sync_model_name") || !strcmp(argv[i], "--fsync")) && i + 1 < argc) {
            frame_sync_model_name = argv[++i];
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

    struct rade *r = rade_rx_v2_pure_c_open(
        model_name,
        frame_sync_model_name,
        flags);
    if (r == NULL) {
        rade_finalize();
        return 1;
    }

    int n_features_out = rade_n_features_in_out(r);
    float *features_out = (float *)malloc(sizeof(float) * n_features_out);
    if (features_out == NULL) {
        rade_close(r);
        rade_finalize();
        return 1;
    }

    int n_rx_in = rade_nin_max(r);
    RADE_COMP *rx_in = (RADE_COMP *)malloc(sizeof(RADE_COMP) * n_rx_in);
    if (rx_in == NULL) {
        free(features_out);
        rade_close(r);
        rade_finalize();
        return 1;
    }

    int nin = rade_nin(r);
    int has_eoo_out = 0;
    float eoo_dummy = 0.0f;

#ifdef _WIN32
    _setmode(_fileno(stdin), O_BINARY);
    _setmode(_fileno(stdout), O_BINARY);
#endif // _WIN32

    while((size_t)nin == fread(rx_in, sizeof(RADE_COMP), nin, stdin)) {
        int n_out = rade_rx_v2_pure_c(r, features_out, &has_eoo_out, &eoo_dummy, rx_in);
        if (n_out) {
            fwrite(features_out, sizeof(float), n_features_out, stdout);
            fflush(stdout);
        }
        nin = rade_nin(r);
    }

    free(features_out);
    free(rx_in);
    rade_close(r);
    rade_finalize();
    return 0;
}
