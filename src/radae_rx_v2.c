#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#endif // _WIN32

#include "rade_api.h"

int main(void)
{
    rade_initialize();

    struct rade *r = rade_rx_v2_pure_c_open(
        "250725/checkpoints/checkpoint_epoch_200.pth",
        "250725a_ml_sync",
        RADE_VERBOSE_0);
    assert(r != NULL);

    int n_features_out = rade_n_features_in_out(r);
    float *features_out = (float *)malloc(sizeof(float) * n_features_out);
    assert(features_out != NULL);

    int n_rx_in = rade_nin_max(r);
    RADE_COMP *rx_in = (RADE_COMP *)malloc(sizeof(RADE_COMP) * n_rx_in);
    assert(rx_in != NULL);

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
