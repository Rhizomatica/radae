/*
 * RADE V2 receiver frame-sync integration ported from rx2.py.
 *
 * Copyright (c) 2026 Rhizomatica, BSD-3-Clause
 */

#include <string.h>

#include "rx2_frame_sync.h"

int rx2_frame_sync_init(struct rx2_frame_sync *fs) {
    if (!fs) {
        return -1;
    }

    memset(fs, 0, sizeof(*fs));
    fs->BETA = 0.999f;
    return 0;
}

void rx2_frame_sync_reset(struct rx2_frame_sync *fs) {
    if (!fs) return;
    fs->metric = 0.0f;
    fs->frame_sync_even = 0.0f;
    fs->frame_sync_odd = 0.0f;
    memset(fs->az_hat, 0, sizeof(fs->az_hat));
}

void rx2_frame_sync_destroy(struct rx2_frame_sync *fs) {
    if (!fs) return;
    memset(fs, 0, sizeof(*fs));
}

int rx2_frame_sync_apply(struct rx2_frame_sync *fs,
                         const float z_hat[FRAME_SYNC_INPUT_DIM], int sym_index) {
    int winning = 0;

    if (!fs || !z_hat) {
        return -1;
    }

    fs->metric = frame_sync_forward(z_hat);
    if (sym_index & 1) {
        fs->frame_sync_odd = fs->BETA * fs->frame_sync_odd + (1.0f - fs->BETA) * fs->metric;
        winning = fs->frame_sync_odd > fs->frame_sync_even;
    } else {
        fs->frame_sync_even = fs->BETA * fs->frame_sync_even + (1.0f - fs->BETA) * fs->metric;
        winning = fs->frame_sync_even > fs->frame_sync_odd;
    }

    if (!winning) {
        return 0;
    }

    memcpy(fs->az_hat, z_hat, sizeof(fs->az_hat));
    return 1;
}
