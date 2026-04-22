/*
 * Frame-sync integration state for the RADE V2 receiver.
 *
 * This ports rx2.py _update_frame_sync():
 *   - FrameSyncNet metric update
 *   - odd/even winner tracking
 *   - winning az_hat retention
 *
 * Copyright (c) 2026 Rhizomatica, BSD-3-Clause
 */

#ifndef RX2_FRAME_SYNC_H
#define RX2_FRAME_SYNC_H

#ifdef __cplusplus
extern "C" {
#endif

#include "frame_sync.h"

struct rx2_frame_sync {
    float BETA;
    float metric;
    float frame_sync_even;
    float frame_sync_odd;
    float az_hat[FRAME_SYNC_INPUT_DIM];
};

/*
 * Initialise the frame-sync tracking state.
 *
 * Returns 0 on success, -1 on failure.
 */
int rx2_frame_sync_init(struct rx2_frame_sync *fs);

/* Reset rolling state to the Python constructor state. */
void rx2_frame_sync_reset(struct rx2_frame_sync *fs);

/* Release dynamic storage or other state. Safe on zeroed structs. */
void rx2_frame_sync_destroy(struct rx2_frame_sync *fs);

/*
 * Update odd/even sync metrics and retain az_hat on the winning alignment.
 *
 * sym_index is the receiver symbol counter self.s from rx2.py.
 *
 * Returns 1 when the current symbol wins, 0 otherwise, -1 on error.
 */
int rx2_frame_sync_apply(struct rx2_frame_sync *fs,
                         const float z_hat[FRAME_SYNC_INPUT_DIM], int sym_index);

#ifdef __cplusplus
}
#endif

#endif /* RX2_FRAME_SYNC_H */
