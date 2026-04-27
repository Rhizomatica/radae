/*---------------------------------------------------------------------------*\

  FILE........: ch_fdmdv.h
  AUTHOR......: David Rowe
  DATE CREATED: May 2015

  Minimal header for ch channel simulation tool - extracted from codec2-dev
  cohpsk to remove codec2-dev dependency.

\*---------------------------------------------------------------------------*/

#ifndef __CH_FDMDV__
#define __CH_FDMDV__

#include "comp.h"

#define COHPSK_NOM_SAMPLES_PER_FRAME 600

void fdmdv_freq_shift_coh(COMP rx_fdm_fcorr[], COMP rx_fdm[], float foff, float Fs,
                          COMP *foff_phase_rect, int nin);

#endif
