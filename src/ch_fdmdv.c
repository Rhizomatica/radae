/*---------------------------------------------------------------------------*\

  FILE........: ch_fdmdv.c
  AUTHOR......: David Rowe
  DATE CREATED: May 2015

  fdmdv_freq_shift_coh() extracted from codec2-dev cohpsk.c to remove
  codec2-dev dependency from the ch channel simulation tool.

\*---------------------------------------------------------------------------*/

/*
  Copyright (C) 2015 David Rowe

  All rights reserved.

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU Lesser General Public License version 2.1, as
  published by the Free Software Foundation.  This program is
  distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
  License for more details.

  You should have received a copy of the GNU Lesser General Public License
  along with this program; if not, see <http://www.gnu.org/licenses/>.
*/

#include <math.h>
#include "comp.h"
#include "comp_prim.h"
#include "ch_fdmdv.h"

/*---------------------------------------------------------------------------*\

  FUNCTION....: fdmdv_freq_shift_coh()
  AUTHOR......: David Rowe
  DATE CREATED: May 2015

  Frequency shift modem signal.  The use of complex input and output allows
  single sided frequency shifting (no images).

\*---------------------------------------------------------------------------*/

void fdmdv_freq_shift_coh(COMP rx_fdm_fcorr[], COMP rx_fdm[], float foff, float Fs,
                          COMP *foff_phase_rect, int nin)
{
    COMP  foff_rect;
    float mag;
    float tau = 2.0f * M_PI;
    float result = tau * foff/Fs;
    int   i;

    foff_rect.real = cosf(result);
    foff_rect.imag = sinf(result);
    for(i=0; i<nin; i++) {
        *foff_phase_rect = cmult(*foff_phase_rect, foff_rect);
        rx_fdm_fcorr[i] = cmult(rx_fdm[i], *foff_phase_rect);
    }

    /* normalise digital oscillator as the magnitude can drift over time */

    mag = cabsolute(*foff_phase_rect);
    foff_phase_rect->real /= mag;
    foff_phase_rect->imag /= mag;
}
