#!/bin/bash -e
#
# Test RADE V2 SNR estimator accuracy.
# Steps through a range of EbNodB values, printing EbNodB, measured SNR3k,
# and estimated SNR3k (mean over sync periods) to stdout.
#
# Usage:
#   ./test/snr_est_test.sh [--channel awgn|mpp|mpg] [--EbNodB_min dB] [--EbNodB_max dB] [--EbNodB_step dB]
#
# Example:
#   ./test/snr_est_test.sh --channel mpg --EbNodB_min 0 --EbNodB_max 20 | tee mpg_snr_est.txt

channel=awgn
EbNodB_min=0
EbNodB_max=20
EbNodB_step=1

function print_help {
    echo
    echo "Test RADE V2 SNR estimator accuracy"
    echo
    echo "  usage: ./test/snr_est_test.sh [--channel awgn|mpp|mpg] [--EbNodB_min dB] [--EbNodB_max dB] [--EbNodB_step dB]"
    echo "  example: ./test/snr_est_test.sh --channel mpg --EbNodB_min 0 --EbNodB_max 20"
    echo
    exit
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --channel)     channel=$2;     shift 2 ;;
        --EbNodB_min)  EbNodB_min=$2;  shift 2 ;;
        --EbNodB_max)  EbNodB_max=$2;  shift 2 ;;
        --EbNodB_step) EbNodB_step=$2; shift 2 ;;
        -h|--help) print_help ;;
        *) echo "Unknown argument: $1"; print_help ;;
    esac
done

if [ "$channel" = "mpp" ]; then
    chan_args="--g_file g_mpp.f32"
elif [ "$channel" = "mpg" ]; then
    chan_args="--g_file g_mpg.f32"
elif [ "$channel" = "awgn" ]; then
    chan_args=""
else
    echo "Unknown channel: $channel (use awgn, mpp, or mpg)"
    exit 1
fi

rx_tmp=$(mktemp /tmp/snr_est_rx_XXXXXX.f32)
snr_tmp=$(mktemp /tmp/snr_est_snr_XXXXXX.f32)
state_tmp=$(mktemp /tmp/snr_est_state_XXXXXX.int16)
trap "rm -f $rx_tmp $snr_tmp $state_tmp" EXIT

echo "# EbNodB  SNR3k_meas  SNR3k_est"

for EbNodB in $(python3 -c "
import numpy as np
steps = round(($EbNodB_max - $EbNodB_min) / $EbNodB_step) + 1
print(' '.join([str(round($EbNodB_min + i*$EbNodB_step, 6)) for i in range(steps)]))")
do
    snr3k_meas=$(./inference.sh 250725/checkpoints/checkpoint_epoch_200.pth wav/all.wav /dev/null --rate_Fs \
        --latent-dim 56 --peak --cp 0.004 --time_offset -16 --correct_time_offset -16 --auxdata --w1_dec 128 \
        --write_rx $rx_tmp --EbNodB $EbNodB \
        $chan_args 2>&1 | awk '/^Measured:/ {print $4}')

    ./rx2.sh 250725/checkpoints/checkpoint_epoch_200.pth 250725a_ml_sync $rx_tmp /dev/null \
        --latent-dim 56 --w1_dec 128 --correct_time_offset -8 \
        --write_snr_est $snr_tmp --write_state $state_tmp 2>/dev/null

    snr3k_est=$(python3 -c "
import numpy as np
snr   = np.fromfile('$snr_tmp',   dtype=np.float32)
state = np.fromfile('$state_tmp', dtype=np.int16)
sync  = state > 0
if sync.sum() > 0:
    print(f'{np.mean(snr[sync]):.2f}')
else:
    print('nan')
")

    printf "%8.2f  %10.2f  %10s\n" "$EbNodB" "$snr3k_meas" "$snr3k_est"
done
