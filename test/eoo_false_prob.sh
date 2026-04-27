#!/bin/bash -e
#
# Measure false EOO detection rate for RADE V2.
# Runs N independent trials using the full all.wav WITHOUT --end_of_over_v2,
# counts how many times rx2.py falsely triggers "EOO detected" per trial.
# Each trial has a fresh noise realisation; for MPP, g_offset steps through
# the channel file to sample different fade positions.
#
# Usage:
#   ./test/eoo_false_prob.sh [--EbNodB <dB>] [--channel <awgn|mpp>] [--N <trials>]

EbNodB=100
channel=awgn
N=20
wav=wav/all.wav

function print_help {
    echo
    echo "Measure RADE V2 EOO false detection rate"
    echo
    echo "  usage: ./test/eoo_false_prob.sh [--EbNodB dB] [--channel awgn|mpp] [--N trials] [--wav wavefile]"
    echo "  example: ./test/eoo_false_prob.sh --EbNodB 10 --channel mpp --N 20"
    echo
    exit
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --EbNodB)  EbNodB=$2;  shift 2 ;;
        --channel) channel=$2; shift 2 ;;
        --N)       N=$2;       shift 2 ;;
        --wav)     wav=$2;     shift 2 ;;
        -h|--help) print_help ;;
        *) echo "Unknown argument: $1"; print_help ;;
    esac
done

wav_dur=$(soxi -D $wav)

if [ "$channel" = "mpp" ]; then
    chan_args="--g_file g_mpp.f32"
    g_step=$(python3 -c "print(int($wav_dur))")   # step by one wav length per trial
    g_mpp_dur=$(python3 -c "import os; print(os.path.getsize('g_mpp.f32')//(2*2*4*8000))")
elif [ "$channel" = "awgn" ]; then
    chan_args=""
else
    echo "Unknown channel: $channel (use awgn or mpp)"
    exit 1
fi

total_false=0
rx_tmp=$(mktemp /tmp/eoo_false_XXXXXX.f32)
trap "rm -f $rx_tmp" EXIT

echo "EOO false detection rate: EbNodB=$EbNodB channel=$channel N=$N"

for i in $(seq 1 $N); do
    g_offset_args=""
    g_off=0
    if [ "$channel" = "mpp" ]; then
        g_off=$(python3 -c "print((($i-1)*$g_step) % $g_mpp_dur)")
        g_offset_args="--g_offset $g_off"
    fi

    ./inference.sh 250725/checkpoints/checkpoint_epoch_200.pth $wav /dev/null --rate_Fs \
        --latent-dim 56 --peak --cp 0.004 --time_offset -16 --correct_time_offset -16 --auxdata --w1_dec 128 \
        --write_rx $rx_tmp --EbNodB $EbNodB \
        $chan_args $g_offset_args 2>/dev/null

    count=$(./rx2.sh 250725/checkpoints/checkpoint_epoch_200.pth 250725a_ml_sync $rx_tmp /dev/null \
        --latent-dim 56 --w1_dec 128 --correct_time_offset -8 2>&1 | grep -c "EOO detected" || true)

    total_false=$((total_false + count))
    echo "  trial $i: $count false triggers (g_offset=$g_off)"
done

if [ "$total_false" -gt 0 ]; then
    avg_time=$(python3 -c "print(f'{$N*$wav_dur/$total_false:.1f}')")
    echo "Result: $total_false false triggers over $N trials, avg time between false triggers = ${avg_time} s"
else
    echo "Result: 0 false triggers over $N trials"
fi
