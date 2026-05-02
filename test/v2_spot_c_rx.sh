#!/bin/bash
#
# RADE V2 native-C receive spot test.
#
# Generates a channel-impaired V2 signal with inference.sh, decodes it once
# with the Python rx2.py path and once with the native radae_rx_v2 binary,
# then compares the resulting feature streams.

set -euo pipefail

RADE_BUILD_DIR=${RADE_BUILD_DIR:-build}
OPUS=${RADE_BUILD_DIR}/src
PATH=${PATH}:${OPUS}

delta=${delta:-0.02}
g_file=""
a_g_file=""
EbNodB=""
a_EbNodB_value=""

while [[ $# -gt 0 ]]; do
    key="$1"
    case "${key}" in
        --g_file)
            g_file="--g_file"
            a_g_file="$2"
            shift 2
            ;;
        --EbNodB)
            EbNodB="--EbNodB"
            a_EbNodB_value="$2"
            shift 2
            ;;
        *)
            echo "unknown option: ${key}" >&2
            exit 1
            ;;
    esac
done

if [[ -n "${a_g_file}" ]]; then
    ./test/make_g.sh
fi

./inference.sh 250725/checkpoints/checkpoint_epoch_200.pth wav/all.wav /dev/null \
    --rate_Fs --latent-dim 56 --peak --cp 0.004 --time_offset -16 \
    --correct_time_offset -16 --auxdata --w1_dec 128 --write_rx 250725_rx.f32 \
    --prepend_noise 1 --append_noise 2 --freq_offset 25 --correct_freq_offset \
    ${g_file} ${a_g_file} ${EbNodB} ${a_EbNodB_value}

./rx2.sh 250725/checkpoints/checkpoint_epoch_200.pth 250725a_ml_sync \
    250725_rx.f32 /dev/null --quiet

"${RADE_BUILD_DIR}/src/radae_rx_v2" --quiet < 250725_rx.f32 > features_out_rx2_c.f32

python3 loss.py features_in.f32 features_out_rx2.f32 \
    --features_hat2 features_out_rx2_c.f32 \
    --clip_start 100 --clip_end 300 --compare --delta "${delta}"
