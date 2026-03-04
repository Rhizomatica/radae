#!/bin/bash -x
#
# Process selection of samples in the wav folder as a RADE V2 sanity check
#
# run from radae dir:
#   ./test/v2_test_wav.sh   

OPUS=build/src
PATH=${PATH}:${OPUS}

function v2_process_4_points {
    wav=$1
    filename=$(basename -- "${wav}")
    filename="${filename%.*}"
    ./test/v2_spot.sh --wav ${wav} --wav_out ${wav_out_path}/${filename}_awgn_high.wav
    ./test/v2_spot.sh --wav ${wav} --wav_out ${wav_out_path}/${filename}_awgn_low.wav --EbNodB 1 --hangover 100
    ./test/v2_spot.sh --wav ${wav} --wav_out ${wav_out_path}/${filename}_mpp_high.wav --g_file g_mpp.f32
    ./test/v2_spot.sh --wav ${wav} --wav_out ${wav_out_path}/${filename}_mpp_low.wav --g_file g_mpp.f32 --EbNodB 6
}

wav_out_path="v2_wav_out"
mkdir -p $wav_out_path
v2_process_4_points wav/brian_g8sez.wav ${wav_out_path}

