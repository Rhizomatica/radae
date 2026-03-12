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
    ./test/v2_spot.sh --wav ${wav} --wav_out ${wav_out_path}/${filename}_awgn_high.wav --end_of_over_v2
    ./test/v2_spot.sh --wav ${wav} --wav_out ${wav_out_path}/${filename}_awgn_low.wav --end_of_over_v2 --EbNodB 1 --hangover 100
    ./test/v2_spot.sh --wav ${wav} --wav_out ${wav_out_path}/${filename}_mpp_high.wav --end_of_over_v2 --g_file g_mpp.f32 --g_offset 2
    ./test/v2_spot.sh --wav ${wav} --wav_out ${wav_out_path}/${filename}_mpp_low.wav --end_of_over_v2 --g_file g_mpp.f32 --EbNodB 6 --g_offset 2
}

wav_out_path="v2_wav_out"
mkdir -p $wav_out_path
#v2_process_4_points wav/brian_g8sez.wav ${wav_out_path}
#v2_process_4_points wav/k0pfx_mel.wav ${wav_out_path}
#v2_process_4_points wav/mooneer.wav ${wav_out_path}
#v2_process_4_points wav/jh0pcf_kanda.wav ${wav_out_path}
#v2_process_4_points wav/david_vk5dgr.wav ${wav_out_path}
v2_process_4_points wav/w0atn_phyllis.wav ${wav_out_path}

