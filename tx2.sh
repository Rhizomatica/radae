#!/bin/bash
#
# Wrapper around tx2.py (RADE V2 transmitter)

OPUS=build/src
PATH=${PATH}:${OPUS}

if [ $# -lt 3 ]; then
    echo "usage:"
    echo "  ./tx2.sh model in.wav rx.iqf32 [optional tx2.py args]"
    exit 1
fi

if [ ! -f $1 ]; then
    echo "can't find $1"
    exit 1
fi
if [ ! -f $2 ]; then
    echo "can't find $2"
    exit 1
fi

model=$1
input_speech=$2
output_iqf32=$3
features_in=features_in.f32

# eat first 3 args before passing rest to tx2.py in $@
shift; shift; shift

lpcnet_demo -features ${input_speech} ${features_in}
python3 ./tx2.py ${model} ${features_in} ${output_iqf32} "$@"
