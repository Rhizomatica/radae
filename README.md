# Radio Autoencoder V2

RADE (Radio AutoEncoder) is a neural codec for transmitting speech over HF radio channels.  A neural encoder compresses speech into a latent vector which is modulated onto an OFDM waveform and transmitted.  At the receiver a neural decoder reconstructs the speech features, which are synthesised into audio by the [FARGAN](https://arxiv.org/abs/2405.21069) vocoder.  The system is trained end-to-end, jointly optimising the encoder, channel layer, and decoder for minimum speech distortion across a range of channel conditions.

RADE V2 builds on V1 with several algorithmic improvements:

| | V1 | V2 |
| --- | --- | --- |
| Carriers | 20, includes pilot symbols | 14, data only (no pilots) |
| Equalisation | Classical DSP, pilot-aided | ML-based, no pilots required |
| 99% Occupied Bandwidth | ~2100 Hz (SSB filter limited) | ~860 Hz |
| Frame duration | ~180 ms | ~40 ms |
| PAPR | 4.2 dB | 3.5 dB |
| Frame sync | DSP | Neural network |
| End-of-over detection | Pilot pend sequence | Channel sparsity metric |
| Threshold SNR (AWGN) | -2 dB | ~-4.5 dB |
| Threshold SNR (MPP) | 0 dB | ~-3 dB |

The elimination of pilot symbols in V2 recovers the bandwidth and power they consumed, enabling a narrower, cleaner waveform and improved high and low SNR performance.  Combined with the PAPR improvement, RADE V2 is approximately 3 dB more sensitive than V1 at low SNRs.

*Threshold SNR values are approximate, based on informal listening tests and objective loss metric.*

# Scope

This repo is intended to support the authors experimental work, with just enough information for the advanced experimenter to reproduce aspects of the work. The focus is on waveform development, not software configuration. It is not intended to be packaged for general use or to work across multiple Linux distros and operating systems - that will come later. Unless otherwise stated, the code is this repo is intended to run only on Ubuntu Linux 22-24 on a non-virtual machine.

This branch is focussed on RADE V2, however it still contains RADE V1 and associated ctests. 

# Quickstart

1. Installation section below.
1. Example Tx and Rx: 
   ```
   ./inference.sh 250725/checkpoints/checkpoint_epoch_200.pth wav/brian_g8sez.wav /dev/null --rate_Fs --latent-dim 56 \
    --peak --cp 0.004 --time_offset -16 --correct_time_offset -8 --auxdata --w1_dec 128 --write_rx 250725_rx.f32 
   ./rx2.sh 250725/checkpoints/checkpoint_epoch_200.pth 250725a_ml_sync 250725_rx.f32 test.wav --latent-dim 56 \
   --w1_dec 128 --correct_time_offset -8
   play test.wav
   ```
1. `test/v2_spot.sh` is a good starting point for experimentation.
   
# Reference and License

D. Rowe, J.-M. Valin, [RADE: A Neural Codec for Transmitting Speech over HF Radio Channels](https://arxiv.org/abs/2505.06671), arXiv:2505.06671, 2025.  This paper describes RADE V1; a V2 paper is planned as future work.

The RADE source code is released under the two-clause BSD license.

# Files

| File | Description |
| --- | --- |
| `inference.py` / `inference.sh` | RADE V2 transmitter: encodes speech and modulates to a complex IQ sample file |
| `rx2.py` / `rx2.sh` | RADE V2 receiver: stateful, streaming decoder |
| `radae/radae.py` | Core RADE model definition (encoder, channel layer, decoder) |
| `train.py` | Training script for the RADE encoder/decoder |
| `ml_sync.py` / `models_sync.py` | ML frame sync: trains and runs the neural frame synchroniser |
| `train_ft_sync.sh` | Automation script for training the ML sync model |
| `loss.py` | Measures ML loss (speech distortion) between encoder and decoder feature vectors |
| `compare_models_inf.sh` | Generates loss versus SNR curves across models and channel types |
| `ota_test.sh` | Over-the-air/over-the-cable test: generates tx signal, decodes rx, measures loss |
| `est_CNo.py` | C/No estimation from a received chirp signal |
| `chirp.py` | Generates a chirp reference signal used for timing and level calibration in OTA tests |
| `int16tof32.py` / `f32toint16.py` | Sample format converters between int16 and float32 |
| `test/v2_spot.sh` | RADE V2 spot test: encodes, applies channel impairments, decodes, checks loss |
| `test/v2_acq.sh` | Acquisition tests: false acquisition rate on noise or noise plus sine wave |
| `test/ota_test_cal.sh` | Calibrated OTA test using the `ch` channel simulator, checks V1 and V2 loss |
| `test/snr_est_test.sh` | Steps through SNR range comparing measured vs estimated SNR3k |
| `test/eoo_detect_prob.sh` | Measures probability of correct EOO detection over a range of channel conditions |
| `test/eoo_false_prob.sh` | Measures EOO false detection rate on noise |

# Installation

## Packages

sox, python3, python3-matplotlib and python3-tqdm, octave, octave-signal, cmake.  Pytorch should be installed using the instructions from the [pytorch](https://pytorch.org/get-started/locally/) web site. 

## codec2-dev

Supplies some utilities used for `ota_test.sh` and `evaluate.sh`.  Removing this dependency is a planned future task.
```
cd ~
git clone https://github.com/drowe67/codec2-dev.git
cd codec2-dev
mkdir build_linux
cd build_linux
cmake -DUNITTEST=1 ..
make ch mksine tlininterp
```

## RADE

Builds the FARGAN vocoder and ctest framework, most of RADAE is in Python.
```
cd ~
git clone https://github.com/drowe67/radae.git
cd radae
mkdir build
cd build
cmake ..
make
```


# Automated Tests

The `cmake/ctest` framework is being used as a build and test framework. The command lines in `CmakeLists.txt` are a good source of examples, if you are interested in running the code in this repo. The ctests are a work in progress and may not pass on all systems (see Scope above).

To run the tests:
```
cd radae/build
ctest
```
To list tests `ctest -N`, to run just one test `ctest -R inference_model5`, to run in verbose mode `ctest -V -R inference_model5`.  You can change the paths to `codec2-dev` on the `cmake` command line:
```
cmake -DCODEC2_DEV=~/tmp/codec2-dev ..
```
A lot of the tests generate a float IQ sample file.  You can listen to this file with: 
```
cat rx.f32 | python3 f32toint16.py --real --scale 8192 | play -t .s16 -r 8000 -c 1 - bandpass 300 2000
```
The scaling `--scale` is required as the low SNRs mean the noise peak amplitude can clip 16 bit samples if not carefully scaled.


# Over the Air/Over the Cable (OTA/OTC)

The `ota_test.sh` script supports stored-file over-the-air and over-the-cable testing.  It assembles a transmit file containing a chirp reference, compressed SSB, RADE V1, and RADE V2 signals in sequence, which can be sent over a real SSB radio or processed through a channel simulator.

Generate a transmit file from an input speech wav (16 kHz mono):
```
./ota_test.sh wav/brian_g8sez.wav -x
```
This produces `tx.wav`.  Pass it through the `ch` channel simulator (from `codec2-dev`) to add noise and fading:
```
~/codec2-dev/build_linux/src/ch tx.wav - --No -20 | sox -t .s16 -r 8000 -c 1 - rx.wav
```
Decode `rx.wav` and measure ML loss against the original speech:
```
./ota_test.sh -r rx.wav -l wav/brian_g8sez.wav
```
The decoded audio files `rx_ssb.wav`, `rx_rade1.wav`, and `rx_rade2.wav` are written to the same directory as `rx.wav`.

The `test/ota_test_cal.sh` script wraps `ota_test.sh` with a calibrated channel simulation suitable for use as a ctest:
```
./test/ota_test_cal.sh ~/codec2-dev/build_linux wav/brian_g8sez.wav -24 0.4
./test/ota_test_cal.sh ~/codec2-dev/build_linux wav/brian_g8sez.wav -30 0.45 --mpp --freq -25
```
Arguments are: path to codec2-dev build, input speech file, noise level (dBW/Hz), loss threshold, and optional channel arguments passed to `ch`.


# Training

This section is optional - pre-trained models that run on a standard laptop CPU are available for experimenting with RADAE. If you wish to perform training, a serious NVIDIA GPU is required - the author used a RTX4090.

1. Generate a training features file using your speech training database `training_input.pcm`, we used 200 hours of speech from open source databases:
   ```
   ./lpcnet_demo -features training_input.pcm training_features_file.f32
   ```
   
1. Generate the MPP channel simulation file:
   ```
   echo "Rs=50; Nc=14; multipath_samples('mpp', Rs, Rs, Nc, 250*60*60, 'h_nc14_mpp_train_test.c64','',1); quit" | octave-cli -qf
   ```

1. Train the RADE V2 encoder/decoder (the `250725` model was trained with these settings):
   ```
   python3 train.py --cuda-visible-devices 0 --sequence-length 400 --batch-size 512 \
     --epochs 200 --lr 0.003 --lr-decay-factor 0.0001 \
      training_features_file.f32 250725 \
     --latent-dim 56 --cp 0.004 --auxdata --w1_dec 128 --peak \
     --h_file h_nc14_mpp_train.c64 --h_complex --range_EbNo --range_EbNo_start 3 \
     --timing_rand --timing_jitter 0.002 --freq_rand --ssb_bpf
   ```
1. Generate latent vectors from the trained model for ML sync training.  This runs one pass through the training data without updating weights:
   ```
   python3 train.py --cuda-visible-devices 0 --sequence-length 400 --batch-size 512 \
     --epochs 200 --lr 0.003 --lr-decay-factor 0.0001 \
     training_features_file.f32 tmp \
     --latent-dim 56 --cp 0.004 --auxdata --w1_dec 128 --peak \
     --h_file h_nc14_mpp_train.c64 --h_complex --range_EbNo --range_EbNo_start 3 \
     --timing_rand --timing_jitter 0.002 --freq_rand --ssb_bpf \
     --plot_EqNo 250725 --initial-checkpoint 250725/checkpoints/checkpoint_epoch_200.pth \
     --write_latent 250725a_z_train.f32
   ```
1. Train the ML frame sync model:
   ```
   python3 ml_sync.py 250725a_z_train.f32 --count 100000 --save_model 250725a_ml_sync --latent_dim 56
   ```


# C Port of Core Encoder and Decoder

The following describes the V1 C port.  A V2 C port is planned as future work.

The model weights can be compiled in or loaded at init-time from a binary blob.  The actual model is hard coded in `rade_enc.c` and `rade_dec.c`, and can't be easily changed.

To compile-in the weights:
1. Export weights:
   ```
   cd radae
   python3 export_rade_weights.py model19_check3/checkpoints/checkpoint_epoch_100.pth src
   ```
1. We need to make some manual changes to the weight files to support changing input dimension at run time.  In `rade_enc_dat.c`, the first call to `linear_init()` should look like:
   ```
   int init_radeenc(RADEEnc *model, const WeightArray *arrays, int input_dim) {
     if (linear_init(&model->enc_dense1, arrays, "enc_dense1_bias", NULL, NULL,"enc_dense1_weights_float", NULL, NULL, NULL, input_dim, 64)) return 1;
   ```
   e.g. the fixed input dimension (84 for `model19_check3`, 80 for earlier models without auxdata) should be changed to the `input_dim` variable. This allows us to enable/disable `auxdata` at init time, without changing the C code for the model.
1. Also make manual changes to support `output_dim` in `rade_dec_dat.c`, `init_radedec()`.
3. Build C code.
4. Run ctests.

To export the compiled in weights to a binary blob:
```
cd radae/build
./src/write_rade_weights ../bin/model05.bin
```
These can then be loaded at init-time, see examples in `src/test_rand_enc.c` and `src/test_rand_dec.c`.

