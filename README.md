# Radio Autoencoder V2

RADE (Radio AutoEncoder) is a neural codec for transmitting speech over HF radio channels.  A neural encoder compresses speech into a latent vector which is modulated onto an OFDM waveform and transmitted.  At the receiver a neural decoder reconstructs the speech features, which are synthesised into audio by the [FARGAN](https://arxiv.org/abs/2405.21069) vocoder.  The system is trained end-to-end, jointly optimising the encoder, channel layer, and decoder for minimum speech distortion across a range of channel conditions.

RADE V2 builds on V1 with several algorithmic improvements:

| | V1 | V2 |
| --- | --- | --- |
| Carriers | 30, includes pilot symbols | 14, data only (no pilots) |
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

This repo is the reference Python implementation for RADE V1 and V2. The current focus is on RADE V2 development, however this repo also contains RADE V1 (including many ctests).

This repo is intended to support experimental work, with just enough information for the advanced experimenter to reproduce aspects of the work. The focus is on waveform development, not software configuration. It is not intended to be packaged for general use or to work across multiple Linux distros and operating systems. Unless otherwise stated, the code in this repo is intended to run only on Ubuntu Linux 22-24 on a non-virtual machine.

For deployment and distribution of RADE V1 please use the [C port](https://github.com/peterbmarks/radae_nopy).  RADE V2 is still under development but we hope to make an initial release soon.

# Quickstart

1. Installation section below.
1. RADE V2 Tx and Rx example:
   ```
   ./inference.sh 250725/checkpoints/checkpoint_epoch_200.pth wav/brian_g8sez.wav /dev/null --rate_Fs --latent-dim 56 \
    --peak --cp 0.004 --time_offset -16 --correct_time_offset -8 --auxdata --w1_dec 128 --write_rx 250725_rx.f32
   ./rx2.sh 250725/checkpoints/checkpoint_epoch_200.pth 250725a_ml_sync 250725_rx.f32 test.wav
   play test.wav
   ```
1. RADE V1 Tx and Rx example:
   ```
   ./inference.sh model19_check3/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav /dev/null \
    --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --auxdata --write_rx v1_rx.f32
   cat v1_rx.f32 | python3 radae_rxe.py --model model19_check3/checkpoints/checkpoint_epoch_100.pth > features_out.f32
   ./build/src/lpcnet_demo -fargan-synthesis features_out.f32 - | aplay -f S16_LE -r 16000
   ```
1. `test/v2_spot.sh` is a good starting point for RADE V2 experimentation.
   
# Reference and License

D. Rowe, J.-M. Valin, [RADE: A Neural Codec for Transmitting Speech over HF Radio Channels](https://arxiv.org/abs/2505.06671), arXiv:2505.06671, 2025.  This paper describes RADE V1; a V2 paper is planned as future work.  The companion branch of this repo (with a RADE V1 focus) is waspaa_2025.

The RADE source code is released under the two-clause BSD license.

# Files

| File | Description |
| --- | --- |
| `inference.py` / `inference.sh` | RADE V2 transmitter: encodes speech and modulates to a complex IQ sample file |
| `rx2.py` / `rx2.sh` | RADE V2 receiver: stateful, streaming decoder |
| `radae_txe.py` / `radae_rxe.py` | RADE V1 transmitter and receiver |
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
To list tests `ctest -N`, to run just one test `ctest -R inference_model5`, to run in verbose mode `ctest -V -R inference_model5`.

## Listening to modulated RADE

A lot of the tests generate a float IQ sample file.  You can listen to this file with: 
```
cat rx.f32 | python3 f32toint16.py --real --scale 8192 | play -t .s16 -r 8000 -c 1 - bandpass 300 2000
```
The scaling `--scale` is required as the low SNRs mean the noise peak amplitude can clip 16 bit samples if not carefully scaled.

## Optional: RADE V1 C Port Tests (radae_nopy)

The [radae_nopy](https://github.com/peterbmarks/radae_nopy) repo contains a C port of the RADE V1 receiver. Its ctests are optional and only enabled when `RADAE_NOPY_BUILD_DIR` is passed to cmake:
```
cd ~
git clone https://github.com/peterbmarks/radae_nopy.git
cd radae_nopy && mkdir build && cd build && cmake .. && make
cd ~/radae/build
cmake -DRADAE_NOPY_BUILD_DIR=~/radae_nopy/build ..
ctest -R radae_nopy
```


# Over the Air/Over the Cable (OTA/OTC)

The `ota_test.sh` script supports stored-file over-the-air and over-the-cable testing.  It assembles a transmit file containing a chirp reference, compressed SSB, RADE V1, and RADE V2 signals in sequence, which can be sent over a real HF channel or processed through a channel simulator.  The script performs a *controlled* test of RADE V2 over real world channels.

Generate a transmit file from an input speech wav (16 kHz mono):
```
./ota_test.sh wav/brian_g8sez.wav -x
```
This produces `tx.wav`, which is suitable for transmission OTA using your SSB transmitter.  We then use a remote HF receiver to sample the received signal to a wave file, e.g. `rx.wav`.

To simulate a real HF channel pass it through the `ch` channel simulator to add noise and fading:
```
./build/src/ch tx.wav - --No -20 | sox -t .s16 -r 8000 -c 1 - rx.wav
```
Decode `rx.wav` and measure ML loss against the original speech:
```
./ota_test.sh -r rx.wav -l wav/brian_g8sez.wav
```
The decoded audio files `rx_ssb.wav`, `rx_rade1.wav`, and `rx_rade2.wav` are written to the same directory as `rx.wav`. A report file and spectrogram is also produced, including objective loss measurements (if `-l` option used).

See `ota_test.sh` for more information.

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
     --timing_rand --freq_rand --ssb_bpf --plot_loss
   ```
1. Generate latent vectors from the trained model for ML sync training.  This runs one pass through the training data without updating weights. Note the addition of +/- 2 ms of timing jitter, to maintain frame sync across the delay spread of multipath channels:
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


# ASR Tests

Automatic Speech Recognition (ASR) is used as an objective speech quality metric to compare RADE V1 against SSB and FreeDV 700D. The [Whisper](https://github.com/openai/whisper) ASR model scores Word Error Rate (WER) on LibriSpeech samples passed through the modems under test.

1. Install dependencies:
   ```
   pip3 install jiwer openai-whisper
   ```

1. Run controls (clean speech, FARGAN vocoder only, 4 kHz bandwidth):
   ```
   ./asr_test.sh clean && ./asr_test.sh fargan && ./asr_test.sh 4kHz
   ```

1. Run a sweep across AWGN channel conditions for each mode (100 samples):
   ```
   ./asr_test_top.sh ssb -n 100
   ./asr_test_top.sh rade -n 100
   ./asr_test_top.sh 700D -n 100
   ```

1. For MPP channel, first generate fading samples (if not already present), then re-run with `--g_file`:
   ```
   ./test/make_g.sh
   ./asr_test_top.sh rade -n 100 --g_file g_mpp.f32
   ```

1. Plot WER curves in Octave:
   ```
   octave:1> radae_plots; plot_wer("241221","241221_asr_test.png")
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

