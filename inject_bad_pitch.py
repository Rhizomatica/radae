#!/usr/bin/env python3
"""Inject an extreme low pitch value into N frames of a features file.

Usage: python3 inject_bad_pitch.py in.f32 out.f32 [pitch_val] [n_frames]

Sets feature[18] (pitch) to pitch_val for n_frames consecutive frames in the
middle of the file, to reliably trigger synthesis pops for testing purposes.
"""
import numpy as np
import sys

nb_features = 36
pitch_idx   = 18
pitch_val   = float(sys.argv[3]) if len(sys.argv) > 3 else -5.0
n_frames    = int(sys.argv[4])   if len(sys.argv) > 4 else 5

feats = np.fromfile(sys.argv[1], dtype=np.float32).reshape(-1, nb_features)
mid = len(feats) // 2
feats[mid : mid + n_frames, pitch_idx] = pitch_val
feats.tofile(sys.argv[2])
