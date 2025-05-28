#!/usr/bin/env python3
# Copyright 2025 FBK

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License
import os
import numpy as np
import wave


def read_sphere_file(sph_file):
    with open(sph_file, 'rb') as f:
        header = f.read(1024)

        # Check if it's a valid SPHERE file and parse header
        if not header.startswith(b'NIST'):
            raise ValueError("Not a valid SPHERE file")

        # Parse the file (this will vary based on the actual SPHERE format)
        data = np.fromfile(f, dtype=np.int16)

        return data

def write_wav_file(data, wav_file):
    with wave.open(wav_file, 'wb') as wav:
        wav.setnchannels(1)  # Mono
        wav.setsampwidth(2)  # 16 bits
        wav.setframerate(16000)
        wav.writeframes(data.tobytes())

def convert_sph_to_wav(sph_file, wav_file):
    try:
        data = read_sphere_file(sph_file)
        write_wav_file(data, wav_file)
        print(f"Converted {sph_file} to {wav_file}")
    except Exception as e:
        print(f"Error converting {sph_file}: {e}")

ROOT_DIR = ''  # Set TIMIT root directory

# Iterate through all .WAV files (which are actually SPH)
for root, dirs, files in os.walk(ROOT_DIR):
    for file in files:
        if file.endswith('.WAV'):
            sph_path = os.path.join(root, file)
            wav_path = sph_path.replace('.WAV', '.wav')
            convert_sph_to_wav(sph_path, wav_path)
