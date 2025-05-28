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
import csv

import wave


def get_wav_info(wav_path):
    try:
        with wave.open(wav_path, 'r') as wav_file:
            n_frames = wav_file.getnframes()
        return n_frames
    except wave.Error:
        print(f"Warning: {wav_path} is not a valid WAV file.")
        return None


def process_txt_file(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    # The text content starts after the first two numbers
    text = ' '.join(lines[0].strip().split()[2:])
    return text

def generate_tsv(root_dir, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = [
            'id',
            'split',
            'audio',
            'n_frames',
            'src_text',
            'tgt_text',
            'speaker',
            'phones',
            'words']
        writer = csv.DictWriter(csvfile, delimiter='\t', fieldnames=fieldnames)
        writer.writeheader()

        for split in ['TRAIN', 'TEST']:
            split_dir = os.path.join(root_dir, split)
            for dr in os.listdir(split_dir):
                dr_path = os.path.join(split_dir, dr)
                if os.path.isdir(dr_path):
                    for speaker in os.listdir(dr_path):
                        speaker_path = os.path.join(dr_path, speaker)
                        if os.path.isdir(speaker_path):
                            for sentence in os.listdir(speaker_path):
                                if sentence.endswith('.wav') and sentence.startswith('SX'):
                                    sentence_id = sentence[:-4]  # Remove .WAV extension
                                    audio_path = os.path.join(speaker_path, f'{sentence_id}.wav')
                                    txt_path = os.path.join(speaker_path, f'{sentence_id}.TXT')
                                    phn_path = os.path.join(speaker_path, f'{sentence_id}.PHN')
                                    wrd_path = os.path.join(speaker_path, f'{sentence_id}.WRD')

                                    if os.path.exists(txt_path):
                                        n_frames = get_wav_info(audio_path)
                                        if n_frames is None:
                                            continue  # Skip this entry if WAV file is not valid
                                        src_text = process_txt_file(txt_path)
                                        tgt_text = src_text  # tgt_text is the same as src_text
                                        unique_id = f'{dr}_{speaker}_{sentence_id}'

                                        writer.writerow({
                                            'id': unique_id,
                                            'split': split,
                                            'audio': audio_path,
                                            'n_frames': n_frames,
                                            'src_text': src_text,
                                            'tgt_text': tgt_text,
                                            'speaker': speaker,
                                            'phones': phn_path,
                                            'words': wrd_path})


def main():
    root_dir = ''  # Path to the TIMIT data directory
    output_file = ''  # Path to the tsv output file
    generate_tsv(root_dir, output_file)


if __name__ == '__main__':
    main()
