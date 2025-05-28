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
from pathlib import Path
from typing import Optional

from tqdm import tqdm
import zipfile

import soundfile as sf

import numpy as np
import torch


# The following functions are adapted from
# https://github.com/hlt-mt/FBK-fairseq/blob/master/examples/speech_to_text/data_utils_new.py
# https://github.com/hlt-mt/FBK-fairseq/blob/master/fairseq/data/audio/audio_utils.py


def _get_kaldi_fbank(waveform, sample_rate, n_bins=80) -> Optional[np.ndarray]:
    """Get mel-filter bank features via PyKaldi."""
    try:
        from kaldi.feat.mel import MelBanksOptions
        from kaldi.feat.fbank import FbankOptions, Fbank
        from kaldi.feat.window import FrameExtractionOptions
        from kaldi.matrix import Vector

        mel_opts = MelBanksOptions()
        mel_opts.num_bins = n_bins
        frame_opts = FrameExtractionOptions()
        frame_opts.samp_freq = sample_rate
        opts = FbankOptions()
        opts.mel_opts = mel_opts
        opts.frame_opts = frame_opts
        fbank = Fbank(opts=opts)
        features = fbank.compute(Vector(waveform), 1.0).numpy()
        return features

    except ImportError:
        return None


def _get_torchaudio_fbank(waveform, sample_rate, n_bins=80) -> Optional[np.ndarray]:
    """Get mel-filter bank features via TorchAudio."""
    try:
        import torchaudio.compliance.kaldi as ta_kaldi

        waveform = torch.from_numpy(waveform).unsqueeze(0)
        features = ta_kaldi.fbank(waveform, num_mel_bins=n_bins, sample_frequency=sample_rate)
        return features.numpy()

    except ImportError:
        return None


def create_zip(data_root: Path, zip_path: Path):
    paths = list(data_root.glob("*.npy"))
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as f:
        for path in tqdm(paths):
            f.write(path, arcname=path.name)


def is_npy_data(data: bytes) -> bool:
    return data[0] == 147 and data[1] == 78


def get_zip_manifest(zip_path: Path, zip_root: Optional[Path] = None):
    _zip_path = zip_path if zip_root is None else Path.joinpath(zip_root, zip_path)
    with zipfile.ZipFile(_zip_path, mode="r") as f:
        info = f.infolist()
    manifest = {}
    for i in tqdm(info):
        utt_id = Path(i.filename).stem
        offset, file_size = i.header_offset + 30 + len(i.filename), i.file_size
        manifest[utt_id] = f"{zip_path.as_posix()}:{offset}:{file_size}"
        with open(_zip_path, "rb") as f:
            f.seek(offset)
            data = f.read(file_size)
            assert len(data) > 1 and is_npy_data(data)
    return manifest


def extract_fbank_features(
        waveform,
        sample_rate: int,
        output_path: Optional[Path] = None,
        n_mel_bins: int = 80,
        overwrite: bool = False):
    if output_path is not None and output_path.is_file() and not overwrite:
        return

    _waveform = waveform * (2 ** 15)  # Kaldi compliance: 16-bit signed integers
    _waveform = _waveform.squeeze().numpy()

    features = _get_kaldi_fbank(_waveform, sample_rate, n_mel_bins)
    if features is None:
        features = _get_torchaudio_fbank(_waveform, sample_rate, n_mel_bins)
    if features is None:
        raise ImportError("Please install pyKaldi or torchaudio to enable fbank feature extraction")

    if output_path is not None:
        np.save(output_path.as_posix(), features)
    else:
        return features


def _convert_to_mono(
        waveform: torch.FloatTensor, sample_rate: int) -> torch.FloatTensor:
    if waveform.shape[0] > 1:
        try:
            import torchaudio.sox_effects as ta_sox
        except ImportError:
            raise ImportError("Please install torchaudio to convert multi-channel audios")
        effects = [['channels', '1']]
        return ta_sox.apply_effects_tensor(waveform, sample_rate, effects)[0]
    return waveform


def convert_to_mono(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    if waveform.shape[0] > 1:
        _waveform = torch.from_numpy(waveform)
        return _convert_to_mono(_waveform, sample_rate).numpy()
    return waveform


def main(tsv_file, feature_root, zip_path, output_tsv_file, n_mel_bins=80):
    feature_root = Path(feature_root)
    zip_path = Path(zip_path)

    with open(tsv_file, 'r') as f:

        dataset = []
        print("Extracting features...")
        next(f)  # Skip header
        for line in tqdm(f):
            id_, split, wav_path, _, src_text, tgt_text, speaker, phones, words = line.strip().split('\t')
            wav_path = Path(wav_path)
            waveform, sample_rate = sf.read(
                wav_path, dtype="float32", always_2d=True, frames=-1, start=0)
            waveform = waveform.T  # T x C -> C x T
            if waveform.shape[0] > 1:
                waveform = convert_to_mono(waveform, sample_rate)
            waveform = torch.from_numpy(waveform)

            fbank = extract_fbank_features(waveform, sample_rate, n_mel_bins=n_mel_bins)
            feature_length = fbank.shape[0]
            np.save(feature_root / f"{id_}.npy", fbank)

            dataset.append({
                'id': id_,
                'split': split,
                'audio': None,  # Will be replaced with ZIP manifest
                'n_frames': feature_length,
                'src_text': src_text,
                'tgt_text': tgt_text,
                'speaker': speaker,
                'phones': phones,
                'words': words})

    # Pack features into ZIP
    zip_path = zip_path / "fbank.zip"
    print("ZIPing features...")
    create_zip(feature_root, zip_path)

    print("Fetching ZIP manifest...")
    zip_manifest = get_zip_manifest(zip_path)

    # Update dataset entries with ZIP manifest
    for entry in dataset:
        utt_id = entry['id']
        if utt_id in zip_manifest:
            entry['audio'] = zip_manifest[utt_id]

    # Optionally, write the updated dataset back to a new TSV file
    with open(output_tsv_file, 'w') as f:
        f.write("\t".join(['id', 'split', 'audio', 'n_frames', 'src_text', 'tgt_text', 'speaker', 'phones', 'words']) + "\n")
        for entry in dataset:
            f.write(
                "\t".join([entry['id'], entry['split'], entry['audio'], str(entry['n_frames']),
                           entry['src_text'], entry['tgt_text'], entry['speaker'], entry['phones'], entry['words']]) + "\n")


if __name__ == '__main__':
    tsv_file = ''  # Path to the tsv file
    feature_root = ''  # Path to the filterbanks folder
    zip_root = ''  # Path to the zip file
    output_tsv = ''  # Path to the output tsv file

    main(tsv_file, feature_root, zip_root, output_tsv)
