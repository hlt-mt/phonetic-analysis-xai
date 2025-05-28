# Echoes of Phonetics: Unveiling Relevant Acoustic Cues for ASR via Feature Attribution


This repository provides instructions to reproduce the results from the Interspeech 2025 paper
[Echoes of Phonetics: Unveiling Relevant Acoustic Cues for ASR via Feature Attribution]().
The study analyzes feature attribution explanations on the [TIMIT corpus](https://catalog.ldc.upenn.edu/LDC93S1) 
using a Conformer-based state-of-the-art ASR model and the with [SPES](https://arxiv.org/abs/2411.01710) 
explainability framework.

### 📦 Data Preprocessing

The first step involves converting TIMIT audio files from SPH to WAV format. Use the following script for conversion:

```python
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

# Set TIMIT root directory
ROOT_DIR = ''

# Iterate through all .WAV files (which are actually SPH)
for root, dirs, files in os.walk(ROOT_DIR):
    for file in files:
        if file.endswith('.WAV'):
            sph_path = os.path.join(root, file)
            wav_path = sph_path.replace('.WAV', '.wav')
            convert_sph_to_wav(sph_path, wav_path)
```

Set the TIMIT root directory in the `ROOT_DIR` variable.


Next, follow the preprocessing steps described in the
[Speechformer README](https://github.com/hlt-mt/FBK-fairseq/blob/master/fbk_works/SPEECHFORMER.md#preprocessing)
to generate a `${DATA_FILENAME}.tsv` file.
Place this file in your `${DATA_FOLDER}` directory.

### 🔍 Generating Saliency Maps

To generate saliency maps, begin by running standard inference to obtain the model's transcriptions:

```bash  
python /path/to/FBK-fairseq/fairseq_cli/generate.py ${DATA_FOLDER} \
        --gen-subset ${DATA_FILENAME} \
        --user-dir examples/speech_to_text \
        --max-tokens 40000 \
        --model-overrides "{'batch_unsafe_relative_shift':False}" \
        --config-yaml config_generate.yaml \
        --beam 5 \
        --task speech_to_text_ctc \
        --criterion ctc_multi_loss \
        --underlying-criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
        --no-repeat-ngram-size 5 \
        --path ${MODEL_CHECKPOINT} > ${TRANSCRIPTION_OUTPUT}

egrep '^H' ${TRANSCRIPTION_OUTPUT} | cut -d"-" -f2- | sort -n | cut -f3 > hyp1
echo "tgt_text" | cat - hyp1 > hyp2
paste <(cut -f1-4 ${DATA_FOLDER}/${DATA_FILENAME}.tsv) <(cut -f1 hyp2) <(cut -f6 ${DATA_FOLDER}/${DATA_FILENAME}.tsv) > ${DATA_FOLDER}/${DATA_FILENAME}_explain.tsv
rm -f hyp1 hyp2
```

Here, `${MODEL_CHECKPOINT}` refers to the path of the pretrained ASR model. Pretrained models can be obtained from 
the [SBAAM](https://github.com/hlt-mt/FBK-fairseq/blob/master/fbk_works/SBAAM.md#-training). 
The resulting transcriptions are saved to `${TRANSCRIPTION_OUTPUT}` and then post-processed into
`${DATA_FILENAME}_explain.tsv`. This processed file is required for generating feature attribution maps.
The following `config_generate.yaml` should be used:

```yaml
bpe_tokenizer:
  bpe: sentencepiece
  sentencepiece_model: ${SENTENCE_PIECE_MODEL}
bpe_tokenizer_src:
  bpe: sentencepiece
  sentencepiece_model: ${SENTENCE_PIECE_MODEL}
input_channels: 1
input_feat_per_channel: 80
sampling_alpha: 1.0
specaugment:
  freq_mask_F: 27
  freq_mask_N: 1
  time_mask_N: 1
  time_mask_T: 100
  time_mask_p: 1.0
  time_wrap_W: 0
transforms:
  '*':
  - utterance_cmvn
  _train:
  - utterance_cmvn
  - specaugment
vocab_filename: ${VOCABULARY}
vocab_filename_src: ${VOCABULARY}
```

Next, compute the original token probabilities needed by SPES, and store them in `${ORIG_PROBS}`. 

```bash
python /path/to/FBK-fairseq/examples/speech_to_text/get_probs_from_constrained_decoding.py ${DATA_FOLDER} \
        --gen-subset ${DATA_FILENAME}_explain \
        --user-dir examples/speech_to_text \
        --max-tokens 10000 \
        --config-yaml config_explain.yaml \
        --task speech_to_text_ctc \
        --model-overrides "{'batch_unsafe_relative_shift':False}" \
        --criterion ctc_multi_loss \
        --underlying-criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
        --path ${MODEL_CHECKPOINT} \
        --save-file ${ORIG_PROBS}
```

Make sure to use this `config_explain.yaml` file:

```yaml
bpe_tokenizer_src:
  bpe: sentencepiece
  sentencepiece_model: ${SENTENCE_PIECE_MODEL}
input_channels: 1
input_feat_per_channel: 80
sampling_alpha: 1.0
specaugment:
  freq_mask_F: 27
  freq_mask_N: 1
  time_mask_N: 1
  time_mask_T: 100
  time_mask_p: 1.0
  time_wrap_W: 0
transforms:
  '*':
  - utterance_cmvn
  _train:
  - utterance_cmvn
  - specaugment
vocab_filename: ${VOCABULARY}
vocab_filename_src: ${VOCABULARY}

```

Now generate the saliency heatmaps, which will be saved in `${SALIENCY_MAPS}`.

```bash
python /path/to/FBK-fairseq/examples/speech_to_text/generate_occlusion_explanation.py ${DATA_FOLDER} \
        --gen-subset ${DATA_FILENAME}_explain \
        --user-dir examples/speech_to_text \
        --max-tokens 160000 \
        --num-workers 1 \
        --config-yaml config_explain.yaml \
        --perturb-config perturb_config.yaml \
        --task speech_to_text_ctc \
        --criterion ctc_multi_loss \
        --underlying-criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
        --no-repeat-ngram-size 5 \
        --path ${MODEL_CHECKPOINT} \
        --original-probs ${ORIG_PROBS} \
        --save-file ${SALIENCY_MAPS}
```

As recommended by SPES, encoder and decoder saliency maps are generated separately, using two different perturbation 
configurations (`perturb_config.yaml`).

For filterbank-based (encoder) explanations, use:

```yaml
fbank_occlusion:
  category: slic_fbank_dynamic_segments
  p: 0.5
  n_segments: [2000, 2500, 3000]
  threshold_duration: 750
  slic_sigma: 0
  n_masks: 20000
decoder_occlusion:
  category: discrete_embed
  p: 0.0
  no_position_occlusion: true
scorer: KL
```

For decoder (previous token) explanations, use:

```yaml
fbank_occlusion:
  category: slic_fbank_dynamic_segments
  p: 0.0
  n_masks: 2000
decoder_occlusion:
  category: discrete_embed
  p: 0.4
  no_position_occlusion: true
scorer: KL
```

Then the two explanations are merged using the following code:

```python
import argparse
from typing import Dict

import h5py
import torch
from torch import Tensor


def read_feature_attribution_maps_from_h5(explanation_path: str) -> Dict[int, Dict[str, Tensor]]:
    explanations = {}
    with h5py.File(explanation_path, "r") as f:
        for key in f.keys():
            explanations[int(key)] = {}
            group = f[key]
            explanations[int(key)]["fbank_heatmap"] = torch.from_numpy(
                group["fbank_heatmap"][()])
            explanations[int(key)]["tgt_embed_heatmap"] = torch.from_numpy(
                group["tgt_embed_heatmap"][()])
            tgt_txt = group["tgt_text"][()]
            explanations[int(key)]["tgt_text"] = [x.decode('UTF-8') for x in tgt_txt.tolist()]
    return explanations


def merge_explanations(
        fbank_explanations: Dict[int, Dict[str, Tensor]],
        tgt_explanations: Dict[int, Dict[str, Tensor]]) -> Dict[int, Dict[str, Tensor]]:
    for key in fbank_explanations.keys():
        try:
            tgt = tgt_explanations[key]
        except KeyError:
            raise KeyError("key {} is missing tgt_embed_heatmap".format(key))
        assert fbank_explanations[key]["tgt_text"] == tgt["tgt_text"], f"For key {key} there is no correspondance between fbank and tgt.\nThese are the two texts:\n{fbank_explanations[key]['tgt_text']}\n{tgt['tgt_text']}"
        fbank_explanations[key]["tgt_embed_heatmap"] = tgt["tgt_embed_heatmap"]
    return fbank_explanations


def write_explanations_to_h5(explanations: Dict[int, Dict[str, Tensor]], save_path: str) -> None:
    with h5py.File(save_path, "w") as f:
        for sample_id in explanations.keys():
            group = f.create_group(str(sample_id))
            for key, value in explanations[sample_id].items():
                group.create_dataset(
                    key, data=value.cpu() if type(value) == Tensor else value)


def main(fbank_explanation_path: str, tgt_explanation_path: str, save_path: str):
    fbank_explanation = read_feature_attribution_maps_from_h5(fbank_explanation_path)
    tgt_explanation = read_feature_attribution_maps_from_h5(tgt_explanation_path)
    merged = merge_explanations(fbank_explanation, tgt_explanation)
    write_explanations_to_h5(merged, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge feature attribution maps from two h5 files")
    parser.add_argument("--fbank-path", type=str, help="Path to fbank explanation h5 file")
    parser.add_argument("--tgt-path", type=str, help="Path to tgt explanation h5 file")
    parser.add_argument("--save-path", type=str, help="Path to save merged explanations h5 file")
    args = parser.parse_args()

    main(args.fbank_path, args.tgt_path, args.save_path)
```

### Analyses

All results and plots included in the paper can be reproduced using 
[this Colab notebook](https://github.com/hlt-mt/phonetic-analysis-xai/blob/main/analyses.ipynb).

### Citation

```bibtex
@inproceedings{fucci-et-al-2025-unveiling,
title = "Echoes of Phonetics: Unveiling Relevant Acoustic Cues for ASR via Feature Attribution",
author = {Fucci, Dennis and Gaido, Marco and Negri, Matteo and Cettolo, Mauro and Bentivogli, Luisa},
booktitle = "Proc. of Interspeech 2025",
year = "2025"
address = "Rotterdam, The Netherlands"
}
```
