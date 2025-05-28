# Echoes of Phonetics: Unveiling Relevant Acoustic Cues for ASR via Feature Attribution

This repository provides instructions to reproduce the results from the Interspeech 2025 paper
[Echoes of Phonetics: Unveiling Relevant Acoustic Cues for ASR via Feature Attribution]().
The study analyzes feature attribution explanations on the [TIMIT corpus](https://catalog.ldc.upenn.edu/LDC93S1) 
using a Conformer-based state-of-the-art ASR model and the with [SPES](https://arxiv.org/abs/2411.01710) 
explainability framework.

### 📦 Preparing Data

### Preparing TIMIT Data

1. **Audio Conversion**  
The first step involves converting TIMIT audio files from SPH to WAV format using the `audio_conversion.py` script.

2. **Generate TSV Metadata**  
Use the `generate_tsv.py` script to generate a TSV file that includes information about each WAV file, along with 
its orthographic and phonetic annotations.

3. **Preprocessing**  
Run the `generate_fbanks.py` script to preprocess the data. This will produce a `${DATA_FILENAME}.tsv` 
file containing the processed metadata. `${DATA_FOLDER}` refers to the directory where this TSV file is saved.

4. **Merge Annotations**  
For the analyses, it is useful to group all orthographic and phonetic annotations (originally stored 
in one file per sample) into two JSON files. This can be done using the `merge_annotations.py` script, which generates:
- `${JSON_ORTHOGRAPHIC}`: A single JSON file containing all orthographic annotations.
- `${JSON_PHONETIC}`: A single JSON file containing all phonetic annotations.

### 🤖 Generating Saliency Maps

To generate saliency maps, we rely on the 
[SPES implementation](https://github.com/hlt-mt/FBK-fairseq/blob/master/fbk_works/XAI_FEATURE_ATTRIBUTION.md).

We begin by running standard inference to obtain the model's transcriptions:

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

As `perturb_config.yaml` use:

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

### 🔍 Analyses

All results and plots included in the paper can be reproduced using the `analyses.ipynb` notebook.

### 📄 Citation

```bibtex
@inproceedings{fucci-et-al-2025-unveiling,
title = "Echoes of Phonetics: Unveiling Relevant Acoustic Cues for ASR via Feature Attribution",
author = {Fucci, Dennis and Gaido, Marco and Negri, Matteo and Cettolo, Mauro and Bentivogli, Luisa},
booktitle = "Proc. of Interspeech 2025",
year = "2025"
address = "Rotterdam, The Netherlands"
}
```
