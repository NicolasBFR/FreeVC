# FreeVC: Towards High-Quality Text-Free One-Shot Voice Conversion

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2210.15418)
![GitHub Repo stars](https://img.shields.io/github/stars/NicolasBFR/FreeVC)
![GitHub](https://img.shields.io/github/license/NicolasBFR/FreeVC)

In this [paper](https://arxiv.org/abs/2210.15418), we adopt the end-to-end framework of [VITS](https://arxiv.org/abs/2106.06103) for high-quality waveform reconstruction, and propose strategies for clean content information extraction without text annotation. We disentangle content information by imposing an information bottleneck to [WavLM](https://arxiv.org/abs/2110.13900) features, and propose the **spectrogram-resize** based data augmentation to improve the purity of extracted content information.

[🤗 Play online at HuggingFace Spaces](https://huggingface.co/spaces/OlaWod/FreeVC).

Visit our [demo page](https://olawod.github.io/FreeVC-demo) for audio samples.

We also provide the [pretrained models](https://1drv.ms/u/s!AnvukVnlQ3ZTx1rjrOZ2abCwuBAh?e=UlhRR5).

<table style="width:100%">
  <tr>
    <td><img src="./resources/train.png" alt="training" height="200"></td>
    <td><img src="./resources/infer.png" alt="inference" height="200"></td>
  </tr>
  <tr>
    <th>(a) Training</th>
    <th>(b) Inference</th>
  </tr>
</table>

## Pre-requisites

1. Clone this repo: `git clone https://github.com/NicolasBFR/FreeVC.git`

2. CD into this repo: `cd FreeVC`

3. Install python requirements: `pip install -r requirements.txt`

4. Download [WavLM-Large](https://github.com/microsoft/unilm/tree/master/wavlm) and put it under directory 'wavlm/'

5. Download the [VCTK](https://datashare.ed.ac.uk/handle/10283/3443) dataset (for training only)

## Inference Example

Download the pretrained checkpoints and run:

```python
# inference with FreeVC
python3 convert.py --hpfile configs/freevc.json --ptfile checkpoints/freevc.pth --txtpath convert.txt --outdir outputs/freevc
```

## Training Example

1. Preprocess

```python
python3 downsample.py --in_dir </path/to/VCTK/wavs>

# run this if you want a different train-val-test split
python3 preprocess_flist.py

# run this if you want to use pretrained speaker encoder
python3 preprocess_spk.py

# run this if you want to train without SR-based augmentation
python3 preprocess_ssl.py
```

2. Train

```python
# train freevc
python3 train.py -c configs/freevc.json -m freevc
```
