# I4C SEM Image Restoration

AI-based restoration of degraded semiconductor microscopy / SEM imagery for the I4C challenge.

## Overview

This repository contains a lightweight single-channel 2x super-resolution model for grayscale SEM images.

**Input:** 128×128 grayscale image  
**Output:** 256×256 restored grayscale image

The submitted SR model uses:

- 64 feature channels
- 12 Residual Channel Attention Blocks (RCABs)
- residual reconstruction around bicubic interpolation
- PixelShuffle ×2 upsampling
- a final high-frequency refinement stage

## Held-out validation result

The model was evaluated on **640 paired held-out LR/GT validation images** using the controlled 2x super-resolution protocol:

| Metric | Result |
|---|---:|
| PSNR | **33.1968 dB** |
| SSIM | **0.8888** |
| LPIPS | **0.1301** |

PSNR and SSIM are higher-is-better metrics. LPIPS is lower-is-better.

> **Evaluation note:** the 33.1968 dB / 0.8888 SSIM / 0.1301 LPIPS numbers are for the held-out paired LR/GT super-resolution validation protocol. They should not be represented as an official degraded-input test-set score unless that exact protocol is used.

## Repository layout

```text
I4C-SEM-Image-Restoration/
├── README.md
├── inference.py
├── train.ipynb
├── requirements.txt
├── models/
│   └── sem_sr.py
├── weights/
│   └── I4C_SEM_SR_33dB.pth
├── examples/
│   └── 33db_results_figure.png
├── results/
│   └── metrics.json
└── outputs/
    └── restored test outputs / download link
```

## Installation

Create a Python environment and install the supplied dependencies:

```bash
python -m venv .venv
```

Activate it, then:

```bash
pip install -r requirements.txt
```

A GPU is recommended for faster inference. CPU inference is supported by the standalone script.

## Inference

The evaluator-facing script accepts an input directory and an output directory and does not require manual edits.

```bash
python inference.py \
    --input_dir ./test_images \
    --output_dir ./outputs \
    --weights ./weights/I4C_SEM_SR_33dB.pth
```

The script processes `.npy` grayscale arrays and common grayscale image formats.

For `.npy` inputs, the restored output is written as a `.npy` file with the same filename.

## Model weights

The trained checkpoint is:

```text
weights/I4C_SEM_SR_33dB.pth
```

If the checkpoint is too large for normal Git hosting, store it using Git LFS or a public model-storage link and replace the link in this README.

## Official test outputs

The 400 restored test outputs generated during the submission workflow are packaged separately as:

```text
I4C_restored_test_outputs.zip
```

For large files, provide the public download URL here and keep the repository itself lightweight.

## Training

The original Kaggle notebook used for the model development is supplied as:

```text
train.ipynb
```

The notebook contains the dataset preparation, model definition, training, validation, and metric evaluation workflow.

## Reproducibility

The exact inference architecture is defined in:

```text
models/sem_sr.py
```

The inference script loads the checkpoint and uses:

```text
SEMSuperResolution(features=64, blocks=12)
```

with grayscale input and 2x output reconstruction.

## Example

See:

```text
examples/33db_results_figure.png
```

for representative LR → restored → ground-truth comparisons.

## Citation / acknowledgements

This project uses concepts from residual learning and channel-attention based image super-resolution. Relevant references should be listed in the final hackathon submission.
