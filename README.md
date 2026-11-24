# PC-MambaGAN: Parallel Contextual Mamba for Cross-Modality Medical Image Synthesis

Official PyTorch implementation of **PC-MambaGAN**, from the paper:

> **PC-MambaGAN: Parallel Contextual Mamba for Cross-Modality Medical Image Synthesis**
> *Expert Systems with Applications (ESWA), Elsevier.*
> Citation details will be added soon (see [Citation](#citation)).

PC-MambaGAN is an adversarial framework for paired cross-modality medical image synthesis. It has been evaluated on MRI ↔ CT (Gold Atlas) and PET ↔ CT (QIN-Breast and an in-house breast PET/CT cohort) translation.

> **Note:** This repository is forked from [I2I-Mamba](https://github.com/icon-lab/I2I-Mamba) and has been fully reworked. The training/testing interface (`train.py`, `test.py`, command-line options, dataset format) follows I2I-Mamba. The network, scanning strategy, fusion modules and loss functions are new.

---

## Method overview

- **Generator.** A convolutional encoder–decoder with an interleaved bottleneck, adapted from I2I-Mamba's *episodic* bottleneck. Ten bottleneck stages alternate **PC-Mamba blocks** with MobileNetV3 refinement stages.
- **PC-Mamba block**
  - **Bidirectional square-scan SSM.** Tokens are traversed in Chebyshev-radius shells (concentric squares), forward and reverse. This keeps spatially neighbouring pixels close in the scan sequence.
  - **Frequency-Aware Channel Fusion (FACF).** Mixes channels in the Fourier domain, using EinFFT-based spectral fusion.
  - **Selective Feature Modulation Aggregation Module.** Cross-gates the parallel branches against each other.
- **Training objective.** Combines adversarial, L1, MS-SSIM and feature-matching losses with a **Gray-Level Hessian Co-occurrence Matrix (GLHCM)** texture loss, which captures curvature-aware texture.

---

## Requirements

The environment is the same as I2I-Mamba's:

```
python >= 3.8.13
cuda   >= 11.6
torch >= 2.2
torchvision >= 0.17
mamba-ssm == 1.1.3
visdom
dominate
scikit-image
h5py
scipy
ml_collections
```

`mamba-ssm` needs a CUDA-capable GPU and a matching CUDA toolkit. See the [mamba repository](https://github.com/state-spaces/mamba) for installation help.

---

## Dataset preparation

Prepare datasets **exactly as described in the [I2I-Mamba repository](https://github.com/icon-lab/I2I-Mamba)**. Use its `data_preparetion/dataprep_one_to_one.py` script for one-to-one tasks. In summary:

1. Choose subjects and split them into train / val / test **at the patient level**, so that no subject appears in more than one split.
2. Extract 2D cross-sections from each co-registered 3D volume.
3. Normalise each 2D slice before training and before computing metrics.
4. Save each slice as one PNG with the **source on the left and the target on the right**, 512 × 256 px for one-to-one synthesis.
5. Give each slice the same filename in the corresponding folders.

Expected layout:

```
datasets/
└── <DATASET>/
    └── <SOURCE>_<TARGET>/        # e.g. T1_CT, PET_CT
        ├── train/
        │   ├── 0001.png
        │   └── ...
        ├── val/
        └── test/
```

With `--which_direction AtoB` the model learns **left → right** (source → target). Use `--which_direction BtoA` for the reverse direction on the same data.

**Datasets used in the paper**

| Dataset | Modalities | Directions |
|---|---|---|
| Gold Atlas | T1-MRI, T2-MRI, CT | T1 ↔ CT, T2 ↔ CT |
| QIN-Breast (TCIA) | PET, CT | PET ↔ CT |
| In-house breast PET/CT (HYPORT, HYPORT-B) | PET, CT | PET ↔ CT |

The in-house cohort is not publicly available.

---

## Training

Set your paths and experiment name first:

```bash
DATA_ROOT=datasets/GoldAtlas/T1_CT     # folder containing train/ val/ test/
EXPNAME=pcmambagan_goldatlas_t1_ct
CKPT_DIR=checkpoints
BS=1                                    # batch size
```

Then train:

```bash
python3 train.py \
  --dataroot "${DATA_ROOT}/" \
  --name "${EXPNAME}" \
  --gpu_ids 0 \
  --model i2i_mamba_one \
  --which_model_netG i2i_mamba \
  --which_direction AtoB \
  --lambda_A 10 \
  --dataset_mode aligned \
  --norm batch \
  --pool_size 0 \
  --output_nc 1 \
  --input_nc 1 \
  --loadSize 256 \
  --fineSize 256 \
  --niter 100 \
  --niter_decay 100 \
  --save_epoch_freq 5 \
  --checkpoints_dir "${CKPT_DIR}" \
  --display_id 0 \
  --lr 0.0001 \
  --batchSize "${BS}"
```

Key arguments:

| Argument | Meaning |
|---|---|
| `--which_direction` | `AtoB` = left image → right image; `BtoA` = the reverse |
| `--lambda_A` | Weight of the L1 reconstruction loss |
| `--niter` / `--niter_decay` | Epochs at constant learning rate / epochs with linearly decaying learning rate (200 epochs in total here) |
| `--lr` | Initial learning rate |
| `--save_epoch_freq` | Checkpoint saving frequency, in epochs |
| `--display_id 0` | Turns off visdom visualisation |

The model and generator names (`i2i_mamba_one`, `i2i_mamba`) are kept from I2I-Mamba for interface compatibility. In this repository they build the PC-MambaGAN network.

---

## Testing

```bash
TEST_DIR=results

python3 test.py \
  --dataroot "${DATA_ROOT}/" \
  --name "${EXPNAME}" \
  --gpu_ids 0 \
  --model i2i_mamba_one \
  --which_model_netG i2i_mamba \
  --which_direction AtoB \
  --dataset_mode aligned \
  --norm batch \
  --phase test \
  --output_nc 1 \
  --input_nc 1 \
  --how_many 10000 \
  --serial_batches \
  --fineSize 256 \
  --loadSize 256 \
  --results_dir "${TEST_DIR}" \
  --checkpoints_dir "${CKPT_DIR}"
```

| Argument | Meaning |
|---|---|
| `--results_dir` | Where synthesised images are written |
| `--how_many` | Maximum number of test images to process |
| `--serial_batches` | Keeps test images in order |
| `--which_epoch` | *(optional)* Checkpoint to load, e.g. `200` or `latest` (default) |

Use the same `--which_direction` at test time as at training time.

---

## Acknowledgements

This work builds on:

- **[I2I-Mamba](https://github.com/icon-lab/I2I-Mamba)** (Atli *et al.*, 2024). This repository is forked from their codebase. We take the idea of an episodic Mamba bottleneck from their work and redesign it as our interleaved bottleneck of PC-Mamba blocks and MobileNetV3 refinement stages. We also keep their dataset preparation and training/testing interface.
- **[HMS-MambaGAN](https://github.com/jlw9999/HMS-MambaGAN)** (Jin *et al.*, 2025). Their use of GLCM texture features in the training loss motivated our Hessian-domain GLHCM texture loss.

We also thank the authors of [ResViT](https://github.com/icon-lab/ResViT) and [Mamba](https://github.com/state-spaces/mamba), whose code I2I-Mamba itself builds on.

We thank the providers of the [Gold Atlas](https://doi.org/10.1002/mp.12748) and [QIN-Breast](https://www.cancerimagingarchive.net/) datasets.

---

## Citation

If you use this code, please cite our paper. The BibTeX entry will be added once the paper is published.

```bibtex
% Coming soon
```

Please also cite I2I-Mamba:

```bibtex
@article{atli2024i2imamba,
  title   = {I2I-Mamba: Multi-modal medical image synthesis via selective state space modeling},
  author  = {Omer F. Atli and Bilal Kabas and Fuat Arslan and Mahmut Yurt and Onat Dalmaz and Tolga {\c{C}}ukur},
  journal = {arXiv:2405.14022},
  year    = {2024}
}
```

---

## License

This project is released under the MIT License, inherited from I2I-Mamba. The original copyright notice (Copyright © 2024, ICON Lab) is kept in `LICENSE`.
