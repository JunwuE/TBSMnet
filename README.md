# TBSMnet: Transformer and Trainable Bilateral Filter for Unsupervised Stereo Matching

## Overview

<img width="800" src="https://github.com/JunwuE/TBSMnet/blob/main/overview.png"/>

TBSMnet is an **unsupervised stereo matching** network that combines **Transformer-based feature extraction** with a **trainable bilateral filter** for accurate disparity estimation without ground-truth supervision.

### Key Features

- **Hybrid CNN-Transformer Backbone**: An hourglass encoder-decoder architecture integrated with DeepViT for capturing both local details and global contextual information.
- **Cascaded Parallax-Attention Module (PAM)**: Multi-scale parallax attention blocks that iteratively refine stereo correspondence at three scales.
- **Trainable Bilateral Filter**: A learnable bilateral filter applied to cost volumes for edge-preserving smoothing during disparity regression.
- **Disparity Refinement**: A dedicated refinement module that further improves initial disparity predictions.
- **Fully Unsupervised Training**: End-to-end training using only photometric, smoothness, and cycle-consistency losses — no ground-truth disparity labels required.

---

## Requirements

- Python 3.7+
- PyTorch ≥ 1.7
- CUDA (GPU training recommended)
- Additional packages:
  ```
  numpy
  scikit-image
  prefetch_generator
  vit-pytorch
  ```

Install dependencies:

```bash
pip install numpy scikit-image prefetch_generator vit-pytorch
```

---

## Project Structure

```
TBSMnet/
├── models/
│   ├── TBSMnet.py            # Main network architecture
│   ├── modules.py            # Core modules (ResB, EncoderB, DecoderB, PAB, Output, BilateralFilter, etc.)
│   ├── shuffle_attention.py  # Shuffle Attention module
│   └── util/                 # Model utilities
├── datasets/
│   ├── __init__.py           # Dataset registry
│   ├── sceneflow_dataset.py  # Scene Flow dataset loader
│   ├── kitti_dataset.py      # KITTI 2012/2015 dataset loader
│   └── data_io.py            # Data I/O utilities
├── train.py                  # Pre-training on Scene Flow
├── finetune.py               # Fine-tuning on KITTI
├── submission.py             # Generate submission results for KITTI benchmark
├── loss.py                   # Unsupervised loss functions
├── train.sh                  # Training script
├── finetune.sh               # Fine-tuning script
├── submission.sh             # Submission script
├── dirmake_scenflow.py       # Generate Scene Flow file lists
├── dirmake_kitti.py          # Generate KITTI file lists
└── overview.png              # Architecture overview figure
```

---

## Dataset Preparation

### Scene Flow

1. Download the [Scene Flow Datasets](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) (FlyingThings3D, Driving, Monkaa).
2. Organize the data as follows:
   ```
   data/
   └── scene_flow/
       ├── driving/
       │   └── <subfolders>/
       │       ├── left/
       │       └── right/
       ├── flyingthing/
       │   └── <subfolders>/
       │       ├── left/
       │       └── right/
       └── monkaa/
           └── <subfolders>/
               ├── left/
               └── right/
   ```
3. Generate file lists:
   ```bash
   python dirmake_scenflow.py
   ```

### KITTI 2015

1. Download [KITTI Stereo 2015](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo).
2. Organize the data as follows:
   ```
   data/
   └── kitti_2015/
       ├── training/
       │   ├── image_2/    # Left images
       │   └── image_3/    # Right images
       └── testing/
           ├── image_2/
           └── image_3/
   ```
3. Generate file lists:
   ```bash
   python dirmake_kitti.py
   ```

---

## Training

### Stage 1: Pre-training on Scene Flow

```bash
python train.py --max_disp=192 \
                --datapath='data'
```

Or use the provided script:

```bash
bash train.sh
```

**Key arguments:**

| Argument         | Default       | Description                                    |
|------------------|---------------|------------------------------------------------|
| `--max_disp`     | `0`           | Prior maximum disparity (0 = unavailable)      |
| `--datapath`     | `data`        | Path to the dataset root                       |
| `--savepath`     | `log/`        | Directory to save checkpoints                  |
| `--lr`           | `1e-3`        | Initial learning rate                          |
| `--batch_size`   | `16`          | Training batch size                            |
| `--n_epochs`     | `10`          | Number of training epochs                      |
| `--n_workers`    | `2`           | Number of data loading workers                 |
| `--resume_model` | `None`        | Path to a checkpoint for resuming training     |
| `--print_freq`   | `100`         | Print loss every N iterations                  |
| `--save_freq`    | `2`           | Save checkpoint every N epochs                 |

### Stage 2: Fine-tuning on KITTI

```bash
python finetune.py --max_disp=192 \
                   --datapath='data' \
                   --resume_model='log/TBSMnet_192/TBSMnet_192_scenceflow_kitti_epoch10.pth'
```

Or use the provided script:

```bash
bash finetune.sh
```

Fine-tuning uses a lower learning rate (`1e-4`) and trains for `80` epochs by default with adjusted loss weights for domain adaptation.

---

## Evaluation & Submission

Generate disparity predictions for KITTI benchmark submission:

```bash
python submission.py --max_disp=192 \
                     --datapath='data' \
                     --ckpt='log/TBSMnet_192/TBSMnet_192_kitti_epoch100.pth.tar'
```

Or use the provided script:

```bash
bash submission.sh
```

Results will be saved to `results/<dataset>_<max_disp>/` as 16-bit PNG disparity maps, ready for upload to the [KITTI benchmark](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo).

---

## Loss Functions

TBSMnet uses a combination of unsupervised loss functions defined in `loss.py`:

| Loss Component              | Description                                                              |
|-----------------------------|--------------------------------------------------------------------------|
| **Photometric Loss**        | L1 + SSIM between the left image and the warped right image              |
| **Smoothness Loss**         | Edge-aware disparity smoothness weighted by image gradients              |
| **PAM Photometric Loss**    | Multi-scale photometric consistency on PAM attention maps                |
| **PAM Cycle Loss**          | Cycle-consistency constraint on left-right-left attention composition    |
| **PAM Smoothness Loss**     | Spatial smoothness regularization on attention maps                      |

---

## Architecture Details

1. **Feature Extraction**: A multi-scale hourglass encoder-decoder with residual blocks extracts features at scales 1, 1/2, 1/4, 1/8, 1/16, and 1/32. A DeepViT branch provides global context that is fused with CNN features.
2. **Cascaded PAM**: Three-stage parallax attention progressively builds stereo cost volumes at 1/4, 1/8, and 1/16 scales, each stage containing four PAB blocks.
3. **Cost Aggregation**: The trainable bilateral filter smooths the cost volume in an edge-preserving manner with learnable spatial and range kernels.
4. **Disparity Output**: Soft-argmin regression converts the cost volume to a continuous disparity map, with occlusion masks derived via morphological processing.
5. **Refinement**: A lightweight U-Net-style refinement module polishes the final disparity using left image features.

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{TBSMnet,
  title={Transformer and Trainable Bilateral Filter for Unsupervised Stereo Matching},
  author={},
  journal={},
  year={}
}
```

---

## Acknowledgements

This project builds upon ideas from several works in unsupervised stereo matching, parallax attention mechanisms, and vision transformers. The disparity warping function is adapted from [MonoDepth-PyTorch](https://github.com/OniroAI/MonoDepth-PyTorch).

---

## License

Please refer to the repository for licensing information.
