# Klein Bottle Topological CNN

Comparison of Klein bottle manifold coordinate injection as CNN first layers on MNIST/SVHN.
All models share an identical downstream architecture (Conv → FC → Softmax); only the first layer differs.

This code accompanies the master's thesis *Topological Convolutional Neural Networks* (KTH, 2026), supervised by Prof. Gunnar Carlsson (Stanford).

## Models

| Model | Description |
|-------|-------------|
| **ManifoldKt** | Translational Klein bottle K^t: 6-channel frozen lift + learnable iter_block + joint conv. |
| **ManifoldKtExact** | K^t variant with exact spatial offset r computed from the patch norm. |
| **KleinFrozenNonUnif** | 64 frozen sin²-weighted Klein bottle scaled filters, no learnable first-layer params. |
| **ManifoldKlein** | Full Klein bottle K: 5-channel frozen lift + learnable iter_block + joint conv. |
| **Baseline** | Standard 3×3 convolutional layer. |

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Run full comparison (5 models x 3 seeds)
python -u main.py 2>&1 | tee experiments.log

# Smoke test (1 epoch, 1 seed)
python -u main.py --epochs 1 --seeds 42 --output /tmp/smoke_test.json
```

## Project Structure

- `lifting.py`: Frozen gradient kernels for Klein bottle coordinate extraction.
- `analytic_filters.py`: Closed-form scaled filter generator for KleinFrozenNonUnif.
- `iteration_block.py`: Four-path pixel/manifold interaction block.
- `reconstruction.py`: Contrast-weighted self-supervised reconstruction loss.
- `noise.py`: Class-correlated noise protocols (Love et al., JMLR 2023).
- `data.py`: MNIST and SVHN dataset loaders.
- `train.py`: Training and evaluation routines.
- `main.py`: Experiment orchestration — noise robustness, transfer, sample efficiency, rate of learning.
