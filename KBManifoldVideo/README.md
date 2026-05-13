# Temporal Topological CNN T(K^t)

Video extension of K^t topological first layer for frame-pair and multi-frame classification.
This repository evaluates temporal manifold lifting within an identical downstream architecture (Conv→FC→Softmax).

## Models

| Model | Description |
|---|---|
| **ManifoldTKt** | Temporal K^t lifting (spatial + temporal frame differences) with iterative refinement. |
| **KtSpatialOnly** | Spatial-only K^t control — ignores the second frame. |
| **TKt5Frame** | N=5 temporal order lifting via binomial finite differences. |
| **Baseline** | Standard convolutional layer on the frame pair. |

## Setup

```bash
pip install -r requirements.txt
```

MNIST downloads automatically. All other datasets (KTH, Weizmann, Jester, SSv2, UCF-101) must be downloaded independently and converted to `.npz` files under `data/`. See `data.py` for expected filenames.

## Usage

```bash
python -u experiments/tangent_comparison.py       # MNIST frame pairs
python -u experiments/kth_experiment.py           # KTH 2-frame action recognition
python -u experiments/broader_video_experiment.py # Jester / SSv2 / UCF-101
python -u experiments/directional_experiment.py   # directional subset discrimination
python -u experiments/five_frame_experiment.py    # KTH 5-frame with ResNet backbone
```

## Project Structure

- `lifting.py`: Frozen K^t and T(K^t) lifting layers.
- `iteration_block.py`: Four-path interaction engine.
- `reconstruction.py`: Self-supervised contrast-weighted reconstruction loss.
- `models.py`: All model classes and experiment spec helpers.
- `data.py`: Dataset loaders for MNIST, KTH, Weizmann, Jester, SSv2, UCF-101.
- `train.py`: Training loop with SepLR support and cosine schedule.
- `experiments/`: One script per thesis subsection.
