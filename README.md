# Algebraic Representations of the Klein Bottle for Efficient Feature Engineering in Image and Video Classification

Master's thesis — Uppsala University, 2026. Supervised by Prof. Gunnar Carlsson (Stanford).

> Paper link will be added upon publication.

## Overview

Natural image patches concentrate on a Klein bottle $\mathcal{K}$ after normalization. Instead of computing frozen filter banks by integrating over submanifolds of $\mathcal{K}$ (which does not scale to video), this work augments each pixel with algebraic manifold coordinates derived from its local neighbourhood. A frozen $3\times3$ convolution extracts gradient coordinates, a four-path iteration block refines them, and a learnable convolution forms the first layer. The approach extends naturally to video via the tangent bundle $T(\mathcal{K}^t)$.

## Repository Structure

```
Manifold-Biasing/
├── Manifold/             # Ch. 5 — S¹ proof of concept (MNIST, SVHN)
├── KleinBottleManifold/  # Ch. 6 — Full K and Kᵗ comparison (MNIST, SVHN)
└── KBManifoldVideo/      # Ch. 7 — Temporal T(Kᵗ) extension (KTH, Jester, SSv2, UCF-101)
```

Each directory is self-contained with its own `requirements.txt`. `KBManifoldVideo/` contains an `experiments/` folder with one script per thesis section.

## Running Experiments

```bash
# Ch. 5 — S¹ proof of concept
cd Manifold && pip install -r requirements.txt
python main.py

# Ch. 6 — Klein bottle comparison
cd KleinBottleManifold && pip install -r requirements.txt
python main.py

# Ch. 7 — Video extension
cd KBManifoldVideo && pip install -r requirements.txt
python -u experiments/tangent_comparison.py       # MNIST frame pairs (controlled ablation)
python -u experiments/kth_experiment.py           # KTH 2-frame action recognition
python -u experiments/broader_video_experiment.py # Jester / SSv2 / UCF-101
python -u experiments/directional_experiment.py   # directional subset discrimination
python -u experiments/five_frame_experiment.py    # KTH 5-frame, 8-layer ResNet
```

MNIST downloads automatically. KTH, Jester, SSv2, UCF-101, and Weizmann must be downloaded independently and converted to `.npz` files under `KBManifoldVideo/data/`. See `KBManifoldVideo/data.py` for expected filenames.

## Citation

> Citation will be added upon publication.
