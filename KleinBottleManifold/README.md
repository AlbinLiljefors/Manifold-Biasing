# Klein Bottle Topological CNN

Klein bottle extension of Carlsson's K^t topological first layer for image classification.
This repository evaluates Klein bottle manifold coordinate injection as CNN first layers within an identical downstream architecture (Conv→FC→Softmax).

## Models

| Model | Description |
|---|---|
| **ManifoldKt** | Translational Klein bottle K^t lifting with iterative refinement and reconstruction loss. |
| **ManifoldKtExact** | K^t with exact spatial offset, otherwise identical to ManifoldKt. |
| **KleinFrozenNonUnif** | 64 frozen sin²-weighted Klein bottle filters. |
| **ManifoldKlein** | Full Klein bottle lifting with iterative refinement and reconstruction loss. |
| **Baseline** | Standard 3×3 convolutional layer. |

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python -u main.py
```

## Project Structure

- `lifting.py`: Frozen Klein bottle and K^t lifting layers.
- `iteration_block.py`: Four-path interaction engine.
- `reconstruction.py`: Self-supervised contrast-weighted reconstruction loss.
- `analytic_filters.py`: Closed-form Klein bottle filter generation.
- `models.py`: All model classes and experiment spec helpers.
- `data.py`: MNIST and SVHN loaders.
- `train.py`: Training loop and evaluation.
- `noise.py`: Class-correlated noise protocols.
- `main.py`: Experiment orchestration.
