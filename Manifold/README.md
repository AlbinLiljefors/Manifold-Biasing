# Topological CNN First Layer Comparison

Clean-room comparison of topological CNN first layers on MNIST/SVHN.
All models share identical downstream (Conv->FC->Softmax); only the first layer differs.

## Models

| Model | First Layer | Learnable 1st-layer params |
|-------|------------|---------------------------|
| Love | 64 frozen primary circle filters | 0 |
| ManifoldFrozen | 64 frozen directional derivative filters | 0 |
| ManifoldRaw | FrozenLift + iter block + Conv(3->64) | 1,809 |
| Baseline | Conv2d(1->64, 3x3) | 640 |

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
# Tests
python -m pytest test_models.py -v

# CV hyperparameter search (~15 min on A100)
python -u cv_search.py 2>&1 | tee cv_search.log

# Final experiments (~13 min on A100)
python -u main.py --output data/final_results.json 2>&1 | tee experiments.log
```

## Final Results (A100, 3 seeds, 5 epochs)

### Transfer & Noise Robustness

| Model | Clean | M->SVHN | S->MNIST | sigma=1.0 |
|-------|-------|---------|----------|-----------|
| ManifoldRaw | **98.6+/-0.0** | 25.7+/-1.3 | **57.1+/-0.3** | **95.1+/-0.7** |
| Love | 98.2+/-0.1 | **36.7+/-0.1** | 51.5+/-1.5 | 83.0+/-3.5 |
| ManifoldFrozen | 98.2+/-0.1 | 35.9+/-0.4 | 51.8+/-0.6 | 86.9+/-1.2 |
| Baseline | 97.9+/-0.0 | 11.6+/-2.0 | 51.6+/-1.0 | 69.2+/-2.1 |

### Love Noise Protocols (tau=0.2, omega^2=0.04)

| Model | Train Noisy/Test Clean | Train Clean/Test Noisy |
|-------|----------------------|----------------------|
| ManifoldFrozen | 82.0+/-4.4 | 97.8+/-0.3 |
| Love | 81.5+/-4.3 | 97.8+/-0.5 |
| ManifoldRaw | 72.7+/-8.4 | **98.5+/-0.1** |
| Baseline | 36.5+/-12.5 | 96.5+/-0.2 |

### CV-Tuned Config (ManifoldRaw SepLR)

Optimized for MNIST->SVHN transfer via 3-fold CV sequential marginal search.

```
joint_conv_lr=1e-3, iter_block_lr=5e-6, recon_lambda=0.05, weight_decay=0.01
```

## Status

Complete. Definitive data in `data/final_results.json` and `data/cv_results.json`.

## Missing Experiments for Love-Comparable Figures (optional, not blocking)

The following experiments would produce figures directly comparable to Love et al. (JMLR 2023). None are currently implemented. `train.py` already supports `log_interval`+`eval_loader` for rate-of-learning curves.

**Love Fig 4 — Accuracy+loss vs batches under noise.** Two panels: (1) train on noisy MNIST, plot accuracy+loss vs batches; (2) train on clean, test on noisy, plot accuracy+loss vs batches. Requires: training under noise with `log_interval` logging enabled. Currently we only record final accuracy per noise level.

**Love Fig 5 — τ/ω² sweep of Love noise parameters.** 2×2 grid: vary τ ∈ [0, 0.8] with ω²=0.04 fixed (left column), vary ω² ∈ [0, 0.6] with τ=0.2 fixed (right column). Top row: train-noisy/test-clean. Bottom row: train-clean/test-noisy. Show accuracy at 1 and 5 epochs. Currently we evaluate at single point (τ=0.2, ω²=0.04) only.

**Love Fig 6 — Sample efficiency bar chart + rate-of-learning curves.** Left: bar chart of accuracy after training on 1,000 images across MNIST, SVHN, USPS. Right: full accuracy vs images-seen curves per dataset. Requires: `load_mnist_subset(1000)` training + `log_interval` RoL logging for MNIST/SVHN. USPS not currently in our evaluation protocol.

**Love Fig 7 — Transfer learning curves (SVHN↔MNIST).** Accuracy and loss vs batches during training, evaluated on the *other* dataset throughout. Two panels: SVHN→MNIST (left), MNIST→SVHN (right). Top: accuracy, bottom: loss. Requires: logging cross-dataset eval accuracy at intervals during training. Currently we only record final transfer accuracy.
