# Klein Bottle Topological CNN

Comparison of Klein bottle manifold coordinate injection as CNN first layers.
All models share identical downstream (Conv->FC->Softmax); only the first layer differs.

## Models

| Model | First Layer | Learnable 1st-layer params |
|-------|------------|---------------------------|
| ManifoldKtLegRaw | FrozenLiftKt(6ch) + KtIterBlock + Conv(7->64) | 4,600 |
| ManifoldKtExactLegRaw | FrozenLiftKtExact(7ch) + KtExactIterBlock + Conv(8->64) | 5,320 |
| KleinLegFrozenNonUniform | 64 frozen sin^2-weighted Klein Legendre filters | 0 |
| ManifoldKleinLegRaw | FrozenLiftKlein(5ch) + KleinIterBlock + Conv(6->64) | 3,889 |
| Baseline | Conv2d(1->64, 3x3) | 640 |

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
# Tests
python -m pytest test_models.py -v

# Main experiment (5 models x 3 seeds, noise robustness + transfer)
python -u main.py 2>&1 | tee experiments.log

# Smoke test (1 epoch, 1 seed)
python -u main.py --epochs 1 --seeds 42 --output /tmp/smoke_test.json
```

## Key Results

**FINAL** — produced on Google Colab (CUDA/T4 GPU) on 2026-03-21. Lambda supercloud was unavailable; Colab used as alternative remote cloud resource. 5 models × 3 seeds × 5 epochs, ~98 min total. Results in `data/comparison_results.json`.

| Model | Clean MNIST | σ=1.0 | M→S | S→M |
|-------|-------------|-------|-----|-----|
| ManifoldKtLegRaw | 98.7±0.1% | 94.1±0.3% | 19.2±2.0% | 57.3±0.5% |
| ManifoldKleinLegRaw | 98.6±0.1% | 91.9±1.1% | 26.2±0.3% | 56.6±1.9% |
| ManifoldKtExactLegRaw | 98.7±0.1% | 90.5±1.5% | 8.8±1.1% | 56.3±2.3% |
| KleinLegFrozenNonUniform | 98.1±0.1% | 85.2±1.7% | 35.7±0.7% | 51.6±1.4% |
| Baseline | 97.9±0.0% | 69.3±2.2% | 11.5±2.0% | 51.2±0.7% |

## Status

**COMPLETE.** Final experiments run on Google Colab (T4 GPU, CUDA) 2026-03-21. SepLR config uses S^1 CV values (1e-3, 5e-6, 0.05, weight_decay=0.01). All protocols completed: noise robustness, transfer, Love noise, sample efficiency, rate of learning. Values transferred to `results.tex` placeholders.

## Missing Experiments for Love-Comparable Figures (optional, not blocking)

Same gaps as `Manifold/` — the following would produce figures directly comparable to Love et al. (JMLR 2023). `train.py` already supports `log_interval`+`eval_loader`.

**Love Fig 4 — Accuracy+loss vs batches under noise.** Train under noise with `log_interval` logging. Currently only final accuracy per noise level is recorded.

**Love Fig 5 — τ/ω² sweep.** Sweep τ ∈ [0, 0.8] and ω² ∈ [0, 0.6] independently. Currently only single-point evaluation at (τ=0.2, ω²=0.04).

**Love Fig 6 — Sample efficiency bar chart + RoL curves.** Bar chart at n=1000 across MNIST/SVHN + accuracy vs images-seen curves. Sample efficiency protocol now exists in `main.py`; RoL logging also added. Bar chart is a plotting task, not an experiment gap.

**Love Fig 7 — Transfer learning curves.** Log cross-dataset eval accuracy at intervals during training (not just final). Requires adding `eval_loader` for the *other* dataset during training loops.