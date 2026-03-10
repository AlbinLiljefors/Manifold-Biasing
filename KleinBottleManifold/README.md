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