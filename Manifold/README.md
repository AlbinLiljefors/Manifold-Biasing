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

# Experiments (~55 min on MPS)
python main.py --output results.json
```

## Key Results (sigma=1.0 noise, mean +/- std, 3 seeds)

- ManifoldRaw (SepLR): 95.0 +/- 0.9
- ManifoldFrozen: 87.0 +/- 1.2
- Love: 83.0 +/- 3.8
- Baseline: 69.2 +/- 2.4
