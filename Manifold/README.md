# Topological CNN First Layer Comparison

Clean-room comparison of topological CNN first layers on MNIST/SVHN. 
This repository evaluates different first-layer priors within an identical downstream architecture (Conv -> FC -> Softmax).

## Models

| Model | Description |
|-------|-------------|
| **ManifoldRaw** | Trainable manifold lifting with iterative refinement and self-supervised reconstruction. |
| **Love** | Reference baseline using 64 frozen primary circle filters (Love et al., JMLR 2023). |
| **ManifoldFused** | Reference baseline using 64 frozen directional derivative filters. |
| **Baseline** | Standard 3x3 Convolutional layer. |

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Run primary comparison experiments
python main.py --output data/final_results.json

# Run noise parameter sweeps
python love_sweep.py --output data/love_sweep.json
```

## Project Structure
- `analytic_filters.py`: Closed-form integration for Love's circle filters.
- `lifting.py`: Initial gradient lifting operations.
- `iteration_block.py`: Four-path interaction engine.
- `reconstruction.py`: Self-supervised contrast-weighted reconstruction loss.
- `noise.py`: Implementation of class-correlated noise protocols.
- `main.py`: Primary experiment orchestration.
- `love_sweep.py`: Parameter sweeps for noise robustness.
