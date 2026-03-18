import math
import json
import os
import time
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from train import train_model, evaluate_model
from models import ManifoldRawNet, LoveNet, BaselineNet, make_param_groups

K_FOLDS = 3
CV_SEED = 42
EPOCHS = 5
LR = 1e-5
BATCH_SIZE = 100

if torch.cuda.is_available():
    DEVICE = 'cuda'
    torch.backends.cudnn.benchmark = True
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'

JOINT_CONV_LR_GRID = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
ITER_BLOCK_LR_GRID = [0.0, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5]
RECON_LAMBDA_GRID = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
WEIGHT_DECAY_GRID = [0.0, 1e-3, 0.01, 0.1]

# Default SepLR config (starting point)
DEFAULT_SEPLR = {
    'joint_conv_lr': 1e-4,
    'iter_block_lr': 1e-6,
    'recon_lambda': 0.1,
    'weight_decay': 0.0,
}

# Early stopping: skip remaining folds if fold 0 transfer below this
EARLY_STOP_THRESHOLD = 15.0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mean_std(values):
    m = sum(values) / len(values)
    s = (sum((v - m) ** 2 for v in values) / max(len(values) - 1, 1)) ** 0.5
    return m, s


def create_kfold_loaders(k=3, batch_size=100, seed=42):
    """Create k-fold train/val splits of MNIST training set."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    full_dataset = datasets.MNIST('./data', train=True, download=True,
                                  transform=transform)

    n = len(full_dataset)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)

    fold_size = n // k
    folds = []
    for i in range(k):
        val_start = i * fold_size
        val_end = val_start + fold_size if i < k - 1 else n
        val_idx = indices[val_start:val_end].tolist()
        train_idx = np.concatenate([indices[:val_start],
                                    indices[val_end:]]).tolist()

        train_loader = DataLoader(
            Subset(full_dataset, train_idx),
            batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True, persistent_workers=True,
        )
        val_loader = DataLoader(
            Subset(full_dataset, val_idx),
            batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True, persistent_workers=True,
        )
        folds.append((train_loader, val_loader))

    return folds


def load_svhn_test(batch_size=100):
    """Load SVHN test set (grayscale, 28x28) for transfer evaluation."""
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test = datasets.SVHN('./data', split='test', download=True, transform=transform)
    return DataLoader(test, batch_size=batch_size, shuffle=False,
                      num_workers=4, pin_memory=True)


def cv_evaluate_config(model_factory, folds, svhn_test_loader, early_stop=True):
    """Evaluate model_factory over CV folds; returns {clean, mnist_to_svhn} stats."""
    fold_results = {'clean': [], 'mnist_to_svhn': []}

    for fold_idx, (train_loader, val_loader) in enumerate(folds):
        result = model_factory()
        if isinstance(result, tuple):
            model = result[0]
            extra_kwargs = {}
            if len(result) >= 2:
                extra_kwargs['param_groups'] = result[1]
            if len(result) >= 3:
                extra_kwargs['recon_lambda'] = result[2]
        else:
            model = result
            extra_kwargs = {}

        model = train_model(
            model, train_loader, epochs=EPOCHS, lr=LR,
            device=DEVICE, verbose=False, **extra_kwargs
        )

        clean_acc = evaluate_model(model, val_loader, device=DEVICE)
        fold_results['clean'].append(clean_acc)

        transfer_acc = evaluate_model(model, svhn_test_loader, device=DEVICE)
        fold_results['mnist_to_svhn'].append(transfer_acc)

        if early_stop and fold_idx == 0:
            if transfer_acc < EARLY_STOP_THRESHOLD:
                print(f"    [EARLY STOP] Fold 0 transfer: {transfer_acc:.1f}% < {EARLY_STOP_THRESHOLD}%")
                remaining = len(folds) - 1
                for key in fold_results:
                    fold_results[key].extend([float('nan')] * remaining)
                break

    summary = {}
    for key, vals in fold_results.items():
        valid = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
        if valid:
            m, s = mean_std(valid)
        else:
            m, s = float('nan'), float('nan')
        summary[key] = {'mean': m, 'std': s, 'per_fold': vals}

    return summary


def make_seplr_factory(config):
    """Returns a factory that creates ManifoldRawNet with SepLR param groups."""
    def factory():
        model = ManifoldRawNet()
        param_groups = make_param_groups(model, config)
        return model, param_groups, config['recon_lambda']
    return factory


def run_phase1_seplr(folds, svhn_test_loader):
    """Run sequential marginal sweeps for SepLR config."""
    print("\n" + "=" * 80)
    print("PHASE 1: ManifoldRaw+SepLR sequential marginal search")
    print("Primary metric: MNIST->SVHN transfer")
    print("=" * 80)

    current_best = dict(DEFAULT_SEPLR)
    all_results = {}

    sweeps = [
        ('joint_conv_lr', JOINT_CONV_LR_GRID),
        ('iter_block_lr', ITER_BLOCK_LR_GRID),
        ('recon_lambda', RECON_LAMBDA_GRID),
        ('weight_decay', WEIGHT_DECAY_GRID),
    ]

    for step_idx, (param_name, grid) in enumerate(sweeps):
        label = chr(ord('a') + step_idx)
        print(f"\n  Step 1{label}: {param_name} sweep")
        print(f"  Grid: {grid}")
        print(f"  Current config: {current_best}")

        step_results = {}
        best_val, best_param = -1.0, current_best[param_name]

        for val in grid:
            config = dict(current_best)
            config[param_name] = val
            print(f"    {param_name}={val}...", end=" ", flush=True)
            t0 = time.time()

            factory = make_seplr_factory(config)
            res = cv_evaluate_config(factory, folds, svhn_test_loader)
            elapsed = time.time() - t0

            step_results[str(val)] = res
            transfer = res['mnist_to_svhn']['mean']
            print(f"done in {elapsed:.0f}s -- transfer={transfer:.1f}%, "
                  f"clean={res['clean']['mean']:.1f}%")

            if not math.isnan(transfer) and transfer > best_val:
                best_val = transfer
                best_param = val

        current_best[param_name] = best_param
        all_results[param_name] = step_results
        print(f"  BEST {param_name}={best_param} -> transfer={best_val:.1f}%")

    print(f"\n  FINAL BEST CONFIG: {current_best}")
    return current_best, all_results


def run_phase2_controls(folds, svhn_test_loader):
    """Evaluate Love + Baseline on same CV folds."""
    print("\n" + "=" * 80)
    print("PHASE 2: Controls (Love + Baseline)")
    print("=" * 80)

    results = {}

    for name, cls in [('Love', LoveNet), ('Baseline', BaselineNet)]:
        print(f"\n  {name}...", end=" ", flush=True)
        t0 = time.time()

        factory = lambda c=cls: c()
        res = cv_evaluate_config(factory, folds, svhn_test_loader, early_stop=False)
        elapsed = time.time() - t0

        results[name] = res
        print(f"done in {elapsed:.0f}s -- transfer={res['mnist_to_svhn']['mean']:.1f}%, "
              f"clean={res['clean']['mean']:.1f}%")

    return results


def main():
    print("=" * 80)
    print("CROSS-VALIDATION: ManifoldRaw SepLR hyperparameter search")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"CV: {K_FOLDS}-fold, seed={CV_SEED}")
    print(f"Training: epochs={EPOCHS}, lr={LR}, batch_size={BATCH_SIZE}")
    print(f"Primary metric: MNIST->SVHN transfer (Love Fig.7 comparable)")
    print(f"Early stop threshold: {EARLY_STOP_THRESHOLD}%")
    print()

    t_start = time.time()

    set_seed(CV_SEED)
    print("Creating k-fold splits...")
    folds = create_kfold_loaders(k=K_FOLDS, batch_size=BATCH_SIZE, seed=CV_SEED)
    print(f"  {K_FOLDS} folds created")

    print("Loading SVHN test set for transfer evaluation...")
    svhn_test_loader = load_svhn_test(batch_size=BATCH_SIZE)
    print(f"  SVHN test loaded\n")

    best_seplr, phase1_results = run_phase1_seplr(folds, svhn_test_loader)
    phase2_controls = run_phase2_controls(folds, svhn_test_loader)

    total_time = time.time() - t_start

    # Save
    output = {
        'config': {
            'k_folds': K_FOLDS,
            'cv_seed': CV_SEED,
            'epochs': EPOCHS,
            'lr': LR,
            'batch_size': BATCH_SIZE,
            'device': DEVICE,
            'primary_metric': 'mnist_to_svhn',
            'early_stop_threshold': EARLY_STOP_THRESHOLD,
            'total_time_minutes': total_time / 60,
        },
        'best_config': best_seplr,
        'grids': {
            'joint_conv_lr': JOINT_CONV_LR_GRID,
            'iter_block_lr': ITER_BLOCK_LR_GRID,
            'recon_lambda': RECON_LAMBDA_GRID,
            'weight_decay': WEIGHT_DECAY_GRID,
        },
        'sweep_results': phase1_results,
        'controls': phase2_controls,
    }

    os.makedirs('data', exist_ok=True)
    with open('data/cv_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nCV results saved to data/cv_results.json")

    print(f"\n{'='*80}")
    print(f"BEST CONFIG: {best_seplr}")
    print(f"Total time: {total_time/60:.1f} minutes")

    from models import MANIFOLD_RAW_SEPLR
    if best_seplr == MANIFOLD_RAW_SEPLR:
        print("Config matches MANIFOLD_RAW_SEPLR in models.py -- no update needed.")
    else:
        print(f"WARNING: Best config differs from MANIFOLD_RAW_SEPLR in models.py!")
        print(f"  models.py:  {MANIFOLD_RAW_SEPLR}")
        print(f"  CV best:    {best_seplr}")
        print(f"  Update MANIFOLD_RAW_SEPLR before running main.py!")

    print("Done!")


if __name__ == '__main__':
    main()
