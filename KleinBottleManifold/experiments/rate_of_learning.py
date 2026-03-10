import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')

import json
import time
import random

import numpy as np
import torch

from data import load_mnist, load_svhn, load_usps
from train import train_model, evaluate_model
from models import get_model_specs, SEPLR_CONFIG
from metrics import count_first_layer_params, compute_convergence_metrics

if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'
EPOCHS = 1
LR = 1e-4
BATCH_SIZE = 100
LOG_INTERVAL = 20
SEEDS = [42, 123, 456]
DATASETS = ['MNIST', 'SVHN', 'USPS']
RESULTS_FILE = os.path.join(PROJECT_ROOT, 'data', 'rate_of_learning.json')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cleanup_device():
    if DEVICE == 'mps':
        torch.mps.synchronize()
        torch.mps.empty_cache()
    elif DEVICE == 'cuda':
        torch.cuda.empty_cache()


def _load_dataset(dataset_name):
    loaders = {
        'MNIST': load_mnist,
        'SVHN': load_svhn,
        'USPS': load_usps,
    }
    return loaders[dataset_name](batch_size=BATCH_SIZE)


def run_all():
    model_specs = get_model_specs()

    print(f"RATE OF LEARNING EXPERIMENT")
    print(f"5 models x 3 seeds x 3 datasets, logging every {LOG_INTERVAL} batches")
    print(f"Device: {DEVICE}, epochs={EPOCHS}, lr={LR}, batch_size={BATCH_SIZE}")
    print(f"Seeds: {SEEDS}, datasets: {DATASETS}")
    print()

    print(f"{'Model':<26} {'1st-layer':>12} {'Total':>12}")
    print("-" * 52)
    for name, cls, _, _ in model_specs:
        info = count_first_layer_params(cls())
        print(f"{name:<26} {info['first_layer_total']:>12,} {info['model_total']:>12,}")
    print()

    all_results = {}
    total_runs = len(DATASETS) * len(model_specs) * len(SEEDS)
    run_idx = 0
    t_start = time.time()

    for ds_name in DATASETS:
        print(f"\n# DATASET: {ds_name}")
        train_loader, test_loader = _load_dataset(ds_name)
        all_results[ds_name] = {}

        for name, cls, recon_lam, pg_fn in model_specs:
            all_results[ds_name][name] = {}
            print(f"\n  Model: {name} on {ds_name}")

            for seed in SEEDS:
                run_idx += 1
                set_seed(seed)
                _cleanup_device()
                print(f"    Seed {seed} ({run_idx}/{total_runs})...", end=" ", flush=True)
                t0 = time.time()

                model = cls()
                param_groups = pg_fn(model) if pg_fn else None

                model, curve = train_model(
                    model, train_loader, epochs=EPOCHS, lr=LR,
                    recon_lambda=recon_lam, device=DEVICE, verbose=False,
                    param_groups=param_groups,
                    log_interval=LOG_INTERVAL, eval_loader=test_loader,
                )

                clean_acc = evaluate_model(model, test_loader, device=DEVICE)
                conv_metrics = compute_convergence_metrics(curve)

                elapsed = time.time() - t0
                print(f"done in {elapsed:.0f}s -- "
                      f"acc={clean_acc:.1f}%, "
                      f"points={len(curve)}, "
                      f"to90={conv_metrics.get('images_to_90', 'N/A')}")

                all_results[ds_name][name][str(seed)] = {
                    'learning_curve': curve,
                    'final_acc': clean_acc,
                    'convergence': conv_metrics,
                }

                del model
                _cleanup_device()

    total_time = time.time() - t_start

    for ds_name in DATASETS:
        print(f"\nCONVERGENCE SUMMARY — {ds_name} (mean over 3 seeds)")
        print(f"{'Model':<26} {'Final Acc':>10} {'To 90%':>12} {'To 95%':>12} {'AUC':>8}")
        print("-" * 70)
        for name, _, _, _ in model_specs:
            seeds_data = all_results[ds_name][name]
            acc_vals = [v['final_acc'] for v in seeds_data.values()]
            to90_vals = [v['convergence'].get('images_to_90') for v in seeds_data.values()]
            to95_vals = [v['convergence'].get('images_to_95') for v in seeds_data.values()]
            auc_vals = [v['convergence'].get('auc', 0) for v in seeds_data.values()]

            acc_m = sum(acc_vals) / len(acc_vals)
            auc_m = sum(auc_vals) / len(auc_vals)

            to90_valid = [v for v in to90_vals if v is not None]
            to95_valid = [v for v in to95_vals if v is not None]
            to90_str = f"{sum(to90_valid)/len(to90_valid):.0f}" if to90_valid else "N/A"
            to95_str = f"{sum(to95_valid)/len(to95_valid):.0f}" if to95_valid else "N/A"

            print(f"{name:<26} {acc_m:>9.1f}% {to90_str:>12} {to95_str:>12} {auc_m:>7.1f}")

    print(f"\nTotal time: {total_time/60:.1f} minutes")

    os.makedirs(os.path.join(PROJECT_ROOT, 'data'), exist_ok=True)
    output = {
        'config': {
            'experiment': 'rate_of_learning',
            'models': '5 (4 Manifold + 1 baseline)',
            'epochs': EPOCHS, 'lr': LR, 'batch_size': BATCH_SIZE,
            'log_interval': LOG_INTERVAL, 'seeds': SEEDS,
            'datasets': DATASETS, 'device': DEVICE,
            'total_time_minutes': total_time / 60,
        },
        'results': all_results,
    }
    with open(RESULTS_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == '__main__':
    run_all()
