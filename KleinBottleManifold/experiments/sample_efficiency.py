import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')

import json
import time
import random

import numpy as np
import torch

from data import load_mnist, load_mnist_subset
from train import train_model, evaluate_model
from models import get_model_specs, SEPLR_CONFIG
from metrics import count_first_layer_params, compute_efficiency_ratios

if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'
EPOCHS = 5
LR = 1e-5
BATCH_SIZE = 100
TRAINING_SIZES = [1000, 5000, 15000, 30000, 60000]
NOISE_SIGMAS = [0.0, 0.5, 1.0]
SEEDS = [42, 123, 456]
RESULTS_FILE = os.path.join(PROJECT_ROOT, 'data', 'sample_efficiency.json')


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


def mean_std(values):
    m = sum(values) / len(values)
    if len(values) > 1:
        s = (sum((v - m) ** 2 for v in values) / (len(values) - 1)) ** 0.5
    else:
        s = 0.0
    return m, s


def fmt(m, s):
    return f"{m:.1f}+/-{s:.1f}"


def run_all():
    model_specs = get_model_specs()
    n_models = len(model_specs)

    print(f"SAMPLE EFFICIENCY EXPERIMENT")
    print(f"{n_models} models x {len(TRAINING_SIZES)} sizes x 3 seeds = "
          f"{n_models * len(TRAINING_SIZES) * 3} runs")
    print(f"Device: {DEVICE}, epochs={EPOCHS}, lr={LR}, batch_size={BATCH_SIZE}")
    print(f"Training sizes: {TRAINING_SIZES}, noise sigmas: {NOISE_SIGMAS}, seeds: {SEEDS}")
    print()

    print(f"{'Model':<26} {'1st-layer':>12} {'Total':>12} {'1st/Total':>10}")
    print("-" * 62)
    for name, cls, _, _ in model_specs:
        info = count_first_layer_params(cls())
        print(f"{name:<26} {info['first_layer_total']:>12,} "
              f"{info['model_total']:>12,} {info['first_layer_ratio']:>9.1%}")
    print()

    print("Loading MNIST test set...")
    _, mnist_test = load_mnist(batch_size=BATCH_SIZE)

    all_results = {}
    total_runs = n_models * len(TRAINING_SIZES) * len(SEEDS)
    run_idx = 0
    t_start = time.time()

    for name, cls, recon_lam, pg_fn in model_specs:
        all_results[name] = {}
        print(f"\nModel: {name}")

        for n_train in TRAINING_SIZES:
            all_results[name][str(n_train)] = {}
            print(f"\n  Training size: {n_train}")

            for seed in SEEDS:
                run_idx += 1
                set_seed(seed)
                _cleanup_device()
                print(f"    Seed {seed} ({run_idx}/{total_runs})...", end=" ", flush=True)
                t0 = time.time()

                train_loader, _ = load_mnist_subset(n_train, seed=seed, batch_size=BATCH_SIZE)

                model = cls()
                param_groups = pg_fn(model) if pg_fn else None

                model = train_model(
                    model, train_loader, epochs=EPOCHS, lr=LR,
                    recon_lambda=recon_lam, device=DEVICE, verbose=False,
                    param_groups=param_groups,
                )

                res = {}
                res['clean'] = evaluate_model(model, mnist_test, device=DEVICE)
                for sigma in NOISE_SIGMAS:
                    res[f'noise_{sigma}'] = evaluate_model(
                        model, mnist_test, device=DEVICE, noise_sigma=sigma)

                elapsed = time.time() - t0
                print(f"done in {elapsed:.0f}s -- "
                      f"clean={res['clean']:.1f}%, "
                      f"s=1.0={res.get('noise_1.0', 0):.1f}%")

                all_results[name][str(n_train)][str(seed)] = res

                del model
                _cleanup_device()

    total_time = time.time() - t_start

    print(f"\nCLEAN ACCURACY vs TRAINING SIZE (mean +/- std)")
    header = f"{'Model':<26}"
    for n in TRAINING_SIZES:
        header += f" {f'n={n}':>14}"
    print(header)
    print("-" * (26 + 15 * len(TRAINING_SIZES)))
    for name, _, _, _ in model_specs:
        row = f"{name:<26}"
        for n in TRAINING_SIZES:
            vals = [v['clean'] for v in all_results[name][str(n)].values()]
            m, s = mean_std(vals)
            row += f" {fmt(m, s):>14}"
        print(row)

    print(f"\nNOISE ACCURACY (s=1.0) vs TRAINING SIZE (mean +/- std)")
    header = f"{'Model':<26}"
    for n in TRAINING_SIZES:
        header += f" {f'n={n}':>14}"
    print(header)
    print("-" * (26 + 15 * len(TRAINING_SIZES)))
    for name, _, _, _ in model_specs:
        row = f"{name:<26}"
        for n in TRAINING_SIZES:
            vals = [v['noise_1.0'] for v in all_results[name][str(n)].values()]
            m, s = mean_std(vals)
            row += f" {fmt(m, s):>14}"
        print(row)

    print(f"\nEFFICIENCY RATIOS (at n=60000, accuracy above random per first-layer param)")
    print(f"{'Model':<26} {'1st-layer':>10} {'Clean eff':>12} {'Noise eff':>12}")
    print("-" * 62)
    for name, cls, _, _ in model_specs:
        info = count_first_layer_params(cls())
        fl_total = info['first_layer_total']
        vals_clean = [v['clean'] for v in all_results[name]['60000'].values()]
        vals_noise = [v['noise_1.0'] for v in all_results[name]['60000'].values()]
        clean_m = sum(vals_clean) / len(vals_clean)
        noise_m = sum(vals_noise) / len(vals_noise)
        eff = compute_efficiency_ratios(clean_m, noise_m, fl_total)
        if fl_total == 0:
            print(f"{name:<26} {'0 (frozen)':>10} {'inf':>12} {'inf':>12}")
        else:
            print(f"{name:<26} {fl_total:>10,} {eff['acc_per_param']:>12.4f} "
                  f"{eff['noise_acc_per_param']:>12.4f}")

    print(f"\nTotal time: {total_time/60:.1f} minutes")

    os.makedirs(os.path.join(PROJECT_ROOT, 'data'), exist_ok=True)
    output = {
        'config': {
            'experiment': 'sample_efficiency',
            'models': '5 (4 Manifold + 1 baseline)',
            'epochs': EPOCHS, 'lr': LR, 'batch_size': BATCH_SIZE,
            'training_sizes': TRAINING_SIZES, 'noise_sigmas': NOISE_SIGMAS,
            'seeds': SEEDS, 'device': DEVICE,
            'total_time_minutes': total_time / 60,
        },
        'results': all_results,
    }
    with open(RESULTS_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == '__main__':
    run_all()
