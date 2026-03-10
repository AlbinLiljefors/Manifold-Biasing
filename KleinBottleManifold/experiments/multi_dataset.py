import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')

import json
import time
import random

import numpy as np
import torch

from data import load_mnist, load_usps, load_cifar10
from train import train_model, evaluate_model
from models import get_model_specs, SEPLR_CONFIG

if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'
EPOCHS = 5
LR = 1e-5
BATCH_SIZE = 100
NOISE_SIGMAS = [0.0, 0.5, 1.0]
SEEDS = [42, 123, 456]
RESULTS_FILE = os.path.join(PROJECT_ROOT, 'data', 'multi_dataset.json')


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

    print(f"MULTI-DATASET EVALUATION")
    print(f"{n_models} models x 3 seeds, train on MNIST, eval on MNIST/USPS/CIFAR-10")
    print(f"Device: {DEVICE}, epochs={EPOCHS}, lr={LR}, batch_size={BATCH_SIZE}")
    print(f"Noise sigmas: {NOISE_SIGMAS}, seeds: {SEEDS}")
    print()

    print("Loading MNIST...")
    mnist_train, mnist_test = load_mnist(batch_size=BATCH_SIZE)
    print("Loading USPS...")
    usps_train, usps_test = load_usps(batch_size=BATCH_SIZE)
    print("Loading CIFAR-10...")
    _, cifar_test = load_cifar10(batch_size=BATCH_SIZE)

    all_results = {}
    total_runs = n_models * len(SEEDS)
    run_idx = 0
    t_start = time.time()

    for name, cls, recon_lam, pg_fn in model_specs:
        all_results[name] = {}
        print(f"\nModel: {name}" + (f"  (recon_lambda={recon_lam})" if recon_lam > 0 else ""))

        for seed in SEEDS:
            run_idx += 1
            set_seed(seed)
            _cleanup_device()
            print(f"\n  Seed {seed} ({run_idx}/{total_runs})...", flush=True)
            t0 = time.time()

            model = cls()
            param_groups = pg_fn(model) if pg_fn else None

            try:
                dev = DEVICE
                model = train_model(model, mnist_train, epochs=EPOCHS, lr=LR,
                                    recon_lambda=recon_lam, device=dev, verbose=False,
                                    param_groups=param_groups)
            except RuntimeError as e:
                if 'command buffer' in str(e) or 'MPS' in str(e):
                    print(f"    [MPS error, retrying on CPU]", flush=True)
                    _cleanup_device()
                    set_seed(seed)
                    dev = 'cpu'
                    model = cls()
                    param_groups = pg_fn(model) if pg_fn else None
                    model = train_model(model, mnist_train, epochs=EPOCHS, lr=LR,
                                        recon_lambda=recon_lam, device=dev, verbose=False,
                                        param_groups=param_groups)
                else:
                    raise

            res = {}
            for ds_name, loader in [('mnist', mnist_test), ('usps', usps_test), ('cifar', cifar_test)]:
                print(f"    {ds_name.upper()}: ", end="", flush=True)
                for sigma in NOISE_SIGMAS:
                    acc = evaluate_model(model, loader, device=dev, noise_sigma=sigma)
                    res[f'{ds_name}_noise_{sigma}'] = acc
                    print(f"s={sigma}:{acc:.1f}%", end=" ", flush=True)
                print()

            res['mnist_to_usps'] = res['usps_noise_0.0']
            elapsed = time.time() - t0
            print(f"    Done in {elapsed:.0f}s")

            all_results[name][str(seed)] = res
            del model
            _cleanup_device()

    total_time = time.time() - t_start

    model_names = [name for name, _, _, _ in model_specs]
    for ds_name, label in [('mnist', 'MNIST'), ('usps', 'USPS'), ('cifar', 'CIFAR-10')]:
        print(f"\n{label} NOISE ROBUSTNESS (mean +/- std)")
        header = f"{'Model':<26}"
        for sigma in NOISE_SIGMAS:
            header += f" {'s='+str(sigma):>14}"
        print(header)
        print("-" * (26 + 15 * len(NOISE_SIGMAS)))
        for name in model_names:
            row = f"{name:<26}"
            for sigma in NOISE_SIGMAS:
                vals = [all_results[name][str(s)][f'{ds_name}_noise_{sigma}'] for s in SEEDS]
                m, s = mean_std(vals)
                row += f" {fmt(m, s):>14}"
            print(row)

    print(f"\nMNIST->USPS TRANSFER (mean +/- std)")
    print(f"{'Model':<26} {'M->USPS':>14}")
    print("-" * 42)
    for name in model_names:
        vals = [all_results[name][str(s)]['mnist_to_usps'] for s in SEEDS]
        m, s = mean_std(vals)
        print(f"{name:<26} {fmt(m, s):>14}")

    print(f"\nTotal time: {total_time/60:.1f} minutes")

    os.makedirs(os.path.join(PROJECT_ROOT, 'data'), exist_ok=True)
    output = {
        'config': {
            'experiment': 'multi_dataset',
            'models': '5 (4 Manifold + 1 baseline)',
            'epochs': EPOCHS, 'lr': LR, 'batch_size': BATCH_SIZE,
            'noise_sigmas': NOISE_SIGMAS, 'seeds': SEEDS,
            'device': DEVICE, 'total_time_minutes': total_time / 60,
        },
        'results': all_results,
    }
    with open(RESULTS_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == '__main__':
    run_all()
