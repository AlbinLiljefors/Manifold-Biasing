import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')

import json
import time
import random

import numpy as np
import torch

from data import load_mnist, load_svhn
from train import train_model, evaluate_model
from models import (ManifoldKtLegRaw, ManifoldKleinLegRaw,
                    Baseline, make_param_groups, SEPLR_CONFIG)

if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'
LR = 1e-5
BATCH_SIZE = 100
EPOCH_COUNTS = [5, 20, 50, 100]
NOISE_SIGMAS = [0.0, 0.5, 1.0]
SEEDS = [42, 123, 456]
RESULTS_FILE = os.path.join(PROJECT_ROOT, 'data', 'epoch_sweep.json')
CHECKPOINT_FILE = os.path.join(PROJECT_ROOT, 'data', 'epoch_sweep_checkpoint.json')

MODEL_SPECS = [
    ('ManifoldKtLegRaw',     ManifoldKtLegRaw,      SEPLR_CONFIG['recon_lambda'], make_param_groups),
    ('ManifoldKleinLegRaw',  ManifoldKleinLegRaw,  SEPLR_CONFIG['recon_lambda'], make_param_groups),
    ('Baseline',             Baseline,              0.0, None),
]


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


def _load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {}


def _save_checkpoint(checkpoint):
    os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)


def _checkpoint_key(name, epochs, seed):
    return f"{name}__ep{epochs}__seed{seed}"


def run_single(model_cls, recon_lam, pg_fn, epochs, mnist_train, mnist_test,
               svhn_train, svhn_test, device=None):
    dev = device or DEVICE
    model = model_cls()
    param_groups = pg_fn(model) if pg_fn else None

    model = train_model(model, mnist_train, epochs=epochs, lr=LR,
                        recon_lambda=recon_lam, device=dev, verbose=False,
                        param_groups=param_groups)

    res = {}
    for sigma in NOISE_SIGMAS:
        res[f'noise_{sigma}'] = evaluate_model(model, mnist_test, device=dev,
                                                noise_sigma=sigma)
    del model
    _cleanup_device()

    fresh = model_cls()
    fresh_pg = pg_fn(fresh) if pg_fn else None
    fresh = train_model(fresh, svhn_train, epochs=epochs, lr=LR,
                        recon_lambda=recon_lam, device=dev, verbose=False,
                        param_groups=fresh_pg)
    res['svhn_to_mnist'] = evaluate_model(fresh, mnist_test, device=dev)
    del fresh
    _cleanup_device()

    return res


def run_all():
    print(f"EPOCH SWEEP EXPERIMENT")
    print(f"{len(MODEL_SPECS)} models x {len(EPOCH_COUNTS)} epochs x {len(SEEDS)} seeds "
          f"= {len(MODEL_SPECS) * len(EPOCH_COUNTS) * len(SEEDS)} runs")
    print(f"Device: {DEVICE}, lr={LR}, batch_size={BATCH_SIZE}")
    print(f"Epoch counts: {EPOCH_COUNTS}, noise sigmas: {NOISE_SIGMAS}, seeds: {SEEDS}")
    print()

    print("Loading MNIST...")
    mnist_train, mnist_test = load_mnist(batch_size=BATCH_SIZE)
    print("Loading SVHN...")
    svhn_train, svhn_test = load_svhn(batch_size=BATCH_SIZE)

    checkpoint = _load_checkpoint()
    n_skipped = len(checkpoint)
    if n_skipped > 0:
        print(f"\nResuming: {n_skipped} runs already completed, skipping them.")

    total_runs = len(MODEL_SPECS) * len(EPOCH_COUNTS) * len(SEEDS)
    run_idx = 0
    t_start = time.time()

    for name, cls, recon_lam, pg_fn in MODEL_SPECS:
        print(f"\nModel: {name}" + (f"  (recon_lambda={recon_lam})" if recon_lam > 0 else ""))

        for epochs in EPOCH_COUNTS:
            print(f"\n  Epochs: {epochs}")
            for seed in SEEDS:
                run_idx += 1
                ck = _checkpoint_key(name, epochs, seed)
                if ck in checkpoint:
                    print(f"    Seed {seed} ({run_idx}/{total_runs})... SKIPPED (checkpoint)")
                    continue

                set_seed(seed)
                _cleanup_device()
                print(f"    Seed {seed} ({run_idx}/{total_runs})...", end=" ", flush=True)
                t0 = time.time()

                try:
                    res = run_single(cls, recon_lam, pg_fn, epochs,
                                     mnist_train, mnist_test, svhn_train, svhn_test)
                except RuntimeError as e:
                    if 'command buffer' in str(e) or 'MPS' in str(e):
                        print(f"\n      [MPS error, retrying on CPU]", flush=True)
                        _cleanup_device()
                        set_seed(seed)
                        res = run_single(cls, recon_lam, pg_fn, epochs,
                                         mnist_train, mnist_test, svhn_train, svhn_test,
                                         device='cpu')
                    else:
                        raise

                elapsed = time.time() - t0
                print(f"done in {elapsed:.0f}s -- "
                      f"clean={res['noise_0.0']:.1f}%, "
                      f"s=1.0={res['noise_1.0']:.1f}%, "
                      f"S->M={res['svhn_to_mnist']:.1f}%")

                checkpoint[ck] = res
                _save_checkpoint(checkpoint)

    total_time = time.time() - t_start

    # Reconstruct structured results
    all_results = {}
    for name, _, _, _ in MODEL_SPECS:
        all_results[name] = {}
        for epochs in EPOCH_COUNTS:
            all_results[name][str(epochs)] = {}
            for seed in SEEDS:
                ck = _checkpoint_key(name, epochs, seed)
                if ck in checkpoint:
                    all_results[name][str(epochs)][str(seed)] = checkpoint[ck]

    summary = {}
    for name, _, _, _ in MODEL_SPECS:
        summary[name] = {}
        for epochs in EPOCH_COUNTS:
            ep_key = str(epochs)
            seed_results = all_results[name][ep_key]
            if not seed_results:
                continue
            seeds_list = list(seed_results.values())
            agg = {}
            for sigma in NOISE_SIGMAS:
                sk = f'noise_{sigma}'
                vals = [s[sk] for s in seeds_list]
                m, s = mean_std(vals)
                agg[sk] = {'mean': m, 'std': s, 'per_seed': vals}
            vals = [s['svhn_to_mnist'] for s in seeds_list]
            m, s = mean_std(vals)
            agg['svhn_to_mnist'] = {'mean': m, 'std': s, 'per_seed': vals}
            summary[name][ep_key] = agg

    for metric_key, metric_label in [('noise_1.0', 'NOISE ROBUSTNESS (s=1.0)'),
                                      ('noise_0.0', 'CLEAN ACCURACY'),
                                      ('svhn_to_mnist', 'SVHN->MNIST TRANSFER')]:
        print(f"\n{metric_label} vs EPOCHS (mean +/- std)")
        header = f"{'Model':<26}"
        for ep in EPOCH_COUNTS:
            header += f" {'ep='+str(ep):>14}"
        print(header)
        print("-" * (26 + 15 * len(EPOCH_COUNTS)))
        for name, _, _, _ in MODEL_SPECS:
            row = f"{name:<26}"
            for ep in EPOCH_COUNTS:
                ep_key = str(ep)
                if ep_key in summary[name]:
                    m = summary[name][ep_key][metric_key]['mean']
                    s = summary[name][ep_key][metric_key]['std']
                    row += f" {fmt(m, s):>14}"
                else:
                    row += f" {'N/A':>14}"
            print(row)

    print(f"\nTotal experiment time: {total_time/60:.1f} minutes")

    os.makedirs(os.path.join(PROJECT_ROOT, 'data'), exist_ok=True)
    output = {
        'config': {
            'experiment': 'epoch_sweep',
            'models': [n for n, _, _, _ in MODEL_SPECS],
            'epoch_counts': EPOCH_COUNTS,
            'lr': LR, 'batch_size': BATCH_SIZE,
            'noise_sigmas': NOISE_SIGMAS, 'seeds': SEEDS,
            'device': DEVICE, 'total_time_minutes': total_time / 60,
            'seplr_config': SEPLR_CONFIG,
        },
        'summary': summary,
        'per_seed': all_results,
    }

    with open(RESULTS_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nFull results saved to {RESULTS_FILE}")

    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print(f"Checkpoint file removed (run complete).")


if __name__ == '__main__':
    run_all()
