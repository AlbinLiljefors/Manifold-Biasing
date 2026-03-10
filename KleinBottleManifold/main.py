import os
import argparse
import json
import time
import random

import numpy as np
import torch

from data import load_mnist, load_svhn
from train import train_model, evaluate_model
from models import get_model_specs, count_params, SEPLR_CONFIG

if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'
NOISE_SIGMAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]


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


def run_single(model_cls, recon_lam, pg_fn, mnist_train, mnist_test,
               svhn_train, svhn_test, epochs, lr, device=None):
    dev = device or DEVICE
    model = model_cls()
    param_groups = pg_fn(model) if pg_fn else None

    model = train_model(model, mnist_train, epochs=epochs, lr=lr,
                        recon_lambda=recon_lam, device=dev, verbose=False,
                        param_groups=param_groups)

    res = {}
    res['mnist_clean'] = evaluate_model(model, mnist_test, device=dev)

    noise = {}
    for sigma in NOISE_SIGMAS:
        noise[str(sigma)] = evaluate_model(model, mnist_test, device=dev,
                                           noise_sigma=sigma)
    res['noise'] = noise
    res['mnist_to_svhn'] = evaluate_model(model, svhn_test, device=dev)

    del model
    _cleanup_device()

    fresh = model_cls()
    fresh_pg = pg_fn(fresh) if pg_fn else None
    fresh = train_model(fresh, svhn_train, epochs=epochs, lr=lr,
                        recon_lambda=recon_lam, device=dev, verbose=False,
                        param_groups=fresh_pg)
    res['svhn_test'] = evaluate_model(fresh, svhn_test, device=dev)
    res['svhn_to_mnist'] = evaluate_model(fresh, mnist_test, device=dev)

    del fresh
    _cleanup_device()
    return res


def mean_std(values):
    m = sum(values) / len(values)
    if len(values) > 1:
        s = (sum((v - m) ** 2 for v in values) / (len(values) - 1)) ** 0.5
    else:
        s = 0.0
    return m, s


def fmt(m, s):
    return f"{m:.1f}+/-{s:.1f}"


def run_all(args):
    epochs = args.epochs
    seeds = args.seeds
    results_file = args.output
    lr = 1e-5
    batch_size = 100

    print(f"Device: {DEVICE}, epochs={epochs}, lr={lr}, batch_size={batch_size}")
    print(f"Seeds: {seeds}, noise sigmas: {NOISE_SIGMAS}")

    print("Loading MNIST...")
    mnist_train, mnist_test = load_mnist(batch_size=batch_size)
    print("Loading SVHN...")
    svhn_train, svhn_test = load_svhn(batch_size=batch_size)

    model_specs = get_model_specs()

    print(f"\n{'Model':<26} {'Trainable':>12} {'Frozen':>10}")
    print("-" * 50)
    for name, cls, _, _ in model_specs:
        m = cls()
        t, f = count_params(m)
        print(f"{name:<26} {t:>12,} {f:>10,}")
    print()

    all_results = {}
    total_runs = len(model_specs) * len(seeds)
    run_idx = 0
    t_start = time.time()

    for name, cls, recon_lam, pg_fn in model_specs:
        all_results[name] = {}
        print(f"\nModel: {name}" + (f"  (recon_lambda={recon_lam})" if recon_lam > 0 else ""))

        for seed in seeds:
            run_idx += 1
            set_seed(seed)
            _cleanup_device()
            print(f"  Seed {seed} ({run_idx}/{total_runs})...", end=" ", flush=True)
            t0 = time.time()
            try:
                res = run_single(cls, recon_lam, pg_fn, mnist_train, mnist_test,
                                 svhn_train, svhn_test, epochs, lr)
            except RuntimeError as e:
                if 'command buffer' in str(e) or 'MPS' in str(e):
                    print(f"\n    [MPS error, retrying on CPU]", flush=True)
                    _cleanup_device()
                    set_seed(seed)
                    res = run_single(cls, recon_lam, pg_fn, mnist_train, mnist_test,
                                     svhn_train, svhn_test, epochs, lr, device='cpu')
                else:
                    raise
            elapsed = time.time() - t0
            print(f"done in {elapsed:.0f}s -- "
                  f"clean={res['mnist_clean']:.1f}%, "
                  f"sigma=1.0={res['noise']['1.0']:.1f}%, "
                  f"M->S={res['mnist_to_svhn']:.1f}%, "
                  f"S->M={res['svhn_to_mnist']:.1f}%")
            all_results[name][seed] = res

    total_time = time.time() - t_start

    # Aggregated stats
    summary = {}
    for name, _, _, _ in model_specs:
        seed_results = all_results[name]
        seeds_list = list(seed_results.values())
        agg = {}
        vals = [s['mnist_clean'] for s in seeds_list]
        agg['mnist_clean'] = {'mean': mean_std(vals)[0], 'std': mean_std(vals)[1],
                              'per_seed': vals}
        agg['noise'] = {}
        for sigma in NOISE_SIGMAS:
            sk = str(sigma)
            vals = [s['noise'][sk] for s in seeds_list]
            agg['noise'][sk] = {'mean': mean_std(vals)[0], 'std': mean_std(vals)[1],
                                'per_seed': vals}
        for key in ['mnist_to_svhn', 'svhn_to_mnist', 'svhn_test']:
            vals = [s[key] for s in seeds_list]
            agg[key] = {'mean': mean_std(vals)[0], 'std': mean_std(vals)[1],
                        'per_seed': vals}
        summary[name] = agg

    model_names = [name for name, _, _, _ in model_specs]

    print(f"\nNOISE ROBUSTNESS (mean +/- std over {len(seeds)} seeds)")
    header = f"{'Model':<26}"
    for sigma in NOISE_SIGMAS:
        header += f" {'s='+str(sigma):>10}"
    print(header)
    print("-" * (26 + 11 * len(NOISE_SIGMAS)))
    for name in model_names:
        row = f"{name:<26}"
        for sigma in NOISE_SIGMAS:
            sk = str(sigma)
            m, s = summary[name]['noise'][sk]['mean'], summary[name]['noise'][sk]['std']
            row += f" {fmt(m,s):>10}"
        print(row)

    print(f"\nTRANSFER LEARNING (mean +/- std over {len(seeds)} seeds)")
    print(f"{'Model':<26} {'Clean MNIST':>12} {'MNIST->SVHN':>12} {'SVHN->MNIST':>12} {'SVHN test':>12}")
    print("-" * 76)
    for name in model_names:
        s = summary[name]
        row = f"{name:<26}"
        row += f" {fmt(s['mnist_clean']['mean'], s['mnist_clean']['std']):>12}"
        row += f" {fmt(s['mnist_to_svhn']['mean'], s['mnist_to_svhn']['std']):>12}"
        row += f" {fmt(s['svhn_to_mnist']['mean'], s['svhn_to_mnist']['std']):>12}"
        row += f" {fmt(s['svhn_test']['mean'], s['svhn_test']['std']):>12}"
        print(row)

    print(f"\nTotal experiment time: {total_time/60:.1f} minutes")

    os.makedirs('data', exist_ok=True)
    output = {
        'config': {
            'architecture': "Paper downstream (Conv->FC->Softmax)",
            'models': '4 Manifold novel + 1 baseline',
            'epochs': epochs,
            'lr': lr,
            'batch_size': batch_size,
            'seplr_config': SEPLR_CONFIG,
            'seeds': seeds,
            'noise_sigmas': NOISE_SIGMAS,
            'device': DEVICE,
            'total_time_minutes': total_time / 60,
        },
        'summary': summary,
        'per_seed': {name: {str(seed): res
                            for seed, res in seed_results.items()}
                     for name, seed_results in all_results.items()},
    }

    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nFull results saved to {results_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Klein bottle comparison suite')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    parser.add_argument('--output', type=str, default='data/comparison_results.json')
    run_all(parser.parse_args())
