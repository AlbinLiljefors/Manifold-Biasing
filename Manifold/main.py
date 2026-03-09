import argparse
import json
import time
import random

import numpy as np
import torch

from data import load_mnist, load_svhn
from train import train_model, evaluate_model
from models import (
    LoveNet, ManifoldFrozenNet, ManifoldRawNet, BaselineNet,
    MANIFOLD_RAW_SEPLR, make_param_groups, count_params,
)

NOISE_SIGMAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_model_specs():
    """Returns [(name, cls, recon_lambda, param_groups_fn or None)]."""
    return [
        ('Love', LoveNet, 0.0, None),
        ('ManifoldFrozen', ManifoldFrozenNet, 0.0, None),
        ('ManifoldRaw', ManifoldRawNet, MANIFOLD_RAW_SEPLR['recon_lambda'],
         lambda m: make_param_groups(m, MANIFOLD_RAW_SEPLR)),
        ('Baseline', BaselineNet, 0.0, None),
    ]


def run_single(model_cls, recon_lam, mnist_train, mnist_test, svhn_train, svhn_test,
               epochs, lr, device, param_groups_fn=None):
    model = model_cls()
    param_groups = param_groups_fn(model) if param_groups_fn else None
    res = {}

    model = train_model(model, mnist_train, epochs=epochs, lr=lr,
                        recon_lambda=recon_lam, device=device, verbose=False,
                        param_groups=param_groups)

    res['mnist_clean'] = evaluate_model(model, mnist_test, device=device)
    res['noise'] = {str(s): evaluate_model(model, mnist_test, device=device, noise_sigma=s)
                    for s in NOISE_SIGMAS}
    res['mnist_to_svhn'] = evaluate_model(model, svhn_test, device=device)

    # SVHN->MNIST transfer (fresh model)
    fresh = model_cls()
    fresh_pg = param_groups_fn(fresh) if param_groups_fn else None
    fresh = train_model(fresh, svhn_train, epochs=epochs, lr=lr,
                        recon_lambda=recon_lam, device=device, verbose=False,
                        param_groups=fresh_pg)
    res['svhn_test'] = evaluate_model(fresh, svhn_test, device=device)
    res['svhn_to_mnist'] = evaluate_model(fresh, mnist_test, device=device)

    return res


def mean_std(values):
    m = sum(values) / len(values)
    s = (sum((v - m) ** 2 for v in values) / max(len(values) - 1, 1)) ** 0.5
    return m, s


def run_all(args):
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    seeds = [int(s) for s in args.seeds.split(',')]

    print(f"Device: {device} | epochs={args.epochs} lr={args.lr} batch={args.batch_size}")
    print(f"Seeds: {seeds}\n")

    mnist_train, mnist_test = load_mnist(batch_size=args.batch_size)
    svhn_train, svhn_test = load_svhn(batch_size=args.batch_size)

    model_specs = get_model_specs()

    # Parameter counts
    for name, cls, _, _ in model_specs:
        t, f = count_params(cls())
        print(f"  {name:<20} trainable={t:,}  frozen={f:,}")
    print()

    all_results = {}
    total_runs = len(model_specs) * len(seeds)
    run_idx = 0
    t_start = time.time()

    for name, cls, recon_lam, pg_fn in model_specs:
        all_results[name] = {}
        for seed in seeds:
            run_idx += 1
            set_seed(seed)
            print(f"[{run_idx}/{total_runs}] {name} seed={seed}...", end=" ", flush=True)
            t0 = time.time()
            res = run_single(cls, recon_lam, mnist_train, mnist_test,
                             svhn_train, svhn_test, args.epochs, args.lr,
                             device, pg_fn)
            dt = time.time() - t0
            all_results[name][seed] = res
            print(f"{dt:.0f}s  clean={res['mnist_clean']:.1f}  "
                  f"s1.0={res['noise']['1.0']:.1f}  "
                  f"M->S={res['mnist_to_svhn']:.1f}  S->M={res['svhn_to_mnist']:.1f}")

    total_time = time.time() - t_start

    # Aggregate
    summary = {}
    for name, _, _, _ in model_specs:
        seeds_data = list(all_results[name].values())
        agg = {}
        for key in ['mnist_clean', 'mnist_to_svhn', 'svhn_to_mnist', 'svhn_test']:
            vals = [s[key] for s in seeds_data]
            m, s = mean_std(vals)
            agg[key] = {'mean': m, 'std': s, 'per_seed': vals}
        agg['noise'] = {}
        for sigma in NOISE_SIGMAS:
            vals = [s['noise'][str(sigma)] for s in seeds_data]
            m, s = mean_std(vals)
            agg['noise'][str(sigma)] = {'mean': m, 'std': s, 'per_seed': vals}
        summary[name] = agg

    # Print results table
    names = [n for n, _, _, _ in model_specs]
    fmt = lambda m, s: f"{m:.1f}+/-{s:.1f}"

    print(f"\n{'Model':<20} {'Clean':>10} {'s=0.5':>10} {'s=1.0':>10} {'M->S':>10} {'S->M':>10}")
    print("-" * 72)
    for n in names:
        s = summary[n]
        print(f"{n:<20} "
              f"{fmt(s['mnist_clean']['mean'], s['mnist_clean']['std']):>10} "
              f"{fmt(s['noise']['0.5']['mean'], s['noise']['0.5']['std']):>10} "
              f"{fmt(s['noise']['1.0']['mean'], s['noise']['1.0']['std']):>10} "
              f"{fmt(s['mnist_to_svhn']['mean'], s['mnist_to_svhn']['std']):>10} "
              f"{fmt(s['svhn_to_mnist']['mean'], s['svhn_to_mnist']['std']):>10}")

    print(f"\nTotal: {total_time/60:.1f} min")

    # Save
    output = {
        'config': {
            'epochs': args.epochs, 'lr': args.lr, 'batch_size': args.batch_size,
            'seeds': seeds, 'noise_sigmas': NOISE_SIGMAS, 'device': device,
            'total_time_minutes': total_time / 60,
        },
        'summary': summary,
        'per_seed': {name: {str(seed): res for seed, res in sdata.items()}
                     for name, sdata in all_results.items()},
    }
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Topological CNN experiments')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--seeds', default='42,123,456')
    parser.add_argument('--device', default=None)
    parser.add_argument('--output', default='results.json')
    run_all(parser.parse_args())
