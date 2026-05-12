import os
import argparse
import json
import time
import random

import numpy as np
import torch

from data import load_mnist, load_svhn, load_mnist_subset
from train import train_model, evaluate_model
from noise import generate_class_noise_params
from models import get_model_specs, count_params, SEPLR_CONFIG

NOISE_SIGMAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
SAMPLE_SIZES = [500, 1000, 5000]

# Love et al. (JMLR 2023) Section 4.1.2 default parameters
LOVE_NOISE_TAU = 0.2
LOVE_NOISE_OMEGA_SQ = 0.04


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cleanup_device(device):
    if device == 'mps':
        torch.mps.synchronize()
        torch.mps.empty_cache()
    elif device == 'cuda':
        torch.cuda.empty_cache()


def run_single(model_cls, recon_lam, pg_fn, mnist_train, mnist_test,
               svhn_train, svhn_test, epochs, lr, device, noise_seed=None):
    model = model_cls()
    param_groups = pg_fn(model) if pg_fn else None

    model = train_model(model, mnist_train, epochs=epochs, lr=lr,
                        recon_lambda=recon_lam, device=device, verbose=False,
                        param_groups=param_groups)

    res = {}
    res['mnist_clean'] = evaluate_model(model, mnist_test, device=device)

    noise = {}
    for sigma in NOISE_SIGMAS:
        noise[str(sigma)] = evaluate_model(model, mnist_test, device=device,
                                           noise_sigma=sigma)
    res['noise'] = noise
    res['mnist_to_svhn'] = evaluate_model(model, svhn_test, device=device)

    del_model_clean = model  # keep reference for Love Protocol B

    fresh = model_cls()
    fresh_pg = pg_fn(fresh) if pg_fn else None
    fresh = train_model(fresh, svhn_train, epochs=epochs, lr=lr,
                        recon_lambda=recon_lam, device=device, verbose=False,
                        param_groups=fresh_pg)
    res['svhn_test'] = evaluate_model(fresh, svhn_test, device=device)
    res['svhn_to_mnist'] = evaluate_model(fresh, mnist_test, device=device)

    del fresh
    _cleanup_device(device)

    # Love noise protocols (Section 4.1.2)
    love_noise = generate_class_noise_params(
        tau=LOVE_NOISE_TAU, omega_sq=LOVE_NOISE_OMEGA_SQ,
        seed=noise_seed if noise_seed is not None else 42)

    # Protocol A: train noisy / test clean
    model_a = model_cls()
    pg_a = pg_fn(model_a) if pg_fn else None
    model_a = train_model(model_a, mnist_train, epochs=epochs, lr=lr,
                          recon_lambda=recon_lam, device=device, verbose=False,
                          param_groups=pg_a, noise_params=love_noise)
    res['love_train_noisy_test_clean'] = evaluate_model(
        model_a, mnist_test, device=device)

    del model_a
    _cleanup_device(device)

    # Protocol B: train clean / test noisy (reuse the clean-trained model)
    res['love_train_clean_test_noisy'] = evaluate_model(
        del_model_clean, mnist_test, device=device, noise_params=love_noise)

    del del_model_clean
    _cleanup_device(device)

    return res


def run_rate_of_learning(model_cls, recon_lam, pg_fn, mnist_train, mnist_test,
                         epochs, lr, device, log_interval=50):
    """Record test accuracy as a function of images seen during training."""
    model = model_cls()
    param_groups = pg_fn(model) if pg_fn else None
    result = train_model(model, mnist_train, epochs=epochs, lr=lr,
                         recon_lambda=recon_lam, device=device, verbose=False,
                         param_groups=param_groups,
                         log_interval=log_interval, eval_loader=mnist_test)
    model, curve = result
    del model
    _cleanup_device(device)
    return curve  # list of (images_seen, accuracy)


def run_sample_efficiency(model_cls, recon_lam, pg_fn, mnist_test,
                          epochs, lr, device, batch_size, seed):
    """Train on balanced subsets and report test accuracy."""
    results = {}
    for n in SAMPLE_SIZES:
        subset_train, _ = load_mnist_subset(n, seed=seed, batch_size=batch_size)
        model = model_cls()
        param_groups = pg_fn(model) if pg_fn else None
        model = train_model(model, subset_train, epochs=epochs, lr=lr,
                            recon_lambda=recon_lam, device=device, verbose=False,
                            param_groups=param_groups)
        acc = evaluate_model(model, mnist_test, device=device)
        results[str(n)] = acc
        del model
        _cleanup_device(device)
    return results


def images_to_threshold(curve, threshold=95.0):
    """First images_seen where test accuracy crosses `threshold`, with linear
    interpolation between surrounding checkpoints. `curve` is a list of
    {'images_seen', 'accuracy'} dicts. Returns None if never crossed."""
    for i, pt in enumerate(curve):
        if pt['accuracy'] >= threshold:
            if i == 0:
                return pt['images_seen']
            prev = curve[i - 1]
            if pt['accuracy'] == prev['accuracy']:
                return prev['images_seen']
            frac = (threshold - prev['accuracy']) / (pt['accuracy'] - prev['accuracy'])
            return prev['images_seen'] + frac * (pt['images_seen'] - prev['images_seen'])
    return None


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
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = 'cuda'
        torch.backends.cudnn.benchmark = True
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    seeds = [int(s) for s in args.seeds.split(',')]
    lr = args.lr
    batch_size = args.batch_size

    print(f"Device: {device} | epochs={args.epochs} lr={lr} batch={batch_size}")
    print(f"Seeds: {seeds}\n")

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
        for seed in seeds:
            run_idx += 1
            set_seed(seed)
            _cleanup_device(device)

            print(f"[{run_idx}/{total_runs}] {name} seed={seed}...", end=" ", flush=True)
            t0 = time.time()
            try:
                res = run_single(cls, recon_lam, pg_fn, mnist_train, mnist_test,
                                 svhn_train, svhn_test, args.epochs, lr,
                                 device, noise_seed=seed)
            except RuntimeError as e:
                if 'command buffer' in str(e) or 'MPS' in str(e):
                    print(f"\n    [MPS error, retrying on CPU]", flush=True)
                    _cleanup_device(device)
                    set_seed(seed)
                    res = run_single(cls, recon_lam, pg_fn, mnist_train, mnist_test,
                                     svhn_train, svhn_test, args.epochs, lr,
                                     'cpu', noise_seed=seed)
                else:
                    raise
            dt = time.time() - t0
            all_results[name][seed] = res
            print(f"{dt:.0f}s  clean={res['mnist_clean']:.1f}  "
                  f"s1.0={res['noise']['1.0']:.1f}  "
                  f"M->S={res['mnist_to_svhn']:.1f}  S->M={res['svhn_to_mnist']:.1f}  "
                  f"LoveA={res['love_train_noisy_test_clean']:.1f}  "
                  f"LoveB={res['love_train_clean_test_noisy']:.1f}")

    # Aggregate
    summary = {}
    for name, _, _, _ in model_specs:
        seeds_data = list(all_results[name].values())
        agg = {}
        for key in ['mnist_clean', 'mnist_to_svhn', 'svhn_to_mnist', 'svhn_test',
                    'love_train_noisy_test_clean', 'love_train_clean_test_noisy']:
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
    model_names = [n for n, _, _, _ in model_specs]

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

    print(f"\nLove noise protocols (tau={LOVE_NOISE_TAU}, omega^2={LOVE_NOISE_OMEGA_SQ}):")
    print(f"{'Model':<26} {'TrainNoisy':>12} {'TestNoisy':>12}")
    print("-" * 52)
    for name in model_names:
        s = summary[name]
        print(f"{name:<26} "
              f"{fmt(s['love_train_noisy_test_clean']['mean'], s['love_train_noisy_test_clean']['std']):>12} "
              f"{fmt(s['love_train_clean_test_noisy']['mean'], s['love_train_clean_test_noisy']['std']):>12}")

    # Rate of learning (all seeds, log every 20 batches)
    print(f"\nRATE OF LEARNING ({len(seeds)} seeds, log every 20 batches)")
    rol_results = {}
    for name, cls, recon_lam, pg_fn in model_specs:
        per_seed_curves = {}
        for seed in seeds:
            print(f"  RoL: {name} seed={seed}...", end=" ", flush=True)
            set_seed(seed)
            _cleanup_device(device)
            curve = run_rate_of_learning(cls, recon_lam, pg_fn, mnist_train,
                                         mnist_test, args.epochs, lr, device,
                                         log_interval=20)
            per_seed_curves[seed] = curve
            print(f"{len(curve)} points, final={curve[-1][1]:.1f}%")
        # Aggregate: seeds share the same images_seen schedule, so mean per checkpoint
        ref = per_seed_curves[seeds[0]]
        mean_curve = []
        for i, (imgs, _) in enumerate(ref):
            accs = [per_seed_curves[s][i][1] for s in seeds]
            mean_curve.append({'images_seen': imgs,
                               'accuracy': sum(accs) / len(accs)})
        # Convergence: images-to-threshold derived from mean curve + per seed
        images_to_95_per_seed = {}
        for s in seeds:
            per_seed_dicts = [{'images_seen': imgs, 'accuracy': acc}
                              for imgs, acc in per_seed_curves[s]]
            images_to_95_per_seed[str(s)] = images_to_threshold(
                per_seed_dicts, 95.0)
        rol_results[name] = {
            'curve_mean': mean_curve,
            'images_to_95_from_mean': images_to_threshold(mean_curve, 95.0),
            'images_to_95_per_seed': images_to_95_per_seed,
            'per_seed': {str(s): [{'images_seen': imgs, 'accuracy': acc}
                                   for imgs, acc in per_seed_curves[s]]
                         for s in seeds},
        }
        i95 = rol_results[name]['images_to_95_from_mean']
        print(f"    images-to-95 (mean curve): "
              f"{'NEVER' if i95 is None else f'{i95:,.0f}'}")
    summary['rate_of_learning'] = rol_results

    # Sample efficiency (all seeds)
    print(f"\nSAMPLE EFFICIENCY (subsets: {SAMPLE_SIZES})")
    se_results = {}
    for name, cls, recon_lam, pg_fn in model_specs:
        se_results[name] = {}
        for seed in seeds:
            set_seed(seed)
            _cleanup_device(device)
            print(f"  SE: {name} seed={seed}...", end=" ", flush=True)
            se = run_sample_efficiency(cls, recon_lam, pg_fn, mnist_test,
                                       args.epochs, lr, device, batch_size, seed)
            se_results[name][seed] = se
            print(f"  {' '.join(f'n={k}: {v:.1f}' for k, v in se.items())}")
    # Aggregate sample efficiency
    se_summary = {}
    for name in model_names:
        se_summary[name] = {}
        for n in SAMPLE_SIZES:
            vals = [se_results[name][s][str(n)] for s in seeds]
            m, s = mean_std(vals)
            se_summary[name][str(n)] = {'mean': m, 'std': s, 'per_seed': vals}
    summary['sample_efficiency'] = se_summary

    print(f"\nSAMPLE EFFICIENCY SUMMARY (mean +/- std)")
    header = f"{'Model':<26}"
    for n in SAMPLE_SIZES:
        header += f" {'n='+str(n):>12}"
    header += f" {'full':>12}"
    print(header)
    print("-" * (26 + 13 * (len(SAMPLE_SIZES) + 1)))
    for name in model_names:
        row = f"{name:<26}"
        for n in SAMPLE_SIZES:
            m, s = se_summary[name][str(n)]['mean'], se_summary[name][str(n)]['std']
            row += f" {fmt(m,s):>12}"
        row += f" {fmt(summary[name]['mnist_clean']['mean'], summary[name]['mnist_clean']['std']):>12}"
        print(row)

    total_time = time.time() - t_start
    print(f"\nTotal: {total_time/60:.1f} min")

    # Save
    output = {
        'config': {
            'architecture': "Paper downstream (Conv->FC->Softmax)",
            'models': '4 Manifold novel + 1 baseline',
            'epochs': args.epochs,
            'lr': lr,
            'batch_size': batch_size,
            'seplr_config': SEPLR_CONFIG,
            'seeds': seeds,
            'noise_sigmas': NOISE_SIGMAS,
            'love_noise': {'tau': LOVE_NOISE_TAU, 'omega_sq': LOVE_NOISE_OMEGA_SQ},
            'device': device,
            'total_time_minutes': total_time / 60,
        },
        'summary': summary,
        'per_seed': {name: {str(seed): res for seed, res in sdata.items()}
                     for name, sdata in all_results.items()},
    }

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Klein bottle comparison suite')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--seeds', default='42,123,456')
    parser.add_argument('--device', default=None)
    parser.add_argument('--output', default='data/comparison_results.json')
    run_all(parser.parse_args())
