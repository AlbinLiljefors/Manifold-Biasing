# Love noise parameter sweep (tau, omega^2) matching Love et al. (JMLR 2023)
# Protocols: (A) Train noisy/test clean, (B) Train clean/test noisy.
import argparse
import json
import os
import time

import torch

from data import load_mnist
from train import train_model, evaluate_model
from noise import generate_class_noise_params
from models import (
    LoveNet, ManifoldFusedFrozenNet, ManifoldRawNet, BaselineNet,
    SEPLR_CONFIG, make_param_groups, count_params, get_model_specs,
)

TAU_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8]
OMEGA_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8]  # omega
TAU_FIXED = 0.2
OMEGA_SQ_FIXED = 0.04


def train_fresh(cls, recon_lam, train_loader, epochs, lr, device, pg_fn,
                noise_params=None):
    model = cls()
    pg = pg_fn(model) if pg_fn else None
    model = train_model(model, train_loader, epochs=epochs, lr=lr,
                        recon_lambda=recon_lam, device=device, verbose=False,
                        param_groups=pg, noise_params=noise_params)
    return model


def mean_std(values):
    m = sum(values) / len(values)
    s = (sum((v - m) ** 2 for v in values) / max(len(values) - 1, 1)) ** 0.5
    return round(m, 2), round(s, 2)


def run_sweep(args):
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    seeds = [int(s) for s in args.seeds.split(',')]

    print(f"Device: {device} | epochs={args.epochs} lr={args.lr}")
    print(f"Seeds: {seeds}")
    print(f"Tau sweep: {TAU_VALUES} (omega^2={OMEGA_SQ_FIXED})")
    print(f"Omega sweep: {OMEGA_VALUES} (tau={TAU_FIXED})")
    print()

    mnist_train, mnist_test = load_mnist(batch_size=args.batch_size)
    model_specs = get_model_specs()
    results = {}
    t_start = time.time()

    for name, cls, recon_lam, pg_fn in model_specs:
        results[name] = {
            'tau_sweep': {'protocol_a': {}, 'protocol_b': {}},
            'omega_sweep': {'protocol_a': {}, 'protocol_b': {}},
        }

        for seed in seeds:
            torch.manual_seed(seed)
            # Train one clean model for Protocol B (reused across all noise levels)
            print(f"  {name} seed={seed}: training clean model...", end=" ", flush=True)
            t0 = time.time()
            clean_model = train_fresh(cls, recon_lam, mnist_train, args.epochs,
                                      args.lr, device, pg_fn)
            print(f"{time.time()-t0:.0f}s")

            # --- Tau sweep (omega^2 fixed) ---
            for tau in TAU_VALUES:
                key = str(tau)
                noise = generate_class_noise_params(
                    tau=tau, omega_sq=OMEGA_SQ_FIXED, seed=seed)

                # Protocol B: train clean / test noisy (fast — just evaluate)
                acc_b = evaluate_model(clean_model, mnist_test, device=device,
                                       noise_params=noise)
                results[name]['tau_sweep']['protocol_b'].setdefault(key, []).append(acc_b)

                # Protocol A: train noisy / test clean (needs fresh model)
                torch.manual_seed(seed)
                noisy_model = train_fresh(cls, recon_lam, mnist_train, args.epochs,
                                          args.lr, device, pg_fn, noise_params=noise)
                acc_a = evaluate_model(noisy_model, mnist_test, device=device)
                results[name]['tau_sweep']['protocol_a'].setdefault(key, []).append(acc_a)

                print(f"    tau={tau}: A={acc_a:.1f}  B={acc_b:.1f}")

            # --- Omega sweep (tau fixed) ---
            for omega in OMEGA_VALUES:
                omega_sq = omega ** 2
                key = str(omega)
                noise = generate_class_noise_params(
                    tau=TAU_FIXED, omega_sq=omega_sq, seed=seed)

                # Protocol B
                acc_b = evaluate_model(clean_model, mnist_test, device=device,
                                       noise_params=noise)
                results[name]['omega_sweep']['protocol_b'].setdefault(key, []).append(acc_b)

                # Protocol A
                torch.manual_seed(seed)
                noisy_model = train_fresh(cls, recon_lam, mnist_train, args.epochs,
                                          args.lr, device, pg_fn, noise_params=noise)
                acc_a = evaluate_model(noisy_model, mnist_test, device=device)
                results[name]['omega_sweep']['protocol_a'].setdefault(key, []).append(acc_a)

                print(f"    omega={omega}: A={acc_a:.1f}  B={acc_b:.1f}")

    total_time = time.time() - t_start

    # Aggregate mean/std
    summary = {}
    for name in results:
        summary[name] = {}
        for sweep in ['tau_sweep', 'omega_sweep']:
            summary[name][sweep] = {}
            for protocol in ['protocol_a', 'protocol_b']:
                summary[name][sweep][protocol] = {}
                for key, vals in results[name][sweep][protocol].items():
                    m, s = mean_std(vals)
                    summary[name][sweep][protocol][key] = {
                        'mean': m, 'std': s, 'per_seed': vals
                    }

    # Print tables
    print(f"\n{'='*70}")
    print("TAU SWEEP (omega^2=0.04 fixed)")
    print(f"{'='*70}")
    for protocol, label in [('protocol_a', 'Train Noisy / Test Clean'),
                            ('protocol_b', 'Train Clean / Test Noisy')]:
        print(f"\n{label}:")
        header = f"{'Model':<20}" + "".join(f"{'t='+str(t):>10}" for t in TAU_VALUES)
        print(header)
        print("-" * len(header))
        for name in summary:
            row = f"{name:<20}"
            for tau in TAU_VALUES:
                d = summary[name]['tau_sweep'][protocol][str(tau)]
                row += f"{d['mean']:>6.1f}±{d['std']:<3.1f}"
            print(row)

    print(f"\n{'='*70}")
    print("OMEGA SWEEP (tau=0.2 fixed)")
    print(f"{'='*70}")
    for protocol, label in [('protocol_a', 'Train Noisy / Test Clean'),
                            ('protocol_b', 'Train Clean / Test Noisy')]:
        print(f"\n{label}:")
        header = f"{'Model':<20}" + "".join(f"{'w='+str(w):>10}" for w in OMEGA_VALUES)
        print(header)
        print("-" * len(header))
        for name in summary:
            row = f"{name:<20}"
            for omega in OMEGA_VALUES:
                d = summary[name]['omega_sweep'][protocol][str(omega)]
                row += f"{d['mean']:>6.1f}±{d['std']:<3.1f}"
            print(row)

    print(f"\nTotal: {total_time/60:.1f} min")

    # Save
    output = {
        'config': {
            'epochs': args.epochs, 'lr': args.lr, 'batch_size': args.batch_size,
            'seeds': seeds, 'device': device,
            'tau_values': TAU_VALUES, 'omega_values': OMEGA_VALUES,
            'tau_fixed': TAU_FIXED, 'omega_sq_fixed': OMEGA_SQ_FIXED,
            'total_time_minutes': round(total_time / 60, 1),
        },
        'summary': summary,
        'per_seed': results,
    }
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Love noise parameter sweep')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--seeds', default='42,123,456')
    parser.add_argument('--device', default=None)
    parser.add_argument('--output', default='data/love_sweep.json')
    run_sweep(parser.parse_args())
