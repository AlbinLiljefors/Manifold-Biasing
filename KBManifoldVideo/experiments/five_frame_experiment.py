#!/usr/bin/env python3
# 5-frame KTH experiment (90x60) with 8-layer ResNet downstream.
# Phases: A (clean+noise), B (sample efficiency), C (scaling), RoL, D (Weizmann transfer).
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import get_kth_5frame_resnet_model_specs, count_params
from data import load_kth_5frame_pairs, load_weizmann_5frame_pairs
from train import train_model, evaluate_model


NOISE_SIGMAS = [0.0, 1.0]
SAMPLE_SIZES = [500, 1000, 5000, "full"]

LOVE_SCALING = {
    "Love_5sub": {"frozen": 12100, "filters": 20},
    "Love_full": {"frozen": 784080, "filters": 1296},
}


def images_to_threshold(curve, threshold=95.0):
    """First images_seen where accuracy crosses threshold (linear interp)."""
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


def _cleanup(device):
    if device == 'mps':
        torch.mps.synchronize()
        torch.mps.empty_cache()
    elif device == 'cuda':
        torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(
        description="5-frame T(K^t) extension experiment on KTH")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seeds", type=str, default="42,123,456")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Base lr for trunk / default group.")
    parser.add_argument("--lr-schedule", type=str, default="cosine",
                        choices=["cosine", "none"])
    parser.add_argument("--rol-log-interval", type=int, default=200,
                        help="Log test accuracy every N batches during Phase A.")
    parser.add_argument("--data-dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'data'))
    parser.add_argument("--output", type=str, default="five_frame_200ep_results.json")
    parser.add_argument("--skip-phase-b", action="store_true",
                        help="Skip Phase B sample-efficiency sweep (expensive). "
                             "Useful when re-running only Phase A+RoL+D.")
    parser.add_argument("--merge-from", type=str, default=None,
                        help="Path to a prior results JSON to merge Phase B from "
                             "(used with --skip-phase-b).")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Phase A + RoL: one training pass per (model, seed) with inline RoL logging,
# final-state dicts stashed for Phase D.
# ---------------------------------------------------------------------------

def run_phase_a_and_rol(specs, seeds, epochs, batch_size, data_dir, device,
                         lr, lr_schedule, rol_log_interval):
    print("=" * 70)
    print("PHASE A + RoL: Clean accuracy, noise robustness, learning curves")
    print("=" * 70)

    results_a = {}
    results_rol = {}
    kept_models = {}

    for name, cls, recon_lambda, pg_fn in specs:
        results_a[name] = {"seeds": {}, "summary": {}}
        per_seed_curves = {}
        kept_models[name] = {}

        print(f"\n--- Model: {name} ---")

        for seed in seeds:
            print(f"\n  Seed {seed}:")
            torch.manual_seed(seed)
            model = cls(n_classes=6)
            trainable, frozen = count_params(model)
            print(f"    Params: {trainable} trainable, {frozen} frozen")

            train_loader, test_loader = load_kth_5frame_pairs(
                batch_size, data_dir, seed=seed)

            param_groups = pg_fn(model) if pg_fn else None
            t0 = time.time()
            model, curve = train_model(
                model, train_loader, epochs=epochs, lr=lr,
                recon_lambda=recon_lambda, device=device, verbose=True,
                param_groups=param_groups,
                log_interval=rol_log_interval, eval_loader=test_loader,
                lr_schedule=(None if lr_schedule == 'none' else lr_schedule),
            )
            train_time = time.time() - t0

            per_seed_curves[seed] = curve
            kept_models[name][seed] = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

            noise_acc = {}
            for sigma in NOISE_SIGMAS:
                acc = evaluate_model(model, test_loader, device=device,
                                     noise_sigma=sigma)
                acc = round(acc, 2)
                noise_acc[str(sigma)] = acc
                print(f"    sigma={sigma}: {acc}%")

            results_a[name]["seeds"][str(seed)] = {
                "noise_acc": noise_acc,
                "first_layer_params": trainable,
                "total_params": trainable + frozen,
                "train_time": round(train_time, 1),
            }

            del model
            _cleanup(device)

        for sigma in NOISE_SIGMAS:
            values = [results_a[name]["seeds"][str(s)]["noise_acc"][str(sigma)]
                      for s in seeds]
            results_a[name]["summary"][str(sigma)] = {
                "mean": round(float(np.mean(values)), 2),
                "std": round(float(np.std(values)), 2),
                "values": values,
            }

        ref = per_seed_curves[seeds[0]]
        mean_curve = []
        for i, (imgs, _) in enumerate(ref):
            accs = [per_seed_curves[s][i][1] for s in seeds]
            mean_curve.append({
                'images_seen': imgs,
                'accuracy': round(float(np.mean(accs)), 3),
            })

        images_to_95_per_seed = {}
        for s in seeds:
            per_seed_dicts = [{'images_seen': imgs, 'accuracy': acc}
                              for imgs, acc in per_seed_curves[s]]
            images_to_95_per_seed[str(s)] = images_to_threshold(per_seed_dicts, 95.0)

        results_rol[name] = {
            'curve_mean': mean_curve,
            'images_to_95_from_mean': images_to_threshold(mean_curve, 95.0),
            'images_to_95_per_seed': images_to_95_per_seed,
            'per_seed': {str(s): [{'images_seen': imgs, 'accuracy': acc}
                                   for imgs, acc in per_seed_curves[s]]
                         for s in seeds},
        }
        i95 = results_rol[name]['images_to_95_from_mean']
        print(f"    images-to-95 (mean curve): "
              f"{'NEVER' if i95 is None else f'{i95:,.0f}'}")

    return results_a, results_rol, kept_models


def run_phase_b(specs, seeds, epochs, batch_size, data_dir, device,
                lr, lr_schedule):
    print("\n" + "=" * 70)
    print("PHASE B: Sample Efficiency")
    print("=" * 70)

    results = {}

    for name, cls, recon_lambda, pg_fn in specs:
        results[name] = {}
        print(f"\n--- Model: {name} ---")

        for n_train in SAMPLE_SIZES:
            n_key = str(n_train)
            results[name][n_key] = {"seeds": {}, "summary": {}}
            print(f"\n  n_train={n_key}:")

            seed_accs = []
            for seed in seeds:
                print(f"    Seed {seed}:")
                torch.manual_seed(seed)
                model = cls(n_classes=6)

                n_arg = None if n_train == "full" else n_train
                train_loader, test_loader = load_kth_5frame_pairs(
                    batch_size, data_dir, n_train=n_arg, seed=seed)

                param_groups = pg_fn(model) if pg_fn else None
                t0 = time.time()
                model = train_model(
                    model, train_loader, epochs=epochs, lr=lr,
                    recon_lambda=recon_lambda, device=device, verbose=False,
                    param_groups=param_groups,
                    lr_schedule=(None if lr_schedule == 'none' else lr_schedule),
                )
                train_time = time.time() - t0

                acc = evaluate_model(model, test_loader, device=device,
                                     noise_sigma=0.0)
                acc = round(acc, 2)
                print(f"      Clean accuracy: {acc}%  ({train_time:.0f}s)")

                results[name][n_key]["seeds"][str(seed)] = {
                    "acc": acc,
                    "train_time": round(train_time, 1),
                }
                seed_accs.append(acc)

                del model
                _cleanup(device)

            results[name][n_key]["summary"] = {
                "mean": round(float(np.mean(seed_accs)), 2),
                "std": round(float(np.std(seed_accs)), 2),
                "values": seed_accs,
            }

    return results


def run_phase_c(specs):
    print("\n" + "=" * 70)
    print("PHASE C: Scaling Comparison (Analytical)")
    print("=" * 70)

    results = {}

    for name, cls, _, _ in specs:
        model = cls(n_classes=6)
        trainable, frozen = count_params(model)

        if 'TKt' in name:
            n_filters = 30
        elif 'Baseline5Frame' in name or 'SingleFrame' in name:
            n_filters = 64
        else:
            n_filters = 0

        results[name] = {
            "trainable": trainable,
            "frozen": frozen,
            "filters": n_filters,
        }
        print(f"  {name}: {trainable} trainable, {frozen} frozen, {n_filters} filters")

    for love_name, love_data in LOVE_SCALING.items():
        results[love_name] = love_data
        print(f"  {love_name}: {love_data['frozen']} frozen, "
              f"{love_data['filters']} filters")

    return results


def run_phase_d(specs, seeds, kept_models, batch_size, data_dir, device):
    print("\n" + "=" * 70)
    print("PHASE D: KTH -> Weizmann Transfer (zero-shot)")
    print("=" * 70)

    try:
        weizmann_loader = load_weizmann_5frame_pairs(batch_size=batch_size,
                                                       data_dir=data_dir)
    except FileNotFoundError as e:
        print(f"  [SKIP] Weizmann 5-frame data not found: {e}")
        return {name: {"skipped": True} for name, _, _, _ in specs}

    results = {}
    for name, cls, _, _ in specs:
        results[name] = {"seeds": {}, "summary": {}}
        print(f"\n--- Model: {name} ---")

        seed_accs = []
        for seed in seeds:
            if seed not in kept_models.get(name, {}):
                print(f"    [skip] no kept model for {name} seed {seed}")
                continue
            model = cls(n_classes=6)
            model.load_state_dict(kept_models[name][seed])
            model.to(device)

            acc = evaluate_model(model, weizmann_loader, device=device,
                                 noise_sigma=0.0)
            acc = round(acc, 2)
            print(f"    Seed {seed}: Weizmann acc = {acc}%")
            results[name]["seeds"][str(seed)] = acc
            seed_accs.append(acc)

            del model
            _cleanup(device)

        if seed_accs:
            results[name]["summary"] = {
                "mean": round(float(np.mean(seed_accs)), 2),
                "std": round(float(np.std(seed_accs)), 2),
                "values": seed_accs,
            }

    return results


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    device = args.device
    data_dir = os.path.abspath(args.data_dir)

    specs = get_kth_5frame_resnet_model_specs()

    print("=" * 70)
    print("FIVE-FRAME EXPERIMENT (KTH, 90x60)")
    print(f"Backbone: resnet8 | 3 models x {len(seeds)} seeds")
    print("=" * 70)
    print(f"Config: epochs={args.epochs}, lr={args.lr}, schedule={args.lr_schedule}, "
          f"batch={args.batch_size}, device={device}")
    print(f"Seeds: {seeds}  |  Output: {args.output}")
    print()

    total_t0 = time.time()

    phase_a, phase_rol, kept = run_phase_a_and_rol(
        specs, seeds, args.epochs, args.batch_size, data_dir, device,
        args.lr, args.lr_schedule, args.rol_log_interval)

    if args.skip_phase_b:
        print("\n[SKIPPED] Phase B — reusing prior results")
        phase_b = {}
        if args.merge_from and os.path.exists(args.merge_from):
            with open(args.merge_from) as f:
                prior = json.load(f)
            phase_b = prior.get("phase_b", {})
            print(f"  Loaded Phase B from {args.merge_from} ({len(phase_b)} models)")
    else:
        phase_b = run_phase_b(specs, seeds, args.epochs, args.batch_size,
                              data_dir, device, args.lr, args.lr_schedule)

    phase_c = run_phase_c(specs)

    phase_d = run_phase_d(specs, seeds, kept, args.batch_size, data_dir, device)

    total_time = time.time() - total_t0

    model_names = [name for name, _, _, _ in specs]

    print("\n" + "=" * 70)
    print("SUMMARY: Phase A -- Clean + Noise")
    print("=" * 70)
    print(f"{'Model':<25}" + ' '.join(f'σ={s:>4}' for s in NOISE_SIGMAS))
    for name in model_names:
        row = f"{name:<25}"
        for sigma in NOISE_SIGMAS:
            s = phase_a[name]["summary"][str(sigma)]
            row += f" {s['mean']:5.1f}±{s['std']:.1f} "
        print(row)

    print("\nPhase RoL images_to_95 (mean curve):")
    for name in model_names:
        i95 = phase_rol[name]['images_to_95_from_mean']
        print(f"  {name:<25} {'NEVER' if i95 is None else f'{i95:>10,.0f}'}")

    if phase_b:
        print("\n" + "=" * 70)
        print("SUMMARY: Phase B -- Sample Efficiency (clean acc)")
        print("=" * 70)
        print(f"{'Model':<25}" + ' '.join(f'n={n:>6}' for n in SAMPLE_SIZES))
        for name in model_names:
            if name not in phase_b:
                continue
            row = f"{name:<25}"
            for n in SAMPLE_SIZES:
                s = phase_b[name][str(n)]["summary"]
                row += f" {s['mean']:5.1f}±{s['std']:3.1f}"
            print(row)

    print("\n" + "=" * 70)
    print("SUMMARY: Phase D -- KTH→Weizmann Transfer")
    print("=" * 70)
    for name in model_names:
        if 'summary' in phase_d.get(name, {}) and phase_d[name]['summary']:
            s = phase_d[name]['summary']
            print(f"  {name:<25} {s['mean']:5.1f}±{s['std']:.1f}")
        else:
            print(f"  {name:<25} [skipped]")

    print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f} min, "
          f"{total_time/3600:.2f}h)")

    output = {
        "config": {
            "backbone": "resnet8",
            "epochs": args.epochs,
            "lr": args.lr,
            "lr_schedule": args.lr_schedule,
            "rol_log_interval": args.rol_log_interval,
            "batch_size": args.batch_size,
            "seeds": seeds,
            "device": device,
            "noise_sigmas": NOISE_SIGMAS,
            "sample_sizes": SAMPLE_SIZES,
            "total_time_seconds": round(total_time, 1),
        },
        "phase_a": phase_a,
        "phase_b": phase_b,
        "phase_c": phase_c,
        "phase_rol": phase_rol,
        "phase_d": phase_d,
    }

    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, args.output)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
