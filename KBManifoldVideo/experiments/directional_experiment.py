#!/usr/bin/env python3
# Directional subset discrimination: SSv2 (14 classes) and Jester (22 classes).
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import get_kth_model_specs, count_params
from data import load_ssv2_direction_pairs, load_jester_direction_pairs
from train import train_model, evaluate_model


DIRECTIONAL_DATASETS = [
    ('ssv2_directional', load_ssv2_direction_pairs, 14),
    ('jester_directional', load_jester_direction_pairs, 22),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Directional subset discrimination experiment")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seeds", type=str, default="42,123,456")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--data-dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'data'))
    return parser.parse_args()


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    device = args.device
    data_dir = os.path.abspath(args.data_dir)

    print("=" * 70)
    print("DIRECTIONAL SUBSET DISCRIMINATION EXPERIMENT")
    print("4 KTH models x 2 directional datasets x {} seeds".format(len(seeds)))
    print("=" * 70)
    print(f"Config: epochs={args.epochs}, seeds={seeds}, device={device}, "
          f"batch_size={args.batch_size}, data_dir={data_dir}")
    print()

    specs = get_kth_model_specs()
    total_t0 = time.time()
    all_results = {}

    for ds_key, load_fn, n_classes in DIRECTIONAL_DATASETS:
        print("=" * 70)
        print(f"DATASET: {ds_key} ({n_classes} classes)")
        print("=" * 70)

        all_results[ds_key] = {"n_classes": n_classes}

        for name, cls, recon_lambda, pg_fn in specs:
            all_results[ds_key][name] = {"seeds": {}, "summary": {}}

            print(f"\n--- Model: {name} ---")

            seed_accs = []
            for seed in seeds:
                print(f"\n  Seed {seed}:")
                torch.manual_seed(seed)
                model = cls(n_classes=n_classes)
                trainable, frozen = count_params(model)
                print(f"    Params: {trainable} trainable, {frozen} frozen")

                train_loader, test_loader = load_fn(
                    args.batch_size, data_dir, seed=seed)

                param_groups = pg_fn(model) if pg_fn else None
                t0 = time.time()
                model = train_model(
                    model, train_loader, epochs=args.epochs, lr=1e-3,
                    recon_lambda=recon_lambda, device=device, verbose=True,
                    param_groups=param_groups,
                )
                train_time = time.time() - t0

                acc = evaluate_model(model, test_loader, device=device,
                                     noise_sigma=0.0)
                acc = round(acc, 2)
                print(f"    Clean accuracy: {acc}%")
                print(f"    Train time: {train_time:.1f}s")

                all_results[ds_key][name]["seeds"][str(seed)] = {
                    "acc": acc,
                    "train_time": round(train_time, 1),
                }
                seed_accs.append(acc)

                del model
                if device == 'mps':
                    torch.mps.synchronize()
                    torch.mps.empty_cache()

            all_results[ds_key][name]["summary"] = {
                "mean": round(float(np.mean(seed_accs)), 2),
                "std": round(float(np.std(seed_accs)), 2),
                "values": seed_accs,
            }

    total_time = time.time() - total_t0

    print("\n" + "=" * 70)
    print("SUMMARY: Clean Accuracy (mean +/- std)")
    print("=" * 70)
    model_names = [name for name, _, _, _ in specs]
    for ds_key, _, n_classes in DIRECTIONAL_DATASETS:
        print(f"\n  {ds_key} ({n_classes} classes):")
        print(f"  {'Model':<25} {'Accuracy':>15}")
        print(f"  {'-'*41}")
        for name in model_names:
            s = all_results[ds_key][name]["summary"]
            print(f"  {name:<25} {s['mean']:.1f}+/-{s['std']:.1f}".rjust(0))

    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} min)")

    output = {
        **all_results,
        "metadata": {
            "seeds": seeds,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "device": device,
            "total_time_seconds": round(total_time, 1),
        },
    }

    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, "directional_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
