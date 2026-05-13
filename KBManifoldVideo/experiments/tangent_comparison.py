#!/usr/bin/env python3
# Tangent bundle lifting comparison: 6 variants on MNIST frame pairs.
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import get_model_specs, count_params
from data import load_moving_mnist, load_direction_mnist
from train import train_model, evaluate_model

SHORT_NAMES = {
    0: "TKt",
    1: "TKtRaw",
    2: "TKtScaled",
    3: "KtSpatialOnly",
    4: "Baseline",
}

NOISE_SIGMAS = [0.0, 0.5, 1.0]


def parse_args():
    parser = argparse.ArgumentParser(description="Tangent bundle lifting comparison")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seeds", type=str, default="42,123,456")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--data-dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'data'))
    return parser.parse_args()


def run_classification(task_name, load_fn, n_classes, specs, seeds, epochs,
                       batch_size, data_dir, device):
    print("=" * 70)
    print(f"PHASE: {task_name}")
    print("=" * 70)

    results = {}

    for idx, (orig_name, cls, recon_lambda, pg_fn) in enumerate(specs):
        short_name = SHORT_NAMES[idx]
        results[short_name] = {"seeds": {}, "summary": {}}

        print(f"\n--- Model: {short_name} ({orig_name}) ---")

        for seed in seeds:
            print(f"\n  Seed {seed}:")
            torch.manual_seed(seed)
            model = cls(n_classes=n_classes)
            trainable, frozen = count_params(model)
            print(f"    Params: {trainable} trainable, {frozen} frozen")

            train_loader, test_loader = load_fn(batch_size, data_dir, seed=seed)

            param_groups = pg_fn(model) if pg_fn else None
            t0 = time.time()
            model = train_model(
                model, train_loader, epochs=epochs, lr=1e-3,
                recon_lambda=recon_lambda, device=device, verbose=True,
                param_groups=param_groups,
            )
            train_time = time.time() - t0

            noise_acc = {}
            for sigma in NOISE_SIGMAS:
                acc = evaluate_model(model, test_loader, device=device,
                                     noise_sigma=sigma)
                noise_acc[str(sigma)] = round(acc, 2)
                print(f"    sigma={sigma}: {acc:.2f}%")

            results[short_name]["seeds"][str(seed)] = {
                "noise_acc": noise_acc,
                "first_layer_params": trainable,
                "train_time": round(train_time, 1),
            }

        for sigma in NOISE_SIGMAS:
            values = [results[short_name]["seeds"][str(s)]["noise_acc"][str(sigma)]
                      for s in seeds]
            results[short_name]["summary"][str(sigma)] = {
                "mean": round(float(np.mean(values)), 2),
                "std": round(float(np.std(values)), 2),
                "values": values,
            }

    return results


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    device = args.device
    data_dir = os.path.abspath(args.data_dir)

    print(f"Config: epochs={args.epochs}, seeds={seeds}, device={device}, "
          f"batch_size={args.batch_size}, data_dir={data_dir}")
    print()

    total_t0 = time.time()

    specs = get_model_specs()

    digit_results = run_classification(
        task_name="Digit Classification (10 classes)",
        load_fn=load_moving_mnist,
        n_classes=10,
        specs=specs,
        seeds=seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        data_dir=data_dir,
        device=device,
    )

    direction_results = run_classification(
        task_name="Direction Classification (4 classes)",
        load_fn=load_direction_mnist,
        n_classes=4,
        specs=specs,
        seeds=seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        data_dir=data_dir,
        device=device,
    )

    total_time = time.time() - total_t0

    output = {
        "digit_classification": digit_results,
        "direction_classification": direction_results,
        "metadata": {
            "seeds": seeds,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "device": device,
            "total_time_seconds": round(total_time, 1),
        },
    }

    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, "tangent_comparison_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"Total time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
