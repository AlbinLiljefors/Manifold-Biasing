#!/usr/bin/env python3
# KTH action recognition experiment: 5 phases (noise, sample eff, RoL, transfer, scaling).
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.utils.data

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import get_kth_model_specs, count_params
from data import load_kth_pairs, load_weizmann_pairs
from train import train_model, evaluate_model


MODEL_SHORT_NAMES = {
    0: "ManifoldTKt",
    1: "ManifoldKt",
    2: "Baseline",
    3: "BaselineSingleFrame",
}

PHASE_A_SIGMAS = [0.0, 0.25, 0.5, 1.0]
PHASE_B_N_TRAIN = [200, 500, 1000, 2000, "full"]
PHASE_B_SIGMAS = [0.0, 0.5, 1.0]
TRANSFER_CLASSES = {0, 2, 4}  # walking, running, handwaving


class ClassSubset(torch.utils.data.Dataset):
    """Filter a dataset to samples whose labels lie in class_indices."""

    def __init__(self, dataset, class_indices):
        self.dataset = dataset
        self.indices = [i for i in range(len(dataset))
                        if dataset[i][1] in class_indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def parse_args():
    parser = argparse.ArgumentParser(description="KTH action recognition experiment")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seeds", type=str, default="42,123,456")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--data-dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'data'))
    return parser.parse_args()


def run_phase_a(specs, seeds, epochs, batch_size, data_dir, device):
    print("=" * 70)
    print("PHASE A: Action Classification + Noise Robustness")
    print("=" * 70)

    results = {}

    for idx, (orig_name, cls, recon_lambda, pg_fn) in enumerate(specs):
        short_name = MODEL_SHORT_NAMES[idx]
        results[short_name] = {"seeds": {}, "summary": {}, "first_layer_params": None}

        print(f"\n--- Model: {short_name} ---")

        for seed in seeds:
            print(f"\n  Seed {seed}:")
            torch.manual_seed(seed)
            model = cls(n_classes=6)
            trainable, frozen = count_params(model)
            total_params = trainable + frozen
            results[short_name]["first_layer_params"] = trainable
            print(f"    Params: {trainable} trainable, {frozen} frozen")

            train_loader, test_loader = load_kth_pairs(
                batch_size, data_dir, seed=seed)

            param_groups = pg_fn(model) if pg_fn else None
            t0 = time.time()
            model = train_model(
                model, train_loader, epochs=epochs, lr=1e-3,
                recon_lambda=recon_lambda, device=device, verbose=True,
                param_groups=param_groups,
            )
            train_time = time.time() - t0

            noise_acc = {}
            for sigma in PHASE_A_SIGMAS:
                acc = evaluate_model(model, test_loader, device=device,
                                     noise_sigma=sigma)
                noise_acc[str(sigma)] = round(acc, 2)
                print(f"    sigma={sigma}: {acc:.2f}%")

            results[short_name]["seeds"][str(seed)] = {
                "noise_acc": noise_acc,
                "first_layer_params": trainable,
                "total_params": total_params,
                "train_time": round(train_time, 1),
            }

        for sigma in PHASE_A_SIGMAS:
            values = [results[short_name]["seeds"][str(s)]["noise_acc"][str(sigma)]
                      for s in seeds]
            results[short_name]["summary"][str(sigma)] = {
                "mean": round(float(np.mean(values)), 2),
                "std": round(float(np.std(values)), 2),
                "values": values,
            }

    return results


def run_phase_b(specs, seeds, epochs, batch_size, data_dir, device):
    print("\n" + "=" * 70)
    print("PHASE B: Sample Efficiency")
    print("=" * 70)

    results = {}

    for idx, (orig_name, cls, recon_lambda, pg_fn) in enumerate(specs):
        short_name = MODEL_SHORT_NAMES[idx]
        results[short_name] = {}

        print(f"\n--- Model: {short_name} ---")

        for n_train in PHASE_B_N_TRAIN:
            n_key = str(n_train)
            results[short_name][n_key] = {"seeds": {}, "summary": {}}

            print(f"\n  n_train={n_train}:")

            for seed in seeds:
                print(f"    Seed {seed}:")
                torch.manual_seed(seed)
                model = cls(n_classes=6)

                n = None if n_train == "full" else n_train
                train_loader, test_loader = load_kth_pairs(
                    batch_size, data_dir, n_train=n, seed=seed)

                param_groups = pg_fn(model) if pg_fn else None
                t0 = time.time()
                model = train_model(
                    model, train_loader, epochs=epochs, lr=1e-3,
                    recon_lambda=recon_lambda, device=device, verbose=True,
                    param_groups=param_groups,
                )
                train_time = time.time() - t0

                noise_acc = {}
                for sigma in PHASE_B_SIGMAS:
                    acc = evaluate_model(model, test_loader, device=device,
                                         noise_sigma=sigma)
                    noise_acc[str(sigma)] = round(acc, 2)
                    print(f"      sigma={sigma}: {acc:.2f}%")

                results[short_name][n_key]["seeds"][str(seed)] = {
                    "noise_acc": noise_acc,
                    "train_time": round(train_time, 1),
                }

            for sigma in PHASE_B_SIGMAS:
                values = [
                    results[short_name][n_key]["seeds"][str(s)]["noise_acc"][str(sigma)]
                    for s in seeds
                ]
                results[short_name][n_key]["summary"][str(sigma)] = {
                    "mean": round(float(np.mean(values)), 2),
                    "std": round(float(np.std(values)), 2),
                    "values": values,
                }

    return results


def run_phase_c(specs, seeds, batch_size, data_dir, device):
    print("\n" + "=" * 70)
    print("PHASE C: Rate of Learning")
    print("=" * 70)

    results = {}

    for idx, (orig_name, cls, recon_lambda, pg_fn) in enumerate(specs):
        short_name = MODEL_SHORT_NAMES[idx]
        results[short_name] = {"seeds": {}}

        print(f"\n--- Model: {short_name} ---")

        for seed in seeds:
            print(f"\n  Seed {seed}:")
            torch.manual_seed(seed)
            model = cls(n_classes=6)

            train_loader, test_loader = load_kth_pairs(
                batch_size, data_dir, seed=seed)

            param_groups = pg_fn(model) if pg_fn else None
            model, learning_curve = train_model(
                model, train_loader, epochs=1, lr=1e-3,
                recon_lambda=recon_lambda, device=device, verbose=True,
                param_groups=param_groups,
                log_interval=20, eval_loader=test_loader,
            )

            curve = [[imgs, round(acc, 2)] for imgs, acc in learning_curve]
            results[short_name]["seeds"][str(seed)] = curve
            print(f"    Logged {len(curve)} learning curve points")

    return results


def run_phase_d(specs, seeds, epochs, batch_size, data_dir, device):
    print("\n" + "=" * 70)
    print("PHASE D: KTH -> Weizmann Transfer (3-class subset)")
    print("=" * 70)

    results = {}

    weizmann_loader = load_weizmann_pairs(batch_size, data_dir)

    for idx, (orig_name, cls, recon_lambda, pg_fn) in enumerate(specs):
        short_name = MODEL_SHORT_NAMES[idx]
        results[short_name] = {"seeds": {}, "summary": {}}

        print(f"\n--- Model: {short_name} ---")

        for seed in seeds:
            print(f"\n  Seed {seed}:")
            torch.manual_seed(seed)
            model = cls(n_classes=6)

            train_loader, test_loader = load_kth_pairs(
                batch_size, data_dir, seed=seed)

            train_subset = ClassSubset(train_loader.dataset, TRANSFER_CLASSES)
            test_subset = ClassSubset(test_loader.dataset, TRANSFER_CLASSES)

            train_subset_loader = torch.utils.data.DataLoader(
                train_subset, batch_size=batch_size, shuffle=True)
            test_subset_loader = torch.utils.data.DataLoader(
                test_subset, batch_size=batch_size, shuffle=False)

            print(f"    KTH 3-class: {len(train_subset)} train, {len(test_subset)} test")

            param_groups = pg_fn(model) if pg_fn else None
            model = train_model(
                model, train_subset_loader, epochs=epochs, lr=1e-3,
                recon_lambda=recon_lambda, device=device, verbose=True,
                param_groups=param_groups,
            )

            kth_acc = evaluate_model(model, test_subset_loader, device=device)
            weizmann_acc = evaluate_model(model, weizmann_loader, device=device)

            print(f"    KTH 3-class acc:  {kth_acc:.2f}%")
            print(f"    Weizmann acc:     {weizmann_acc:.2f}%")

            results[short_name]["seeds"][str(seed)] = {
                "kth_3class_acc": round(kth_acc, 2),
                "weizmann_acc": round(weizmann_acc, 2),
            }

        kth_values = [results[short_name]["seeds"][str(s)]["kth_3class_acc"]
                      for s in seeds]
        weiz_values = [results[short_name]["seeds"][str(s)]["weizmann_acc"]
                       for s in seeds]
        results[short_name]["summary"] = {
            "kth_3class": {
                "mean": round(float(np.mean(kth_values)), 2),
                "std": round(float(np.std(kth_values)), 2),
                "values": kth_values,
            },
            "weizmann": {
                "mean": round(float(np.mean(weiz_values)), 2),
                "std": round(float(np.std(weiz_values)), 2),
                "values": weiz_values,
            },
        }

    return results


def run_phase_e(specs):
    print("\n" + "=" * 70)
    print("PHASE E: Scaling Analysis")
    print("=" * 70)

    results = {}

    for idx, (orig_name, cls, recon_lambda, pg_fn) in enumerate(specs):
        short_name = MODEL_SHORT_NAMES[idx]
        model = cls(n_classes=6)
        trainable, frozen = count_params(model)

        results[short_name] = {
            "trainable": trainable,
            "frozen": frozen,
        }

        print(f"  {short_name}: {trainable} trainable, {frozen} frozen")

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

    specs = get_kth_model_specs()

    phase_a = run_phase_a(specs, seeds, args.epochs, args.batch_size,
                          data_dir, device)
    phase_b = run_phase_b(specs, seeds, args.epochs, args.batch_size,
                          data_dir, device)
    phase_c = run_phase_c(specs, seeds, args.batch_size, data_dir, device)
    phase_d = run_phase_d(specs, seeds, args.epochs, args.batch_size,
                          data_dir, device)
    phase_e = run_phase_e(specs)

    total_time = time.time() - total_t0

    output = {
        "phase_a": phase_a,
        "phase_b": phase_b,
        "phase_c": phase_c,
        "phase_d": phase_d,
        "phase_e": phase_e,
        "metadata": {
            "seeds": seeds,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "device": device,
            "total_time_seconds": round(total_time, 1),
        },
    }

    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, "kth_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"Total time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
