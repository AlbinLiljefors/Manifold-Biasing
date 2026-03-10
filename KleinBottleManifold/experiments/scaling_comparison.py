import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')

import json
import time

import numpy as np
import torch

from analytic_filters import klein_bottle_legendre_filters_nonuniform
from models import get_model_specs
from metrics import count_first_layer_params


def _time_filter_generation(func, *args, n_trials=5):
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        filters = func(*args)
        times.append((time.perf_counter() - t0) * 1000)
    mean_ms = sum(times) / len(times)
    if isinstance(filters, np.ndarray):
        mem = filters.nbytes
    elif isinstance(filters, torch.Tensor):
        mem = filters.nelement() * filters.element_size()
    else:
        mem = 0
    return filters, mean_ms, mem


def run_all():
    print("SCALING COMPARISON — Manifold Dimensionality vs Filter/Channel Count")
    print()

    t_start = time.time()
    results = {}

    # 1. Theoretical scaling table
    print("1. THEORETICAL SCALING TABLE")
    scaling_table = [
        {'manifold': 'S^1 (circle)', 'dimension': 1, 'discretized_default': 64, 'algebraic_channels': 2},
        {'manifold': 'K (Klein bottle)', 'dimension': 2, 'discretized_default': 64, 'algebraic_channels': 5},
        {'manifold': 'K^t (translational Klein)', 'dimension': 3, 'discretized_default': 256, 'algebraic_channels': 6},
        {'manifold': 'T(K^t) (video tangent bundle)', 'dimension': 6, 'discretized_default': 1296, 'algebraic_channels': 12},
    ]

    print(f"\n{'Manifold':<28} {'Dim':>4} {'Discretized':>14} {'Algebraic ch':>13} {'Ratio':>8}")
    print("-" * 70)
    for entry in scaling_table:
        ratio = entry['discretized_default'] / entry['algebraic_channels']
        print(f"{entry['manifold']:<28} {entry['dimension']:>4} "
              f"{entry['discretized_default']:>14} {entry['algebraic_channels']:>13} "
              f"{ratio:>7.1f}x")
    results['scaling_table'] = scaling_table

    # 2. Empirical filter generation
    print(f"\n2. EMPIRICAL FILTER GENERATION — Timing and Memory")

    print("\n  Algebraic Klein non-uniform (klein_bottle_legendre_filters_nonuniform):")
    for n1, n2 in [(4, 4), (8, 8), (16, 16)]:
        filters, t_ms, mem = _time_filter_generation(
            klein_bottle_legendre_filters_nonuniform, n1, n2, 3)
        print(f"    {n1}x{n2}={n1*n2:>4}: {filters.shape[0]} filters, {t_ms:.2f}ms, {mem/1024:.1f}KB")

    # 3. Parameter counts
    print(f"\n3. PARAMETER COUNT — All 5 Models")
    model_specs = get_model_specs()
    print(f"\n{'Model':<26} {'1st learn':>10} {'1st frozen':>11} {'1st total':>10} "
          f"{'Model total':>12} {'1st/total':>10}")
    print("-" * 82)
    for name, cls, _, _ in model_specs:
        info = count_first_layer_params(cls())
        print(f"{name:<26} {info['first_layer_learnable']:>10,} "
              f"{info['first_layer_frozen']:>11,} {info['first_layer_total']:>10,} "
              f"{info['model_total']:>12,} {info['first_layer_ratio']:>9.1%}")

    # 4. Scaling growth rates
    print(f"\n4. SCALING GROWTH RATES")
    growth_rates = {}
    n_per_dim = 8
    print(f"\n  Using n={n_per_dim} discretization points per dimension:")
    print(f"\n  {'Dimension':>10} {'Discretized':>12} {'Algebraic':>10} {'Ratio':>8}")
    print(f"  {'-'*42}")
    for d in range(1, 7):
        disc_count = n_per_dim ** d
        alg_count = d + d * (d + 1) // 2 + 1
        ratio = disc_count / alg_count
        growth_rates[str(d)] = {'discretized': disc_count, 'algebraic': alg_count, 'ratio': ratio}
        print(f"  {d:>10} {disc_count:>12,} {alg_count:>10} {ratio:>7.0f}x")
    results['growth_rates'] = growth_rates

    total_time = time.time() - t_start
    print(f"\nTotal time: {total_time:.1f}s")

    os.makedirs(os.path.join(PROJECT_ROOT, 'data'), exist_ok=True)
    output = {'config': {'experiment': 'scaling_comparison', 'total_time_seconds': total_time}, 'results': results}
    with open(os.path.join(PROJECT_ROOT, 'data', 'scaling_comparison.json'), 'w') as f:
        json.dump(output, f, indent=2, default=_json_default)
    print(f"\nResults saved to data/scaling_comparison.json")


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if obj == float('inf'):
        return "inf"
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


if __name__ == '__main__':
    run_all()
