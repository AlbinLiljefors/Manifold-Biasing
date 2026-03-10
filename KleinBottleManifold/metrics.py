# Efficiency metrics for topological CNN first-layer comparison
import numpy as np


def count_first_layer_params(model):
    fl_learnable = 0
    fl_frozen = 0
    if hasattr(model, 'first_layer'):
        for p in model.first_layer.parameters():
            if p.requires_grad:
                fl_learnable += p.numel()
            else:
                fl_frozen += p.numel()
        for buf in model.first_layer.buffers():
            fl_frozen += buf.numel()

    model_learnable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    for buf in model.buffers():
        model_frozen += buf.numel()

    fl_total = fl_learnable + fl_frozen
    model_total = model_learnable + model_frozen

    return {
        'first_layer_learnable': fl_learnable,
        'first_layer_frozen': fl_frozen,
        'first_layer_total': fl_total,
        'model_learnable': model_learnable,
        'model_frozen': model_frozen,
        'model_total': model_total,
        'first_layer_ratio': fl_total / model_total if model_total > 0 else 0.0,
    }


def compute_convergence_metrics(learning_curve, thresholds=None):
    if thresholds is None:
        thresholds = [90, 95]

    if not learning_curve:
        return {f'images_to_{t}': None for t in thresholds}

    images = [pt[0] for pt in learning_curve]
    accs = [pt[1] for pt in learning_curve]

    result = {}
    for t in thresholds:
        reached = None
        for i, acc in enumerate(accs):
            if acc >= t:
                if i == 0:
                    reached = images[0]
                else:
                    prev_acc = accs[i - 1]
                    if prev_acc >= t:
                        reached = images[i - 1]
                    else:
                        frac = (t - prev_acc) / (acc - prev_acc)
                        reached = images[i - 1] + frac * (images[i] - images[i - 1])
                break
        result[f'images_to_{t}'] = reached

    if len(images) >= 2:
        auc = float(np.trapezoid(accs, images))
        total_range = images[-1] - images[0]
        result['auc'] = auc / total_range if total_range > 0 else 0.0
    else:
        result['auc'] = accs[0] if accs else 0.0

    return result


def compute_efficiency_ratios(accuracy, noise_accuracy, first_layer_params):
    if first_layer_params == 0:
        return {
            'acc_per_param': float('inf') if accuracy > 10 else 0.0,
            'noise_acc_per_param': float('inf') if noise_accuracy > 10 else 0.0,
        }
    return {
        'acc_per_param': (accuracy - 10.0) / first_layer_params,
        'noise_acc_per_param': (noise_accuracy - 10.0) / first_layer_params,
    }
