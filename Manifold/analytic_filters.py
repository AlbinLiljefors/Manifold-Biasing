# Closed-form filter generators for topological CNN first layers.
# Both apply Love's post-processing: mean-subtract, L2-normalize, scale by m*STD.
import math
import numpy as np

_STD = 0.1


def _integrate_clamped_linear(r, m, x_i, y_i):
    """Integrate clamp(r*(x - m/2) + m/2 - y_i, 0, 1) from x_i to x_i+1."""
    c = -r * m / 2.0 + m / 2.0 - y_i
    lo, hi = float(x_i), float(x_i + 1)

    if abs(r) < 1e-15:
        return max(0.0, min(1.0, c)) * (hi - lo)

    x_zero = -c / r
    x_one = (1.0 - c) / r

    breakpoints = [lo]
    for bp in sorted([x_zero, x_one]):
        if lo < bp < hi:
            breakpoints.append(bp)
    breakpoints.append(hi)

    total = 0.0
    for i in range(len(breakpoints) - 1):
        a, b = breakpoints[i], breakpoints[i + 1]
        g_mid = r * (a + b) / 2.0 + c
        if g_mid <= 0:
            pass
        elif g_mid >= 1:
            total += b - a
        else:
            total += r * (b**2 - a**2) / 2.0 + c * (b - a)
    return total


def analytic_primary_circle(n_filters=64, m=3):
    """Reproduce Love's primary_circle filters via closed-form integration."""
    filters = np.zeros((n_filters, m, m))

    for w in range(n_filters):
        theta = w * 2 * math.pi / n_filters
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        r = sin_t / cos_t if abs(cos_t) > 1e-15 else math.copysign(1e16, sin_t)
        sign = np.sign(r) or 1.0

        M = np.zeros((m, m))
        for x_i in range(m):
            for y_i in range(m):
                M[y_i, x_i] = sign * _integrate_clamped_linear(r, m, x_i, y_i)

        if theta > math.pi:
            M = np.rot90(M, 2)

        M = M.flatten() - np.mean(M)
        norm = np.linalg.norm(M)
        if norm > 1e-12:
            M = M / norm
        filters[w] = M.reshape(m, m)

    return float(m) * _STD * filters


def directional_derivative_filters(n_filters=64, m=3):
    """Fused cos(theta)*I + sin(theta)*J as single m x m kernels."""
    assert m % 2 == 1
    half = m // 2

    # Least-squares gradient kernels
    I_kernel = np.zeros((m, m))
    J_kernel = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            I_kernel[i, j] = i - half
            J_kernel[i, j] = j - half
    norm_factor = m * sum((k - (m - 1) / 2.0) ** 2 for k in range(m))
    I_kernel /= norm_factor
    J_kernel /= norm_factor

    filters = np.zeros((n_filters, m, m))
    for k in range(n_filters):
        theta = k * 2 * math.pi / n_filters
        F_k = math.cos(theta) * I_kernel + math.sin(theta) * J_kernel
        F_k = F_k - np.mean(F_k)
        norm = np.linalg.norm(F_k)
        if norm > 1e-12:
            F_k = F_k / norm
        filters[k] = F_k

    return float(m) * _STD * filters
