# Closed-form filter generators for topological CNN first layers
import math
import numpy as np


_STD = 0.1


def klein_bottle_scaled_filters_nonuniform(num_th1=8, num_th2=8, m=3):
    """Klein bottle scaled filters with sin^2(theta2) weighted sampling."""
    assert m % 2 == 1
    n_filters = num_th1 * num_th2

    NORM_Q2 = 2.0 / math.sqrt(3.0)
    NORM_Q3 = 1.0 / math.sqrt(5.0)

    half = m // 2
    positions = [(i - half, j - half) for i in range(m) for j in range(m)]
    X = np.array([[1, ci, cj, ci**2, ci*cj, cj**2] for ci, cj in positions])
    XtX_inv = np.linalg.inv(X.T @ X)
    K_all = XtX_inv @ X.T
    K_Ix  = K_all[1].reshape(m, m)
    K_Iy  = K_all[2].reshape(m, m)
    K_Ixx = K_all[3].reshape(m, m)
    K_Ixy = K_all[4].reshape(m, m)
    K_Iyy = K_all[5].reshape(m, m)

    # sin^2 weighted theta2 via inverse CDF sampling
    n_sample = 1000
    t_fine = np.linspace(0, 2 * math.pi, n_sample, endpoint=False)
    weights = np.sin(t_fine) ** 2
    weights = weights / weights.sum()
    cdf = np.cumsum(weights)
    cdf = cdf / cdf[-1]
    quantiles = np.linspace(0.5 / num_th2, 1.0 - 0.5 / num_th2, num_th2)
    theta2_values = np.interp(quantiles, cdf, t_fine)

    filters = np.zeros((n_filters, m, m))
    idx = 0
    for ti in range(num_th1):
        theta1 = ti * math.pi / num_th1
        cos1 = math.cos(theta1)
        sin1 = math.sin(theta1)
        for tj in range(num_th2):
            theta2 = theta2_values[tj]
            cos2 = math.cos(theta2)
            sin2 = math.sin(theta2)

            w_Ix  = sin2 * NORM_Q2 * cos1
            w_Iy  = sin2 * NORM_Q2 * sin1
            w_Ixx = cos2 * NORM_Q3 * cos1**2
            w_Ixy = cos2 * NORM_Q3 * 2 * cos1 * sin1
            w_Iyy = cos2 * NORM_Q3 * sin1**2

            F_k = (w_Ix * K_Ix + w_Iy * K_Iy +
                   w_Ixx * K_Ixx + w_Ixy * K_Ixy + w_Iyy * K_Iyy)
            F_k = F_k - np.mean(F_k)
            norm = np.linalg.norm(F_k)
            if norm > 1e-12:
                F_k = F_k / norm
            filters[idx] = F_k
            idx += 1

    return float(m) * _STD * filters
