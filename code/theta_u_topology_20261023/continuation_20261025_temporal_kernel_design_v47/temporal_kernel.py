"""V47 design-only algebraic temporal-moment kernel, not a new filter.

No setup, numerical call or experiment occurs on import.  The caller retains
V46 prediction, logprior, CAVI, hyperparameters, stopping rule and point outputs.
Only the implementation of each already-declared old/current edge joint's
moments is replaced.  Numeric qualification and cost measurement require a
separate future frozen finite protocol; this module contains no benchmark.
"""
from __future__ import annotations

import math
import time
import numpy as np
from numba import njit


@njit(cache=False, fastmath=False)
def _edge_temporal_centered(logold, logtransition, logprior, q, values,
                            current_mean, current_var):
    m, k = q.shape
    old_mean = np.empty(m)
    old_var = np.empty(m)
    cross = np.empty(m)
    delta_mean = np.empty(m)
    delta_var = np.empty(m)
    # These three O(K) buffers are reused for all edges and columns.  The
    # already-supplied transition input remains K-by-K; no temporary joint is.
    conditional_mean = np.empty(k)
    conditional_var = np.empty(k)
    weights = np.empty(k)
    current_marginal_error = 0.
    conditional_normalization_error = 0.
    formula_error = 0.
    covariance_determinant_minimum = np.inf
    for e in range(m):
        for j in range(k):
            if not math.isfinite(logprior[e, j]):
                if q[e, j] != 0.:
                    raise FloatingPointError('Unreachable current node acquired mass')
                conditional_mean[j] = 0.
                conditional_var[j] = 0.
                continue
            maximum = -np.inf
            for i in range(k):
                score = logold[e, i]+logtransition[i, j]
                maximum = max(maximum, score)
            if not math.isfinite(maximum):
                raise FloatingPointError('Unresolved reachable backward conditional')
            total = 0.
            mean = 0.
            centered_sum = 0.
            for i in range(k):
                score = logold[e, i]+logtransition[i, j]
                weight = math.exp(score-maximum)
                weights[i] = weight
                if weight > 0.:
                    following_total = total+weight
                    difference = values[i]-mean
                    following_mean = mean+(weight/following_total)*difference
                    centered_sum += weight*difference*(values[i]-following_mean)
                    mean = following_mean
                    total = following_total
            if total <= 0. or not math.isfinite(total):
                raise FloatingPointError('Unresolved reachable backward conditional')
            # The supplied predictive logarithm is deliberately not recomputed
            # or used to change q.  This is the raw V46 conditional-normalizer
            # diagnostic, calculated stably from the same conditional scores.
            log_raw_sum = (maximum-logprior[e, j])+math.log(total)
            raw_sum = math.exp(log_raw_sum)
            if raw_sum <= 0. or not math.isfinite(raw_sum):
                raise FloatingPointError('Unresolved reachable backward conditional')
            raw_error = abs(math.expm1(log_raw_sum))
            conditional_normalization_error = max(conditional_normalization_error,
                                                   raw_error)
            conditional_mean[j] = mean
            conditional_var[j] = centered_sum/total
            # Actually sum the normalized current marginal; do not set its
            # numerical error to zero merely because the identity is algebraic.
            represented_current_mass = 0.
            for i in range(k):
                represented_current_mass += (weights[i]/total)*q[e, j]
            current_marginal_error = max(current_marginal_error,
                abs(represented_current_mass-q[e, j]))
        smoothed_mean = 0.
        for j in range(k):
            smoothed_mean += q[e, j]*conditional_mean[j]
        old_mean[e] = smoothed_mean
        delta_mean[e] = current_mean[e]-smoothed_mean
        smoothed_var = 0.
        covariance = 0.
        physical_delta_var = 0.
        for j in range(k):
            old_difference = conditional_mean[j]-smoothed_mean
            current_difference = values[j]-current_mean[e]
            delta_difference = values[j]-conditional_mean[j]-delta_mean[e]
            smoothed_var += q[e, j]*(conditional_var[j]+old_difference**2)
            covariance += q[e, j]*current_difference*old_difference
            physical_delta_var += q[e, j]*(conditional_var[j]+delta_difference**2)
        old_var[e] = smoothed_var
        cross[e] = covariance
        delta_var[e] = physical_delta_var
        identity = current_var[e]+smoothed_var-2.*covariance
        formula_error = max(formula_error, abs(identity-physical_delta_var))
        determinant = current_var[e]*smoothed_var-covariance**2
        covariance_determinant_minimum = min(covariance_determinant_minimum,
                                             determinant)
    return (old_mean, old_var, cross, delta_mean, delta_var,
            current_marginal_error, conditional_normalization_error,
            formula_error, covariance_determinant_minimum)


def _finite(value, name):
    if not np.all(np.isfinite(value)):
        raise FloatingPointError('Nonfinite temporal-kernel '+name)


def temporal_moments(logold, logtransition, logprior, q, values):
    """Return the identical V46 temporal_moments field contract.

    The output moments are algebraically equivalent, with different floating
    summation order.  q and the caller's posterior/point history are read only.
    t0 remains the caller's explicit NA branch and must not call this function.
    """
    logold = np.asarray(logold, dtype=float)
    logtransition = np.asarray(logtransition, dtype=float)
    logprior = np.asarray(logprior, dtype=float)
    q = np.asarray(q, dtype=float)
    values = np.asarray(values, dtype=float)
    if q.ndim != 2:
        raise ValueError('Edge-by-node posterior array required')
    m, k = q.shape
    if (m < 1 or k < 1 or logold.shape != (m, k) or logprior.shape != (m, k)
            or logtransition.shape != (k, k) or values.shape != (k,)):
        raise ValueError('Temporal-kernel edge/state dimensions differ')
    _finite(q, 'current probabilities'); _finite(values, 'node values')
    for name, logs in [('old logarithms', logold), ('transition logarithms', logtransition),
                       ('predictive logarithms', logprior)]:
        if np.any(np.isnan(logs)) or np.any(np.isposinf(logs)):
            raise FloatingPointError('Invalid temporal-kernel '+name)
    # Keep these two NumPy expressions identical to V46's current endpoint.
    # They are not folded into the compiled kernel or approximated.
    current_mean = q @ values
    current_var = np.sum(q*(values[None, :]-current_mean[:, None])**2, axis=1)
    _finite(current_mean, 'current mean'); _finite(current_var, 'current variance')
    if np.any(current_var < 0.):
        raise FloatingPointError('Negative centered atomic variance')
    (old_mean, old_var, cross, delta_mean, delta_var,
     current_marginal_error, conditional_normalization_error,
     formula_error, covariance_determinant_minimum) = _edge_temporal_centered(
         np.ascontiguousarray(logold), np.ascontiguousarray(logtransition),
         np.ascontiguousarray(logprior), np.ascontiguousarray(q),
         np.ascontiguousarray(values), np.ascontiguousarray(current_mean),
         np.ascontiguousarray(current_var))
    for name, value in [('old mean', old_mean), ('old variance', old_var),
                        ('cross covariance', cross), ('delta mean', delta_mean),
                        ('delta variance', delta_var)]:
        _finite(value, name)
    _finite((current_marginal_error, conditional_normalization_error,
             formula_error, covariance_determinant_minimum), 'diagnostics')
    # Preserve V46's existing moment gates.  No floor, jitter, clipping, new
    # convergence criterion or stronger conditional-normalizer gate is added.
    if (np.any(old_var < 0.) or np.any(delta_var < 0.)
            or covariance_determinant_minimum < -1e-12
            or current_marginal_error > 1e-12 or formula_error > 1e-10):
        raise FloatingPointError('Edge-wise temporal joint moment qualification failed')
    return dict(old_smoothed_mean=old_mean, old_smoothed_variance=old_var,
        cross_diag=cross, delta_mean=delta_mean, delta_variance=delta_var,
        temporal_current_marginal_error=float(current_marginal_error),
        temporal_conditional_normalization_error=float(conditional_normalization_error),
        temporal_delta_formula_error=float(formula_error),
        temporal_covariance_determinant_minimum=float(covariance_determinant_minimum))


def signature_inventory():
    return dict(temporal_centered_welford=[str(s)
                for s in _edge_temporal_centered.signatures])


def setup():
    """Define one artificial m3/K3 warmup; the owning future parent calls it."""
    started = time.perf_counter()
    values = np.asarray([0., .3, .7])
    old = np.asarray([[.3, .4, .3], [.2, .5, .3], [.4, .1, .5]])
    transition = np.asarray([[.8, .1, .1], [.1, .85, .05], [.1, .05, .85]])
    logold, logtransition = np.log(old), np.log(transition)
    prior = old @ transition
    logprior = np.log(prior)
    logits = logprior+np.asarray([[0., .1, -.2], [.2, -.1, .3], [-.1, .4, .1]])
    logits = logits-np.max(logits, axis=1, keepdims=True)
    q = np.exp(logits)
    q /= np.sum(q, axis=1, keepdims=True)
    result = temporal_moments(logold, logtransition, logprior, q, values)
    return dict(artificial_only=True, m=3, K=3, RNG_used=False,
        scientific_observations_used=False, cache=False, fastmath=False,
        parallel=False, wall_seconds=float(time.perf_counter()-started),
        signatures=signature_inventory(),
        temporal_current_marginal_error=result['temporal_current_marginal_error'],
        temporal_delta_formula_error=result['temporal_delta_formula_error'])
